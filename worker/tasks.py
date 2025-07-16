import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
import time
import json # Import json for Donut output parsing

from celery import Celery
from dotenv import load_dotenv
from PIL import Image
from sqlmodel import Session, select
from opentelemetry import trace

from pdf2image import convert_from_path
from pipeline.prepare import convert_pdf_to_images, deskew_image, ocr_image
from pipeline.models.donut import DonutModel # Import DonutModel
from pipeline.models.layoutlm_invoice import LayoutLMv3Invoice
from db.database import get_session
from db.models import Document, Extraction, ModelVersion
from utils.observability import configure_opentelemetry, DOC_PROCESSING_TIME, DOC_STATUS_COUNTER

# Load environment variables
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")

REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

celery_app = Celery(
    "docreader_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"]
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1,
)

# Configure OpenTelemetry for Celery
configure_opentelemetry(celery_app=celery_app)

# Initialize models to None. They will be loaded once per worker process when the first task is executed.
donut_model_client = None
layoutlm_client = None

@celery_app.task(bind=True)
def process_document_task(self, file_path: str, document_id: int) -> Dict[str, Any]:
    """
    Celery task to process a document, extract fields, and store results.
    """
    global donut_model_client, layoutlm_client

    # Lazily initialize models on the first task execution per worker
    if donut_model_client is None:
        donut_model_client = DonutModel(model_name_or_path=os.getenv("DONUT_MODEL_PATH", "naver-clova-ix/donut-base-finetuned-docvqa"))
    if layoutlm_client is None:
        layoutlm_client = LayoutLMv3Invoice(model_name_or_path=os.getenv("LAYOUTLMV3_MODEL_PATH"))
    start_time = time.time()
    session = next(get_session()) # Get a database session
    tracer = trace.get_tracer(__name__)

    try:
        with tracer.start_as_current_span("process_document_task") as span:
            span.set_attribute("document.id", document_id)
            span.set_attribute("document.file_path", file_path)

            # Retrieve the document from the database
            document = session.get(Document, document_id)
            if not document:
                raise ValueError(f"Document with ID {document_id} not found in database.")

            document.status = "PROCESSING"
            session.add(document)
            session.commit()
            session.refresh(document)

            self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Starting processing'})

            doc_path = Path(file_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Document not found at {file_path}")

            # 1. Pre-processing: Convert PDF to images, deskew
            with tracer.start_as_current_span("preprocessing"):
                self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'status': 'Converting to images and deskewing'})
                images: List[Image.Image] = []
                if doc_path.suffix.lower() == '.pdf':
                    images = convert_from_path(str(doc_path))
                else:
                    images.append(Image.open(doc_path).convert("RGB"))

                processed_images = [deskew_image(img) for img in images]

            all_extracted_data = []
            for i, img in enumerate(processed_images):
                with tracer.start_as_current_span(f"page_processing_page_{i+1}"):
                    self.update_state(state='PROGRESS', meta={'current': 50 + i*10, 'total': 100, 'status': f'Running model inference on page {i+1}'})
                    
                    # Use Donut for end-to-end extraction
                    extracted_fields = donut_model_client.predict(img)

                    

                    all_extracted_data.append(extracted_fields) # For now, just append raw extracted fields

            # 4. Store results in DB
            with tracer.start_as_current_span("store_results_in_db"):
                self.update_state(state='PROGRESS', meta={'current': 90, 'total': 100, 'status': 'Storing results'})

                # For now, we'll assume a dummy model version ID. This will be properly managed later.
                # In a real scenario, you'd query for the model_version based on name/version or create if not exists.
                # Using Donut model version for this extraction
                model_version_name = os.getenv("DONUT_MODEL_PATH", "naver-clova-ix/donut-base-finetuned-docvqa")
                dummy_model_version = session.exec(select(ModelVersion).where(ModelVersion.model_name == model_version_name, ModelVersion.version == "1.0.0")).first()
                if not dummy_model_version:
                    dummy_model_version = ModelVersion(
                        model_name=model_version_name,
                        version="1.0.0",
                        path=model_version_name,
                        metrics={}
                    )
                    session.add(dummy_model_version)
                    session.commit()
                    session.refresh(dummy_model_version)

                extraction = Extraction(
                    document_id=document.id,
                    model_version_id=dummy_model_version.id,
                    extracted_data={"pages": all_extracted_data}
                )
                session.add(extraction)
                session.commit()
                session.refresh(extraction)

                document.status = "SUCCESS"
                session.add(document)
                session.commit()
                session.refresh(document)

            # Clean up the uploaded file after processing
            doc_path.unlink(missing_ok=True)

            self.update_state(state='SUCCESS', meta={'current': 100, 'total': 100, 'status': 'Processing complete', 'result': all_extracted_data})
            DOC_PROCESSING_TIME.observe(time.time() - start_time) # Observe processing time
            DOC_STATUS_COUNTER.labels(status='SUCCESS').inc() # Increment success counter
            return {"status": "SUCCESS", "document_id": document.id, "extracted_data": all_extracted_data}

    except Exception as e:
        if 'document' in locals() and document:
            document.status = "FAILED"
            session.add(document)
            session.commit()
            session.refresh(document)
        self.update_state(state='FAILURE', meta={'current': 100, 'total': 100, 'status': f'Processing failed: {str(e)}'})
        DOC_STATUS_COUNTER.labels(status='FAILED').inc() # Increment failed counter
        return {"status": "FAILURE", "document_id": document_id, "error": str(e)}
    finally:
        session.close()