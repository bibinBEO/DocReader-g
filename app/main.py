from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Dict, Any
from pathlib import Path
import uuid
import os
import shutil
from sqlmodel import Session, select
from prometheus_client import generate_latest

from dotenv import load_dotenv

from worker.tasks import process_document_task
from worker.celery_app import celery_app
from db.database import get_session
from db.models import Document, Extraction, ModelVersion
from utils.observability import configure_opentelemetry, DOC_UPLOAD_COUNTER, DOC_PROCESSING_TIME, DOC_STATUS_COUNTER

# Load environment variables
load_dotenv()

app = FastAPI(
    title="DocReader API",
    description="API for ingesting and extracting data from documents.",
    version="0.1.0",
)

# Configure OpenTelemetry for FastAPI
configure_opentelemetry(app=app)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Placeholder for JWT authentication (Phase 5)
async def verify_jwt_token(request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    # In a real scenario, decode and validate the JWT token here
    # For now, any non-empty token starting with "Bearer " is considered valid.
    return True

@app.post("/upload", summary="Upload a document for processing")
async def upload_document(file: UploadFile = File(...), session: Session = Depends(get_session), authenticated: bool = Depends(verify_jwt_token)) -> Dict[str, str]:
    """
    Uploads a document (PDF or image) for asynchronous processing.
    Returns a task ID to check the processing status.
    """
    DOC_UPLOAD_COUNTER.inc() # Increment upload counter

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")

    # Generate a unique filename to avoid collisions
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_location = UPLOAD_DIR / unique_filename

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not save file: {e}")

    # Create a new document entry in the database
    document = Document(
        filename=file.filename,
        file_path=str(file_location),
        status="PENDING"
    )
    session.add(document)
    session.commit()
    session.refresh(document)

    # Enqueue the processing task
    task = process_document_task.delay(str(file_location), document.id)

    # Update the document with the Celery task ID
    document.task_id = task.id
    session.add(document)
    session.commit()
    session.refresh(document)

    return JSONResponse({"message": "Document uploaded and processing started", "task_id": task.id, "document_id": document.id})

@app.get("/status/{task_id}", summary="Get the status of a document processing task")
async def get_task_status(task_id: str, session: Session = Depends(get_session), authenticated: bool = Depends(verify_jwt_token)) -> Dict[str, Any]:
    """
    Retrieves the current status and progress of a document processing task.
    """
    task = celery_app.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {
            'status': task.state,
            'message': 'Task is pending or not found',
            'progress': 0
        }
    elif task.state == 'PROGRESS':
        response = {
            'status': task.state,
            'message': task.info.get('status', 'Processing...'),
            'progress': task.info.get('current', 0)
        }
    elif task.state == 'SUCCESS':
        response = {
            'status': task.state,
            'message': 'Task completed successfully',
            'progress': 100,
            'result': task.info.get('extracted_data', None) # Extracted data will be here
        }
        DOC_STATUS_COUNTER.labels(status='SUCCESS').inc() # Increment success counter
    elif task.state == 'FAILURE':
        response = {
            'status': task.state,
            'message': task.info.get('status', 'Task failed'),
            'progress': 100,
            'error': str(task.info.get('error', 'Unknown error'))
        }
        DOC_STATUS_COUNTER.labels(status='FAILED').inc() # Increment failed counter
    else:
        response = {
            'status': task.state,
            'message': 'Unknown task state',
            'progress': 0
        }
    return JSONResponse(response)

@app.get("/result/{document_id}", summary="Get extracted results for a document")
async def get_document_result(document_id: int, session: Session = Depends(get_session), authenticated: bool = Depends(verify_jwt_token)) -> Dict[str, Any]:
    """
    Retrieves the final extracted data for a given document ID from the database.
    """
    document = session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # Eagerly load extractions and model_version
    statement = select(Document).where(Document.id == document_id).options(
        selectinload(Document.extractions).selectinload(Extraction.model_version_rel) # Corrected relationship name
    )
    document = session.exec(statement).first()

    if not document or not document.extractions:
        return JSONResponse({"message": f"No extraction results found for document ID {document_id}. Status: {document.status}", "data": {}})

    # Assuming you want the latest extraction or all of them
    latest_extraction = document.extractions[-1] # Or iterate to find specific one

    return JSONResponse({
        "document_id": document.id,
        "filename": document.filename,
        "status": document.status,
        "extracted_data": latest_extraction.extracted_data,
        "model_version": {
            "name": latest_extraction.model_version_rel.model_name, # Corrected relationship name
            "version": latest_extraction.model_version_rel.version, # Corrected relationship name
            "metrics": latest_extraction.model_version_rel.metrics # Corrected relationship name
        } if latest_extraction.model_version_rel else None,
        "extraction_time": latest_extraction.extraction_time.isoformat()
    })

@app.get("/metrics", summary="Prometheus metrics endpoint")
async def metrics():
    """
    Exposes Prometheus metrics.
    """
    return PlainTextResponse(generate_latest().decode('utf-8'))