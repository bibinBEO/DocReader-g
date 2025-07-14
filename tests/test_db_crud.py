
import pytest
from sqlmodel import Session, SQLModel, create_engine, select
from db.models import Document, Extraction, ModelVersion
from datetime import datetime

# In-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///test.db"
engine = create_engine(TEST_DATABASE_URL, echo=False)

@pytest.fixture(name="session")
def session_fixture():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(name="test_model_version")
def test_model_version_fixture(session: Session):
    model_version = ModelVersion(
        model_name="LayoutLMv3",
        version="1.0.0",
        path="/models/layoutlmv3_v1.0.0",
        metrics={"f1": 0.95}
    )
    session.add(model_version)
    session.commit()
    session.refresh(model_version)
    return model_version

def test_create_document(session: Session):
    document = Document(
        filename="test_invoice.pdf",
        file_path="/uploads/test_invoice.pdf",
        status="PENDING",
        task_id="test_task_id_123"
    )
    session.add(document)
    session.commit()
    session.refresh(document)

    assert document.id is not None
    assert document.filename == "test_invoice.pdf"
    assert document.status == "PENDING"

def test_read_document(session: Session):
    document = Document(
        filename="test_invoice.pdf",
        file_path="/uploads/test_invoice.pdf",
        status="PENDING",
        task_id="test_task_id_123"
    )
    session.add(document)
    session.commit()
    session.refresh(document)

    retrieved_document = session.get(Document, document.id)
    assert retrieved_document is not None
    assert retrieved_document.filename == "test_invoice.pdf"

def test_update_document_status(session: Session):
    document = Document(
        filename="test_invoice.pdf",
        file_path="/uploads/test_invoice.pdf",
        status="PENDING",
        task_id="test_task_id_123"
    )
    session.add(document)
    session.commit()
    session.refresh(document)

    document.status = "SUCCESS"
    session.add(document)
    session.commit()
    session.refresh(document)

    updated_document = session.get(Document, document.id)
    assert updated_document.status == "SUCCESS"

def test_delete_document(session: Session):
    document = Document(
        filename="test_invoice.pdf",
        file_path="/uploads/test_invoice.pdf",
        status="PENDING",
        task_id="test_task_id_123"
    )
    session.add(document)
    session.commit()
    document_id = document.id

    session.delete(document)
    session.commit()

    deleted_document = session.get(Document, document_id)
    assert deleted_document is None

def test_create_extraction(session: Session, test_model_version: ModelVersion):
    document = Document(
        filename="test_invoice.pdf",
        file_path="/uploads/test_invoice.pdf",
        status="SUCCESS",
        task_id="test_task_id_123"
    )
    session.add(document)
    session.commit()
    session.refresh(document)

    extraction = Extraction(
        document_id=document.id,
        model_version_id=test_model_version.id,
        extracted_data={"vendor_name": "Test Vendor", "invoice_number": "INV-001"},
        accuracy_score=0.92
    )
    session.add(extraction)
    session.commit()
    session.refresh(extraction)

    assert extraction.id is not None
    assert extraction.document_id == document.id
    assert extraction.model_version_id == test_model_version.id
    assert extraction.extracted_data["vendor_name"] == "Test Vendor"

def test_read_extraction_with_relationships(session: Session, test_model_version: ModelVersion):
    document = Document(
        filename="test_invoice.pdf",
        file_path="/uploads/test_invoice.pdf",
        status="SUCCESS",
        task_id="test_task_id_123"
    )
    session.add(document)
    session.commit()
    session.refresh(document)

    extraction = Extraction(
        document_id=document.id,
        model_version_id=test_model_version.id,
        extracted_data={"vendor_name": "Test Vendor", "invoice_number": "INV-001"},
        accuracy_score=0.92
    )
    session.add(extraction)
    session.commit()
    session.refresh(extraction)

    # Retrieve document and check its extractions
    retrieved_document = session.get(Document, document.id)
    assert len(retrieved_document.extractions) == 1
    assert retrieved_document.extractions[0].extracted_data["vendor_name"] == "Test Vendor"

    # Retrieve extraction and check its relationships
    retrieved_extraction = session.get(Extraction, extraction.id)
    assert retrieved_extraction.document.filename == "test_invoice.pdf"
    assert retrieved_extraction.model_version.model_name == "LayoutLMv3"

def test_create_model_version(session: Session):
    model_version = ModelVersion(
        model_name="Donut",
        version="0.1.0",
        path="/models/donut_v0.1.0",
        metrics={"f1": 0.90, "latency": 0.5}
    )
    session.add(model_version)
    session.commit()
    session.refresh(model_version)

    assert model_version.id is not None
    assert model_version.model_name == "Donut"
    assert model_version.metrics["f1"] == 0.90

