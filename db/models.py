
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlmodel import Field, SQLModel, Relationship
from sqlalchemy import Column
from sqlalchemy.types import JSON # Use generic JSON type for SQLModel to handle serialization


class DocumentBase(SQLModel):
    filename: str
    file_path: str
    upload_time: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    status: str = Field(default="PENDING", nullable=False) # PENDING, PROCESSING, SUCCESS, FAILED
    task_id: Optional[str] = Field(index=True) # Celery task ID

class Document(DocumentBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    extractions: List["Extraction"] = Relationship(back_populates="document")

class ExtractionBase(SQLModel):
    model_version_id: Optional[int] = Field(default=None, foreign_key="modelversion.id")
    extracted_data: Dict[str, Any] = Field(sa_column=Column(JSON), default={}) # Explicitly use Column(JSON)
    extraction_time: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    accuracy_score: Optional[float] = None # Field-level F1 score for this extraction

class Extraction(ExtractionBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: Optional[int] = Field(default=None, foreign_key="document.id")

    document: Optional[Document] = Relationship(back_populates="extractions")
    model_version_rel: Optional["ModelVersion"] = Relationship(back_populates="extractions")

class ModelVersionBase(SQLModel):
    model_name: str
    version: str = Field(index=True)
    path: str # Path to the model artifact or identifier
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    metrics: Optional[Dict[str, Any]] = Field(sa_column=Column(JSON), default={}) # Explicitly use Column(JSON)

class ModelVersion(ModelVersionBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    extractions: List[Extraction] = Relationship(back_populates="model_version_rel")

