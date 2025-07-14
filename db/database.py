
import os
from typing import Generator

from sqlmodel import create_engine, Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use SQLite for local development due to system restrictions
DATABASE_URL = "sqlite:///./docreader.db"

engine = create_engine(DATABASE_URL, echo=True)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

