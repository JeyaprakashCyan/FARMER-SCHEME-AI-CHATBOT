from typing import Optional
from pydantic import BaseModel


class DocumentMetadata(BaseModel):

    document_id: str

    document_name: str

    document_type: Optional[str] = None

    source_organization: Optional[str] = None

    source_url: Optional[str] = None

    language: str = "English"

    country: str = "India"

    publication_date: Optional[str] = None

    last_updated: Optional[str] = None

    version: Optional[str] = None

    status: str = "active"

    scheme_id: Optional[str] = None

    scheme_level: Optional[str] = "Central"

    applicable_scope: str = "India"

    state: Optional[str] = None

    district: Optional[str] = None