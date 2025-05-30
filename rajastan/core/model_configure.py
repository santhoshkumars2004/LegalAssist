from pydantic import BaseModel, Field
from typing import Optional

class Document(BaseModel):
    page_content: str  # Example field for the content of a document page
    metadata: dict  # Example field for metadata, you can define more specific fields

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document: {self.page_content[:50]}..."

    def add_metadata(self, key: str, value: Optional[str] = None):
        """Add metadata to the document."""
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value based on the key."""
        return self.metadata.get(key)
