from pydantic import BaseModel

class SuggestDTO(BaseModel):
    id: str
    suggest: str