"""``POST /extract`` route (Task 6).

Extracts the structured after-sale intent from free text and returns it as
JSON. FastAPI serializes the ``request_type`` enum to its Chinese value.
"""

from fastapi import APIRouter

from app.schemas.extraction import AfterSaleExtraction, ExtractRequest
from app.services.extraction import extract

router = APIRouter()


@router.post("/extract", response_model=AfterSaleExtraction)
async def extract_route(req: ExtractRequest) -> AfterSaleExtraction:
    """Extract the structured after-sale intent from ``req.text``."""
    return await extract(req.text)
