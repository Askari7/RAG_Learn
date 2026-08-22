from fastapi import APIRouter, status
from app.usage_store import get_usage_totals

router = APIRouter()

@router.get("/usage", status_code=status.HTTP_200_OK)
def get_usage():
    """
    Returns cumulative token usage and estimated cost, persisted in Postgres
    across all requests and server instances.
    """
    return get_usage_totals()
