from fastapi import APIRouter, status

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify the API is running.
    Returns a simple message indicating the API is healthy.
    """
    return {"status": "healthy"}