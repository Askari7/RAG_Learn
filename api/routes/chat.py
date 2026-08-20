from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from main import compiled_graph
router = APIRouter()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", status_code=status.HTTP_200_OK)
def chat(request: ChatRequest)-> ChatResponse:
    """
    Chat endpoint to handle chat requests.
    Returns a simple message indicating the chat endpoint is working.
    """
    try:
        question = request.question
        response = compiled_graph.invoke({"question": question})
        return ChatResponse(response=response["answer"])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))