from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat
from api.routes import health
from api.routes import usage

app = FastAPI(title="P3 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat.router)
app.include_router(health.router)
app.include_router(usage.router)