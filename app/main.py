from fastapi import FastAPI
from app.core.config import settings
from app.api.main import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION
)

app.include_router(api_router)
