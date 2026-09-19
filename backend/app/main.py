from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.config import settings
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up %s...", settings.PROJECT_NAME)
    logger.info("Environment: %s", settings.ENVIRONMENT)
    logger.info("Storage Data Dir: %s", settings.DATA_DIR)
    yield
    logger.info("Shutting down %s...", settings.PROJECT_NAME)


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)


@app.get("/", tags=["Root"])
def root() -> Dict[str, str]:
    return {
        "message": "Semantic Plagiarism Agent API is running",
        "status": "healthy",
    }


if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health", tags=["Root Health"])
def root_health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "project": settings.PROJECT_NAME,
    }


app.include_router(api_router, prefix=settings.API_V1_STR)