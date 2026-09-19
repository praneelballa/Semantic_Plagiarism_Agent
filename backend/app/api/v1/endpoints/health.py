from typing import Dict
from fastapi import APIRouter, status
from app.config import settings

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="System Health & Diagnostic Check",
    response_description="System uptime, status, and environment confirmation",
)
def get_health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "project": settings.PROJECT_NAME,
        "api_version": "v1",
    }