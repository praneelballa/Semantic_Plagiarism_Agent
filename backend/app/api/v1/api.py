from fastapi import APIRouter
from app.api.v1.endpoints import health, corpus, analyze

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(corpus.router, prefix="/corpus", tags=["Corpus"])
api_router.include_router(analyze.router, prefix="/analyze", tags=["Analysis"])