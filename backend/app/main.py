
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.comparison import router as comparison_router
from app.api.damage import router as damage_router
from app.api.nlp import router as nlp_router
from app.api.hybrid_comparison import router as hybrid_router
from app.api.inspection import router as inspection_router

app = FastAPI(
    title="AI Assessment API",
    description="FastAPI backend for AI/ML assessment tasks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(nlp_router, prefix="/api/nlp", tags=["NLP"])


app.include_router(
    comparison_router,
    prefix="/api/comparison",
    tags=["Image Comparison"]
)

app.include_router(
    hybrid_router,
    prefix="/api/comparison",
    tags=["Hybrid Damage Comparison"]
)

app.include_router(
    damage_router,
    prefix="/api/damage",
    tags=["Building Damage Detection"]
)

app.include_router(
    inspection_router,
    prefix="/api/inspection",
    tags=["AI Inspection & Severity"]
)




@app.get("/")
def home():
    return {"message": "AI Assessment API is running"}


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "AI Assessment"}
