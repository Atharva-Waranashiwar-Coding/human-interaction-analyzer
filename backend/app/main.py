from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .api.understanding import router as understanding_router
from .database import Base, engine, SessionLocal, initialize_database
from .models import Video

initialize_database()
# One backend process is supported; recover jobs interrupted by a restart.
with SessionLocal() as db:
    db.query(Video).filter(Video.analysis_status == "processing").update({"analysis_status": "failed", "status": "error", "status_message": "Analysis interrupted by backend restart; click Analyze to retry."})
    db.commit()

app = FastAPI(title="Human Interaction Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(understanding_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
