from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import documents

app = FastAPI(
    title="Ultra Doc-Intelligence",
    description="AI-powered logistics document assistant — upload, ask, extract.",
    version="1.0.0",
)

app.include_router(documents.router)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))
