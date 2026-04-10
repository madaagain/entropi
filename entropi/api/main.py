from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from entropi.api.routes.compress import router as compress_router
from entropi.api.routes.decompress import router as decompress_router
from entropi.api.routes.auth import router as auth_router
from entropi.api.routes.keys import router as keys_router
from entropi.api.errors import setup_error_handlers
from entropi.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        init_db()
    except Exception as e:
        print(f"Warning: DB init failed: {e}")
    yield
    # shutdown — nothing to do for now


app = FastAPI(
    title="Entropi",
    description="Vector compression API built on TurboQuant",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://entropi.dev"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_error_handlers(app)
app.include_router(compress_router)
app.include_router(decompress_router)
app.include_router(auth_router)
app.include_router(keys_router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/")
def root():
    return {
        "name": "Entropi",
        "description": "Vector compression API built on TurboQuant",
        "docs": "/docs",
        "paper": "https://arxiv.org/abs/2504.19874",
    }
