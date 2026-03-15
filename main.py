from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware

from db import Base, add_request_data, engine, get_user_requests
from gemini_client import get_answer_from_gemini


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    print("Database tables created")
    yield

app = FastAPI(
    title="Gemini API FastAPI Example",
    lifespan=lifespan,
)


def get_allowed_origins() -> list[str]:
    # Keep local origins and allow prod frontend domains via env variable.
    default_origins = {
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://[::1]:5500",
    }
    configured = os.getenv("FRONTEND_ORIGINS", "")
    parsed_origins = {
        origin.strip()
        for origin in configured.split(",")
        if origin.strip()
    }
    return sorted(default_origins.union(parsed_origins))


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Gemini API FastAPI Example",
        "docs": "/docs",
        "requests_endpoint": "/requests",
    }


@app.get("/requests")
def get_requests(request: Request):
    user_ip_address = request.client.host if request.client else "unknown"
    print(f"User IP address: {user_ip_address}")
    user_requests = get_user_requests(ip_address=user_ip_address)
    return user_requests


@app.post("/requests")
def send_prompt(
    request: Request,
    prompt: str = Body(embed=True),
):
    user_ip_address = request.client.host if request.client else "unknown"
    answer = get_answer_from_gemini(prompt)
    add_request_data(
        ip_address=user_ip_address,
        prompt=prompt,
        response=answer,
    )
    return {"data": answer}
