from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os

from fastapi import FastAPI, Body, HTTPException, Request
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


GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "45"))
gemini_executor = ThreadPoolExecutor(max_workers=2)


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

    try:
        future = gemini_executor.submit(get_answer_from_gemini, prompt)
        answer = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
    except FutureTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "Gemini response timeout. Try again or reduce prompt size."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini request failed: {str(exc)}",
        ) from exc

    add_request_data(
        ip_address=user_ip_address,
        prompt=prompt,
        response=answer,
    )
    return {"data": answer}
