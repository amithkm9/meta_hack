from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="SignAdapt — Adaptive Sign-Language Tutoring Env",
    version="1.0.0",
    description="An OpenEnv environment for adaptive sign-language tutoring planning.",
)

app.include_router(router)
