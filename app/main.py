import uvicorn
from fastapi import FastAPI

from app.core.exc import register_exception_handlers
from app.routers import chat, health


def create_app() -> FastAPI:
    app = FastAPI(title="LangUp AI", version="0.1.0")
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8100, reload=True)
