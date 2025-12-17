from fastapi import FastAPI
import logging
import config

from routers.health import router as health_router

def create_app() -> FastAPI:
    config.load_config()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    app = FastAPI()
    app.include_router(health_router)

    return app
