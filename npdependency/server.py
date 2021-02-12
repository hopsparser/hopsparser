import fastapi
import pydantic


class Settings(pydantic.BaseSettings):
    device: str = "cpu"
    model: str


settings = Settings()

app = fastapi.FastAPI()


@app.get("/models")
def get_model():
    return {
        "models": {settings.model: ["tagger", "parser"]},
        "default_model": settings.model,
    }
