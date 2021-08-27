from typing import Dict, List, Literal, Optional

import fastapi
import pydantic

from hopsparser import __version__, parser


class Settings(pydantic.BaseSettings):
    device: str = "cpu"
    models: Dict[str, str]


settings = Settings()


models = {
    model_name: parser.BiAffineParser.load(config_path).to(settings.device)
    for model_name, config_path in settings.models.items()
}
default_model = next(iter(models.keys()))


class ParseRequest(pydantic.BaseModel):
    data: str = pydantic.Field(
        ...,
        description="Input data in UTF-8.",
        example="Je reconnais l'existence du kiwi.",
    )
    input: Literal["conllu", "horizontal"] = pydantic.Field(
        "conllu", description="Input data in UTF-8."
    )
    model: str = pydantic.Field(
        default_model,
        description="The model to use to parse the data. See `models` for a list.",
    )
    output: Literal["conllu"] = pydantic.Field(
        "conllu", description="The output format to use."
    )
    parser: Optional[str] = pydantic.Field(
        None, description="Ignored (for compatibility with UDPipe.)"
    )
    tagger: Optional[str] = pydantic.Field(
        None, description="Ignored (for compatibility with UDPipe.)"
    )
    tokenizer: Optional[str] = pydantic.Field(
        None, description="Ignored (for compatibility with UDPipe.)"
    )


class ParseResponse(pydantic.BaseModel):
    model: str = pydantic.Field(..., description="The model used to parse the data.")
    acknowledgements: List[str] = pydantic.Field(
        ..., description="A list of acknowledgements for the model used."
    )
    data: str = pydantic.Field(..., description="The processed output.")


app = fastapi.FastAPI(
    title="HOPSparser REST endpoint",
    description="A honest parser of sentences.",
    version=__version__,
)


@app.get("/models", summary="Enumerate the loaded models.")
async def get_models():
    return {
        "models": {m: ["tagger", "parser"] for m in models.keys()},
        "default_model": default_model,
    }


@app.post("/process", summary="Parse data.", response_model=ParseResponse)
async def process(req: ParseRequest) -> ParseResponse:
    model_name = req.model if req.model is not None else next(iter(models.keys()))
    parser = models.get(model_name)
    if parser is None:
        raise fastapi.HTTPException(
            status_code=404,
            detail="Requested model not loaded",
        )
    parsed = "".join(
        [
            f"{tree.to_conllu()}\n\n"
            for tree in parser.parse(
                req.data.splitlines(), raw=req.input == "horizontal"
            )
        ]
    )
    return ParseResponse(model=model_name, acknowledgements=[""], data=parsed)
