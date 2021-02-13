import io
from typing import Dict, List, Literal, Optional
import fastapi
import pydantic

from npdependency import __version__
from npdependency import deptree, graph_parser, lexers


class Settings(pydantic.BaseSettings):
    device: str = "cpu"
    models: Dict[str, str]


settings = Settings()


models = {
    model_name: graph_parser.BiAffineParser.from_config(
        config_path, overrides={"device": settings.device}
    )
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
    if req.input == "conllu":
        treebank_inpt = io.StringIO(req.data)
    elif req.input == "horizontal":
        tree_strs = []
        for line in req.data.splitlines():
            if not line or line.isspace():
                continue
            tree_strs.append(
                "\n".join(
                    f"{i}\t{w}" for i, w in enumerate(line.strip().split(), start=1)
                )
            )
        treebank_inpt = io.StringIO("\n\n".join(tree_strs))
    trees = deptree.DependencyDataset.read_conll(treebank_inpt)
    ft_dataset = lexers.FastTextDataSet(
        parser.ft_lexer, special_tokens=[deptree.DepGraph.ROOT_TOKEN]
    )
    treebank = deptree.DependencyDataset(
        trees,
        parser.lexer,
        parser.charset,
        ft_dataset,
        use_labels=parser.labels,
        use_tags=parser.tagset,
    )
    parsed = io.StringIO()
    parser.predict_batch(treebank, parsed, greedy=False)
    return ParseResponse(
        model=model_name, acknowledgements=[""], result=parsed.getvalue()
    )
