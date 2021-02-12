import io
from typing import Dict, Literal, Optional
import fastapi
import pydantic

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
    data: str
    input: Literal["conllu", "horizontal"] = "conllu"
    model: str = default_model
    output: Literal["conllu"] = "conllu"
    parser: str = ""
    tagger: str = ""
    tokenizer: Optional[str] = None


app = fastapi.FastAPI()


@app.get("/models")
async def get_model():
    return {
        "models": {m: ["tagger", "parser"] for m in models.keys()},
        "default_model": default_model,
    }


@app.post("/process")
async def process(req: ParseRequest):
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
    return {"model": model_name, "acknowledgements": "", "result": parsed.getvalue()}
