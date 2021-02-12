Use HOPSparser in server mode
=============================

For convenience, we provide a server implementing a small subset of the [UDPipe REST
API](https://lindat.mff.cuni.cz/services/udpipe/api-reference.php).

**Warning** This server mode is very basic, comes with no security expertise and has mostly been
designed with uses on a local machine in mind, so you probably shouldn't expose it on a
public-facing endpoint. For instance, it is completely trivial to use it to hog all the memory of
the host (just pass sufficiently big input data to `process`). That said, security expertise is very much welcome, so don't hesitate to file issues and send PRs.

## Run in server mode

Running

```console
hopsparser serve MODEL_PATH
```

will start a server providing a [FastAPI](https://fastapi.tiangolo.com) REST endpoint on port 8000
(configurable with --port). The API documentation will then be available at
<http://127.0.0.1:8000/docs>. There will only be one model loaded with the name `default`.

You can also use a different device using the `--device` option.

## API methods

### `models`

The `models` method is called with a `GET` request and enumerate the available models.

Example: <http://127.0.0.1:8000/models>

### `process`

The `process` method is called with a `POST` request

| Parameter | Mandatory | Data type                        | Description                                                  |
| :-------- | :-------- | :------------------------------- | :----------------------------------------------------------- |
| data      | yes       | string                           | Input text in UTF-8.                                         |
| model     | no        | string                           | Model to use from those returned by `models`.                |
| tokenizer | no        | string                           | Ignored (for compatibility with UDPipe).                     |
| input     | no        | string (`conllu` / `horizontal`) | The input is assumed to be in this format; default `conllu`. |
| tagger    | no        | string                           | Ignored (for compatibility with UDPipe).                     |
| parser    | no        | string                           | Ignored (for compatibility with UDPipe).                     |
| output    | no        | string (`conllu`)                | The output format to use; default `conllu`.                  |

 The response is in JSON format of the following structure:

```json
{
  "model": "Model used",
  "acknowledgements": ["URL with acknowledgements", ...],
  "result": "processed_output"
}
```

Where `processed_output` is the output of parser in the requested output format. 

## Going further

`hopsparser serve` is actually a very thin wrapper for running `npdependency.server:app` in
[Uvicorn](https://www.uvicorn.org). `npdependency.server:app` uses the
[ASGI](https://asgi.readthedocs.io/en/latest/) interface and you can use it directly for fun and
profits. For instance you could

- Load several models (passing the model mapping using the `MODELS` environment variable)
- Run several workers using the `--workers` Uvicorn parameter
- Use another ASGI server such as [Hypercorn](https://pypi.org/project/Hypercorn/)

To avoid pulling too many dependencies, we only depend on the minimal versions of FastAPI and
Uvicorn, and although it is very unlikeley to be a bottleneck, installing their extra dependencies
using `fastapi[all]` and `uvicorn[standard]` *might* make the server faster.
