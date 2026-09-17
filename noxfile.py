import pathlib
from os import environ

from nox import Session, options, project, session

options.default_venv_backend = "uv"

ci = environ.get("CI")


@session(python=["3.11", "3.12", "3.13", "3.14"])
def tests(s: Session):
    if ci:
        s.install(".", "--group", "ci")
        posargs = s.posargs
        # TODO: figure out which tests are light enough to run every time CI runs
        # posargs = [*s.posargs, "-m", "light"]

    else:
        s.install(".", "--group", "tests")
        posargs = s.posargs

    s.run("pytest", "tests", "--basetemp", s.create_tmp(), *posargs)


@session()
def lint(s: Session):
    pyproject = project.load_toml("pyproject.toml")
    s.install(*project.dependency_groups(pyproject, "lint"))
    if ci:
        fmt = "github"
    else:
        fmt = "full"
    s.run("ruff", "check", ".", "--select=E9,F63,F7,F82", f"--output-format={fmt}")
    # exit-zero treats all errors as warnings.
    s.run("ruff", "check", ".", "--exit-zero", f"--output-format={fmt}")


@session
def docs(s: Session):
    pyproject = project.load_toml("pyproject.toml")
    s.install(*project.dependency_groups(pyproject, "docs"))
    doc_path = pathlib.Path("docs")
    if "--serve" in s.posargs:
        command = "serve"
        args = ["--watch", ".", "--watch-theme"]
    else:
        command = "build"
        args = ["--site-dir", "_build"]

    s.run("mkdocs", command, "--config-file", doc_path / "mkdocs.yml", *args)
