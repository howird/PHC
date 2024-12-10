# setup environment with uv

- an almost vanilla uv setup except some hacks we have to do because chumpy assumes that `pip` and `setuptools` are installed and available at install time:

```bash
uv venv
source .venv/bin/activate
uv pip install pip setuptools
uv sync --no-build-isolation
```
