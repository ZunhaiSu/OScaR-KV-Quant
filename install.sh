rm -rf oscar.egg-info build/ oscar_cuda.cpython-310-x86_64-linux-gnu.so dist/

# flash-attn imports torch and psutil while building, but does not declare them
# as build dependencies. Install them into the active environment before uv sync
# can attempt to build flash-attn.
uv pip install "torch==2.6.0+cu124" psutil --index https://download.pytorch.org/whl/cu124

uv sync --active --no-install-project
uv pip install --no-build-isolation -e .
