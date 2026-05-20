rm -rf oscar.egg-info build/ oscar_cuda.cpython-310-x86_64-linux-gnu.so dist/

uv sync --extra cu124 --extra eval --no-install-project
uv pip install --no-build-isolation -e .

# uv sync --extra cu124 --extra eval
