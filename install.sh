rm -rf oscar.egg-info build/ oscar_cuda.cpython-310-x86_64-linux-gnu.so dist/

python -m pip install --no-build-isolation -e .

# python -m pip install --no-build-isolation -e ".[eval]"
