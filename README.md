# AI4META
Explainable AI for Metabolomics datasets

## Docker
How to _build_ and _run_ the package in a _Docker_ container.

1. Use __build.sh__ script to _build_ the _Dockerfile_: `./build.sh`
2. Use __run.sh__ script to _run_ the _Dockerfile_: `./run.sh`

How to start a __Jupyter Notebook__ inside the container:
`jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root`

If it asks you for a _Token_, you can find the token writen in the URL or in the terminal with the Jupyter session.WWW