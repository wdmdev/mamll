# MAMLL - My Advanced Machine Learning Library
A python project I have created for my learnings in the course [02460 - Advanced Machine Learning](https://kurser.dtu.dk/course/02460) at [The Technical University of Denmark](https://www.dtu.dk/).

## Content
An overview of the content of this library in relation to the course weeks:

* **Week 1**: Deep latent variable models: `dgm/vae_bernoulli.py` and `dgm/mog.py`

## Installation
I make use of [`poetry`](https://python-poetry.org/) and [conda](https://docs.conda.io/projects/miniconda/en/latest/), in which case the installation flow is to first activate your conda environment with python in it and then run:
```
make poetry
```

Otherwise it should also be straight forward to use [`pip`](https://pypi.org/project/pip/) e.g. in a conda environment with python installed. In that case run:
```
make pip
```
or use 
```
make pip_env
```
to create a virtual environment named `env` with the necessary dependencies.