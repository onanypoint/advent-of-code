# Advent Of Code Solutions

Advent of Code is yearly computer programming challenges that follow an Advent calendar. Puzzles consist of two parts that must be solved in order, with the second part becoming visible once the first part is solved correctly.

## Editions

- 2015 ([solutions](notebooks/2015.ipynb), [site](https://adventofcode.com/2015))

## Usage

This repository is setup using [Poetry](https://python-poetry.org/) (a complimentary `requirements.txt` is available) and uses [Jupyter Notebook](https://jupyter.org/) for code development. The notebooks are processed via [jupytext](https://jupytext.readthedocs.io/) to move between notebooks (`.ipynb`) and Python files.

If you wish to generate the `ipynb` files, you can uncomment the `tool.jupytext.formats` part of the `pyproject.toml` and enable a pre-commit hook to run `juyptext-sync`.

**Add a new year (or notebooks)**

1. Navigate in Jupyter to `notebooks/`
2. Create a new notebook using your favorite kernel
3. Watch a python file be automagically created in the `script` folder (if you activated the option).
4. Enjoy
