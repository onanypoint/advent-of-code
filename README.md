# Advent Of Code Solutions

Advent of Code is yearly computer programming challenges that follow an Advent calendar. Puzzles consist of two parts that must be solved in order, with the second part becoming visible once the first part is solved correctly.

## Editions

- 2015 ([solutions](solutions/aoc2015.py), [site](https://adventofcode.com/2015))
- 2016 ([solutions](solutions/aoc2016.py), [site](https://adventofcode.com/2016))
- 2017 ([solutions](solutions/aoc2017.py), [site](https://adventofcode.com/2017))
- 2018 ([solutions](solutions/aoc2018.py), [site](https://adventofcode.com/2018))
- 2019 ([solutions](solutions/aoc2019.py), [site](https://adventofcode.com/2019))
- 2020 ([solutions](solutions/aoc2020.py), [site](https://adventofcode.com/2020))
- 2021 ([solutions](solutions/aoc2021.py), [site](https://adventofcode.com/2021))
- 2021 ([solutions](solutions/aoc2022.py), [site](https://adventofcode.com/2022))

## Usage

This repository is setup using [Poetry](https://python-poetry.org/) and uses [Jupyter Notebook](https://jupyter.org/) for code development. The notebooks are processed via [jupytext](https://jupytext.readthedocs.io/) to move between notebooks (`.ipynb`) and Python files.

If you wish to generate the `ipynb` files, you can uncomment the `tool.jupytext.formats` part of the `pyproject.toml` and enable a pre-commit hook to run `juyptext-sync`.

**Add a new year (or notebooks)**

1. Navigate in Jupyter to `solutions/`
2. Create a new notebook using your favorite kernel
3. Watch a python file be automagically created in the `solutions` folder (if you activated the option).
4. Enjoy
