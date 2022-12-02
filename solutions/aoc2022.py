# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:aoc]
#     language: python
#     name: conda-env-aoc-py
# ---

# %% [markdown]
# # AOC 2022

# %%
from dataclasses import dataclass
import collections
import copy
import enum
import functools
import hashlib
import itertools
import json
import math
import operator
import random
import re
import string
import sys

from aocd import get_data, submit
from toolbox import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import scipy


# %% [markdown]
# ## Day 01
# https://adventofcode.com/2022/day/1

# %%
data = get_data(year=2022, day=1).split("\n\n")
data = [sum(mapl(int, entry.split("\n"))) for entry in data]

# %% [markdown]
# Part 1

# %%
max(data)

# %% [markdown]
# Part 2

# %%
sum(sorted(data, reverse=True)[:3])

# %% [markdown]
# ## Day 02
# https://adventofcode.com/2022/day/2

# %%
data = get_data(year=2022, day=2).split("\n")
data = [l.split() for l in data]


# %%
moves = {
    "A": 1,
    "B": 2,
    "C": 3,
}

shapes = {
    "X": 1,
    "Y": 2,
    "Z": 3,
}

# %% [markdown]
# Part 1

# %%
score = 0
for x, y in data:
    if moves[x] == shapes[y]:
        score += 3
    elif (x, y) in [("A", "Y"), ("B", "Z"), ("C", "X")]:
        score += 6

    score += shapes[y]

score

# %% [markdown]
# Part 2

# %%
decisions = {
    "X": {"A": "Z", "B": "X", "C": "Y"},
    "Y": {"A": "X", "B": "Y", "C": "Z"},
    "Z": {"A": "Y", "B": "Z", "C": "X"},
}

# %%
score = 0
for x, y in data:

    move = decisions[y][x]

    match y:
        case "X":
            score += 0
        case "Y":
            score += 3
        case "Z":
            score += 6

    score += shapes[move]

score

# %% [markdown]
# ## Day 03
# https://adventofcode.com/2022/day/3

# %%
data = get_data(year=2022, day=3)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 04
# https://adventofcode.com/2022/day/4

# %%
data = get_data(year=2022, day=4)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 05
# https://adventofcode.com/2022/day/5

# %%
data = get_data(year=2022, day=5)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 06
# https://adventofcode.com/2022/day/6

# %%
data = get_data(year=2022, day=6)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 07
# https://adventofcode.com/2022/day/7

# %%
data = get_data(year=2022, day=7)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 08
# https://adventofcode.com/2022/day/8

# %%
data = get_data(year=2022, day=8)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 09
# https://adventofcode.com/2022/day/9

# %%
data = get_data(year=2022, day=9)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 10
# https://adventofcode.com/2022/day/10

# %%
data = get_data(year=2022, day=10)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 11
# https://adventofcode.com/2022/day/11

# %%
data = get_data(year=2022, day=11)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 12
# https://adventofcode.com/2022/day/12

# %%
data = get_data(year=2022, day=12)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 13
# https://adventofcode.com/2022/day/13

# %%
data = get_data(year=2022, day=13)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 14
# https://adventofcode.com/2022/day/14
# %%
data = get_data(year=2022, day=14)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 15
# https://adventofcode.com/2022/day/15
# %%
data = get_data(year=2022, day=15)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 16
# https://adventofcode.com/2022/day/16
# %%
data = get_data(year=2022, day=16)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 17
# https://adventofcode.com/2022/day/17
# %%
data = get_data(year=2022, day=17)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 18
# https://adventofcode.com/2022/day/18
# %%
data = get_data(year=2022, day=18)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 19
# https://adventofcode.com/2022/day/19
# %%
data = get_data(year=2022, day=19)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 20
# https://adventofcode.com/2022/day/20
# %%
data = get_data(year=2022, day=20)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 21
# https://adventofcode.com/2022/day/21
# %%
data = get_data(year=2022, day=21)

# %%


# %%

# %% [markdown]
#
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 22
# https://adventofcode.com/2022/day/22
# %%
data = get_data(year=2022, day=22)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 23
# https://adventofcode.com/2022/day/23
# %%
data = get_data(year=2022, day=23)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 24
# https://adventofcode.com/2022/day/24
# %%
data = get_data(year=2022, day=24)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 25
# https://adventofcode.com/2022/day/25
# %%
data = get_data(year=2022, day=25)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%
