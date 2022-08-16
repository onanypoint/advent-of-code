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
# # AOC 2017

# %%
from dataclasses import dataclass
import collections
import copy
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
# https://adventofcode.com/2017/day/1

# %%
data = get_data(year=2017, day=1)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 02
# https://adventofcode.com/2017/day/2

# %%
data = get_data(year=2017, day=2)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 03
# https://adventofcode.com/2017/day/3

# %%
data = get_data(year=2017, day=3)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 04
# https://adventofcode.com/2017/day/4

# %%
data = get_data(year=2017, day=4)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 05
# https://adventofcode.com/2017/day/5

# %%
data = get_data(year=2017, day=5)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 06
# https://adventofcode.com/2017/day/6

# %%
data = get_data(year=2017, day=6)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 07
# https://adventofcode.com/2017/day/7

# %%
data = get_data(year=2017, day=7)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 08
# https://adventofcode.com/2017/day/8

# %%
data = get_data(year=2017, day=8)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 09
# https://adventofcode.com/2017/day/9

# %%
data = get_data(year=2017, day=9)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 10
# https://adventofcode.com/2017/day/10

# %%
data = get_data(year=2017, day=10)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 11
# https://adventofcode.com/2017/day/11

# %%
data = get_data(year=2017, day=11)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 12
# https://adventofcode.com/2017/day/12

# %%
data = get_data(year=2017, day=12)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 13
# https://adventofcode.com/2017/day/13

# %%
data = get_data(year=2017, day=13)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 14
# https://adventofcode.com/2017/day/14
# %%
data = get_data(year=2017, day=14)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 15
# https://adventofcode.com/2017/day/15
# %%
data = get_data(year=2017, day=15)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 16
# https://adventofcode.com/2017/day/16
# %%
data = get_data(year=2017, day=16)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 17
# https://adventofcode.com/2017/day/17
# %%
data = get_data(year=2017, day=17)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 18
# https://adventofcode.com/2017/day/18
# %%
data = get_data(year=2017, day=18)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 19
# https://adventofcode.com/2017/day/19
# %%
data = get_data(year=2017, day=19)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 20
# https://adventofcode.com/2017/day/20
# %%
data = get_data(year=2017, day=20)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 21
# https://adventofcode.com/2017/day/21
# %%
data = get_data(year=2017, day=21).split("\n")

# %%
mapping = {}

for entry in data:
    key, pattern = mapl(lambda x: np.array(mapl(list, x.split("/"))) == "#", entry.split(" => "))

    mapping[key.tobytes()] = pattern

    mapping[np.fliplr(key).tobytes()] = pattern

    mapping[np.fliplr(np.rot90(key, k=1)).tobytes()] = pattern
    mapping[np.fliplr(np.rot90(key, k=2)).tobytes()] = pattern
    mapping[np.fliplr(np.rot90(key, k=3)).tobytes()] = pattern

    mapping[np.rot90(key, k=1).tobytes()] = pattern
    mapping[np.rot90(key, k=2).tobytes()] = pattern
    mapping[np.rot90(key, k=3).tobytes()] = pattern


# %%
def enhance(h, w, step, out):
    new = np.zeros((h // step, w // step, out, out)).astype(bool)

    for yidx, y in enumerate(range(0, h, step)):
        for xidx, x in enumerate(range(0, w, step)):
            new[yidx, xidx, :] = mapping[start[y : y + step, x : x + step].tobytes()]

    return np.hstack(np.hstack(new))


def step(image):
    h, w = image.shape
    return enhance(h, w, 2 if h % 2 == 0 else 3, 3 if h % 2 == 0 else 4)


# %%
start = np.array(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)

# %% [markdown]
#
# Part 1

# %%
for i in range(5):
    start = step(start)

start.sum()

# %% [markdown]
# Part 2

# %%
for i in range(18 - 5):
    start = step(start)

start.sum()


# %% [markdown]
# ## Day 22
# https://adventofcode.com/2017/day/22
# %%
data = get_data(year=2017, day=22)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 23
# https://adventofcode.com/2017/day/23
# %%
data = get_data(year=2017, day=23)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 24
# https://adventofcode.com/2017/day/24
# %%
data = get_data(year=2017, day=24)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 25
# https://adventofcode.com/2017/day/25
# %%
data = get_data(year=2017, day=25)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%
