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
# # AOC 2021

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
# https://adventofcode.com/2021/day/1

# %%
data = get_data(year=2021, day=1)
depths = mapl(int, data.split("\n"))


# %%
def get_reports(depths):
    return sum((dep - dep_os) < 0 for dep, dep_os in zip(depths[:-1], depths[1:]))


# %% [markdown]
# Part 1

# %%
get_reports(depths)

# %% [markdown]
# Part 2

# %%
get_reports(mapl(sum, zip(depths[2:], depths[1:-1], depths[:-2])))

# %% [markdown]
# ## Day 02
# https://adventofcode.com/2021/day/2

# %%
data = get_data(year=2021, day=2)
commands = mapl(str.split, data.split("\n"))


# %%
aim = 0
horiz = 0
depth1 = 0
depth2 = 0

for cmd, x in commands:
    x = int(x)
    if cmd == "down":
        depth1 += x
        aim += x
    elif cmd == "up":
        depth1 -= x
        aim -= x
    else:
        horiz += x
        depth2 += aim * x

# %% [markdown]
# Part 1

# %%
horiz * depth1

# %% [markdown]
# Part 2

# %%
horiz * depth2

# %% [markdown]
# ## Day 03
# https://adventofcode.com/2021/day/3

# %%
data = get_data(year=2021, day=3)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 04
# https://adventofcode.com/2021/day/4

# %%
nums, _, *boards = get_data(year=2021, day=4).split("\n")
nums = np.array(nums.split(",")).astype(int)

boards = np.array([[list(map(int, filter(lambda x: x != "", row.split(" ")))) for row in boards[i : i + 5]] for i in range(0, len(boards), 6)])


# %%
def calculate_win_number_score(board, marks, num_list, i=0):
    num = num_list[i]
    marks[board == num] = True
    if any(np.sum(marks, axis=0) == 5) or any(np.sum(marks, axis=1) == 5):
        return [i, int(np.sum(board[marks == False]) * num)]
    return calculate_win_number_score(board, marks, num_list, i + 1)


# %%
win_numbers_scores = np.array([calculate_win_number_score(board, np.zeros((5, 5)), nums) for board in boards])

# %% [markdown]
# Part 1

# %%
win_numbers_scores[np.argmin(win_numbers_scores[:, 0]), 1]

# %% [markdown]
# Part 2

# %%
win_numbers_scores[np.argmax(win_numbers_scores[:, 0]), 1]

# %% [markdown]
# ## Day 05
# https://adventofcode.com/2021/day/5

# %%
data = get_data(year=2021, day=5)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 06
# https://adventofcode.com/2021/day/6

# %%
data = get_data(year=2021, day=6)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 07
# https://adventofcode.com/2021/day/7

# %%
data = get_data(year=2021, day=7)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 08
# https://adventofcode.com/2021/day/8

# %%
data = get_data(year=2021, day=8)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 09
# https://adventofcode.com/2021/day/9

# %%
data = get_data(year=2021, day=9)


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 10
# https://adventofcode.com/2021/day/10

# %%
data = get_data(year=2021, day=10)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 11
# https://adventofcode.com/2021/day/11

# %%
data = get_data(year=2021, day=11)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 12
# https://adventofcode.com/2021/day/12

# %%
data = get_data(year=2021, day=12)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 13
# https://adventofcode.com/2021/day/13

# %%
data = get_data(year=2021, day=13)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 14
# https://adventofcode.com/2021/day/14
# %%
data = get_data(year=2021, day=14)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 15
# https://adventofcode.com/2021/day/15
# %%
data = get_data(year=2021, day=15)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 16
# https://adventofcode.com/2021/day/16
# %%
data = get_data(year=2021, day=16)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 17
# https://adventofcode.com/2021/day/17
# %%
data = get_data(year=2021, day=17)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 18
# https://adventofcode.com/2021/day/18
# %%
data = get_data(year=2021, day=18)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 19
# https://adventofcode.com/2021/day/19
# %%
data = get_data(year=2021, day=19)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 20
# https://adventofcode.com/2021/day/20
# %%
data = get_data(year=2021, day=20)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 21 #TODO
# https://adventofcode.com/2021/day/21
# %%
data = get_data(year=2021, day=21)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 22
# https://adventofcode.com/2021/day/22
# %%
data = get_data(year=2021, day=22)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 23
# https://adventofcode.com/2021/day/23
# %%
data = get_data(year=2021, day=23)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 24
# https://adventofcode.com/2021/day/24
# %%
data = get_data(year=2021, day=24)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 25
# https://adventofcode.com/2021/day/25
# %%
data = get_data(year=2021, day=25)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%
