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
# # AOC 2016

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
# https://adventofcode.com/2016/day/1

# %%
data = get_data(year=2016, day=1).split(", ")
data = [(value[0], int(value[1:])) for value in data]

# %% [markdown]
# Part 1

# %%
x, y = 0, 0

for direction, move in data:

    if direction == "L":
        x, y = y, -x
    else:
        x, y = -y, x

    x += move

abs(x) + abs(y)

# %% [markdown]
# Part 2 #TODO

# %%


# %% [markdown]
# ## Day 02
# https://adventofcode.com/2016/day/2

# %%
data = get_data(year=2016, day=2).split()


# %% [markdown]
# Part 1

# %%
keypad = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)


def clip(value):
    return min(max(0, value), 2)


x, y = (1, 1)
for instruction in data:
    for c in instruction:

        match c:
            case "U":
                x, y = x, clip(y - 1)
            case "L":
                x, y = clip(x - 1), y
            case "R":
                x, y = clip(x + 1), y
            case "D":
                x, y = x, clip(y + 1)

    print(keypad[y, x], end="")

# %% [markdown]
# Part 2

# %%
keypad = np.array(
    [
        ["0", "0", "1", "0", "0"],
        ["0", "2", "3", "4", "0"],
        ["5", "6", "7", "8", "9"],
        ["0", "A", "B", "C", "0"],
        ["0", "0", "D", "0", "0"],
    ]
)


def clip(x, y, new_x, new_y):
    if 0 <= new_x <= 4 and 0 <= new_y <= 4 and keypad[new_y, new_x] != "0":
        return new_x, new_y

    return x, y


x, y = (0, 2)
for instruction in data:
    for c in instruction:
        match c:
            case "U":
                x, y = clip(x, y, x, y - 1)
            case "L":
                x, y = clip(x, y, x - 1, y)
            case "R":
                x, y = clip(x, y, x + 1, y)
            case "D":
                x, y = clip(x, y, x, y + 1)

    print(keypad[y, x], end="")

# %% [markdown]
# ## Day 03
# https://adventofcode.com/2016/day/3

# %%
data = mapl(lambda x: mapl(int, x.split()), get_data(year=2016, day=3).split("\n"))


# %% [markdown]
# Part 1

# %%
sum(a + b > c for a, b, c in map(sorted, data))

# %% [markdown]
# Part 2

# %%
sum(a + b > c for a, b, c in map(sorted, np.ravel(data, "F").reshape((len(data), 3))))

# %% [markdown]
# ## Day 04
# https://adventofcode.com/2016/day/4

# %%
data = get_data(year=2016, day=4).split("\n")
rooms = mapl(lambda x: re.search(r"([\w-]+)-(\d+)\[(\w+)\]", x).groups(), data)


# %% [markdown]
# Part 1

# %%
def is_real(name, checksum):
    return checksum == "".join(
        mapl(operator.itemgetter(0), sorted(collections.Counter(name.replace("-", "")).items(), key=lambda x: (-x[1], x[0]))[:5])
    )


real_rooms = [(name, int(sector), checksum) for name, sector, checksum in rooms if is_real(name, checksum)]

# %%
sum(sector for _, sector, _ in real_rooms)


# %% [markdown]
# Part 2

# %%
def rotate(c, count):
    return chr((ord(c) - 97 + count) % 26 + 97)


def decode(s, count):
    return "".join(rotate(c, count) if c != "-" else " " for c in s)


mapping = {decode(name, int(sector)): sector for name, sector, checksum in real_rooms}

# %%
for name, sector in mapping.items():
    if "north" in name:
        print(name, sector)
        break

# %% [markdown]
# ## Day 05
# https://adventofcode.com/2016/day/5

# %%
data = get_data(year=2016, day=5)


# %% [markdown]
# Part 1

# %%
password = ""
idx = 0

while True:
    hsh = hashlib.md5(f"{data}{idx}".encode()).hexdigest()
    if hsh.startswith("00000"):
        password += hsh[5]

    idx += 1

    if len(password) == 8:
        break

# %% [markdown]
# Part 2

# %%
password = ""
idx = 0

while True:
    hsh = hashlib.md5(f"{data}{idx}".encode()).hexdigest()
    if hsh.startswith("00000") and hsh[5].isdigit():
        password[hsh[5]] += hsh[6]

    idx += 1

    if len(password_1) == 8:
        break

# %% [markdown]
# ## Day 06
# https://adventofcode.com/2016/day/6

# %%
data = get_data(year=2016, day=6).split("\n")


# %% [markdown]
# Part 1

# %%
"".join(mapl(lambda x: collections.Counter(x).most_common(1)[0][0], np.array(mapl(list, data)).T))

# %% [markdown]
# Part 2

# %%
"".join(mapl(lambda x: collections.Counter(x).most_common()[-1][0][0], np.array(mapl(list, data)).T))

# %% [markdown]
# ## Day 07
# https://adventofcode.com/2016/day/7

# %%
data = get_data(year=2016, day=7).split("\n")
data = mapl(lambda x: x.replace("[", "]").split("]"), data)
data = mapl(lambda x: (x[::2], x[1::2]), data)


# %% [markdown]
#
# Part 1

# %%
abba_pattern = re.compile(r"(\w)((?!\1)\w)\2\1")
sum(map(lambda x: any(map(abba_pattern.search, x[0])) and not any(map(abba_pattern.search, x[1])), data))

# %% [markdown]
# Part 2

# %%
aba_pattern = re.compile(r"(\w)((?!\1)\w)\1.*#.*\2\1\2")

sum(mapl(lambda x: bool(aba_pattern.search(" ".join([*x[0], "#", *x[1]]))), data))

# %% [markdown]
# ## Day 08
# https://adventofcode.com/2016/day/8

# %%
data = get_data(year=2016, day=8).split("\n")


# %% [markdown]
# Part 1

# %%
display = np.zeros((6, 50))
y, x = display.shape

for line in data:
    command, *parsed = line.split(" ")

    if command == "rect":
        a, b = mapl(int, parsed[0].split("x"))
        display[:b, :a] = 1

    elif command == "rotate":
        side, a, _, b = parsed
        a, b = int(a[2:]), int(b)
        if side == "column":
            display[:, a] = np.concatenate([display[-b % y :, a], display[: -b % y, a]])
        else:
            display[a, :] = np.concatenate([display[a, -b % x :], display[a, : -b % x]])

# %%
display.sum()

# %% [markdown]
# Part 2

# %%
mapl(lambda x: "".join(x).replace("1", "#").replace("0", " "), display.astype(int).astype(str))

# %% [markdown]
# ## Day 09
# https://adventofcode.com/2016/day/9

# %%
data = get_data(year=2016, day=9)


# %% [markdown]
# Part 1

# %%
s = ""
i = 0

while i < len(data):

    if match := re.search(r"^\((\d*)x(\d*)\)", data[i:]):
        length, rep = mapl(int, match.groups())
        s += data[match.span()[1] + 1 :][:length] * rep
        i += match.span()[1] + length
    else:
        s += data[i]
        i += 1

# %%
len(s)


# %% [markdown]
# Part 2

# %%
def expand(s):
    if not s:
        return 0

    if match := re.search(r"^\((\d*)x(\d*)\)", s):
        length, rep = mapl(int, match.groups())
        tmp = s[match.span()[1] :][:length]
        return expand(tmp) * int(rep) + expand(s[match.span()[1] + length :])
    else:
        return 1 + expand(s[1:])


expand(data)

# %% [markdown]
# ## Day 10
# https://adventofcode.com/2016/day/10

# %%
data = get_data(year=2016, day=10).split("\n")

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 11
# https://adventofcode.com/2016/day/11

# %%
data = get_data(year=2016, day=11)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 12
# https://adventofcode.com/2016/day/12

# %%
data = get_data(year=2016, day=12)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 13
# https://adventofcode.com/2016/day/13

# %%
data = get_data(year=2016, day=13)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 14
# https://adventofcode.com/2016/day/14
# %%
data = get_data(year=2016, day=14)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 15
# https://adventofcode.com/2016/day/15
# %%
data = get_data(year=2016, day=15)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 16
# https://adventofcode.com/2016/day/16
# %%
data = get_data(year=2016, day=16)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 17
# https://adventofcode.com/2016/day/17
# %%
data = get_data(year=2016, day=17)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 18
# https://adventofcode.com/2016/day/18
# %%
data = get_data(year=2016, day=18)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 19
# https://adventofcode.com/2016/day/19
# %%
data = get_data(year=2016, day=19)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 20
# https://adventofcode.com/2016/day/20
# %%
data = get_data(year=2016, day=20)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 21
# https://adventofcode.com/2016/day/21
# %%
data = get_data(year=2016, day=21)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 22
# https://adventofcode.com/2016/day/22
# %%
data = get_data(year=2016, day=22)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 23
# https://adventofcode.com/2016/day/23
# %%
data = get_data(year=2016, day=23)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 24
# https://adventofcode.com/2016/day/24
# %%
data = get_data(year=2016, day=24)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 25
# https://adventofcode.com/2016/day/25
# %%
data = get_data(year=2016, day=25)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%
