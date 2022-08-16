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
# # AOC 2020

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
# https://adventofcode.com/2020/day/1

# %%
data = mapl(int, get_data(year=2020, day=1).split("\n"))

# %% [markdown]
# Part 1

# %%
next(a * b for a, b in itertools.combinations(data, r=2) if a + b == 2020)

# %% [markdown]
# Part 2

# %%
next(a * b * c for a, b, c in itertools.combinations(data, r=3) if a + b + c == 2020)

# %% [markdown]
# ## Day 02
# https://adventofcode.com/2020/day/2

# %%
data = get_data(year=2020, day=2).split("\n")


# %% [markdown]
# Part 1

# %%
total = 0
for password in data:
    s, e, c, pswd = re.match(r"(\d+)-(\d+) (\w): (\w+)", password).groups()
    if int(s) <= pswd.count(c) <= int(e):
        total += 1

total

# %% [markdown]
# Part 2

# %%
total = 0
for password in data:
    s, e, c, pswd = re.match(r"(\d+)-(\d+) (\w): (\w+)", password).groups()
    if (pswd[int(s) - 1] == c) ^ (pswd[int(e) - 1] == c):
        total += 1

total

# %% [markdown]
# ## Day 03
# https://adventofcode.com/2020/day/3

# %%
data = get_data(year=2020, day=3).split("\n")


# %%
def get_tree(right, down):
    offset = count = 0
    for row in data[::down]:
        count += row[offset % len(data[0])] == "#"
        offset += right
    return count


# %% [markdown]
# Part 1

# %%
get_tree(3, 1)

# %% [markdown]
# Part 2

# %%
(get_tree(1, 1) * get_tree(3, 1) * get_tree(5, 1) * get_tree(7, 1) * get_tree(1, 2))

# %% [markdown]
# ## Day 04
# https://adventofcode.com/2020/day/4

# %%
data = get_data(year=2020, day=4)
data = [l.replace("\n", " ") for l in data.split("\n\n")]


# %%
CODES = {"ecl", "pid", "eyr", "hcl", "byr", "iyr", "hgt"}

# %% [markdown]
# Part 1

# %%
sum(map(lambda x: not (CODES - set(re.findall(r"(\w+):", x))), data))

# %% [markdown]
# Part 2

# %%
solution = -1
for line in data:
    pairs = re.findall(r"(\w+):(\S+)", line)
    if CODES - {p[0] for p in pairs}:
        continue

    valid = True
    for k, v in pairs:
        if k == "byr":
            valid &= 1920 <= int(v) <= 2002

        elif k == "iyr":
            valid &= 2010 <= int(v) <= 2020

        elif k == "eyr":
            valid &= 2020 <= int(v) <= 2030

        elif k == "hgt":
            if v.endswith("cm"):
                valid &= 150 <= int(v[:-2]) <= 193
            elif v.endswith("in"):
                valid &= 59 <= int(v[:-2]) <= 76

        elif k == "hcl":
            valid &= bool(re.fullmatch(r"#[0-9a-f]{6}", v))

        elif k == "ecl":
            valid &= v in {"amb", "blu", "brn", "gry", "grn", "hzl", "oth"}

        elif k == "pid":
            valid &= bool(re.fullmatch(r"[0-9]{9}", v))

        elif k == "cid":
            valid &= True

    solution += valid

solution

# %% [markdown]
# ## Day 05
# https://adventofcode.com/2020/day/5

# %%
data = get_data(year=2020, day=5).split("\n")


# %%
mapping = str.maketrans("FLBR", "0011")
IDS = sorted(int(line.translate(mapping), 2) for line in data)

# %% [markdown]
# Part 1

# %%
IDS[-1]

# %% [markdown]
# Part 2

# %%
next(id2 - 1 for id1, id2 in zip(IDS, IDS[1:]) if id2 - id1 - 1)

# %% [markdown]
# ## Day 06
# https://adventofcode.com/2020/day/6

# %%
data = get_data(year=2020, day=6).split("\n\n")


# %% [markdown]
# Part 1

# %%
sum(len(set.union(*[set(answer) for answer in answers.split()])) for answers in data)

# %% [markdown]
# Part 2

# %%
sum(len(set.intersection(*[set(answer) for answer in answers.split()])) for answers in data)

# %% [markdown]
# ## Day 07
# https://adventofcode.com/2020/day/7

# %%
data = get_data(year=2020, day=7).split("\n")


# %%
mapping = collections.defaultdict(set)
mapping_inverse = collections.defaultdict(list)
for line in data:
    color = re.match(r"([\w ]+) bags contain", line)[1]
    for count, incolor in re.findall(r"(\d+) ([\w ]+) bag", line):
        mapping[incolor].add(color)
        mapping_inverse[color].append((int(count), incolor))

# %% [markdown]
# Part 1

# %%
acc = set()


def recurse(color):
    for c in mapping[color]:
        acc.add(c)
        recurse(c)


recurse("shiny gold")
len(acc)


# %% [markdown]
# Part 2

# %%
def cost(color):
    total = 0
    for count, incolor in mapping_inverse[color]:
        total += count
        total += count * cost(incolor)
    return total


cost("shiny gold")

# %% [markdown]
# ## Day 08
# https://adventofcode.com/2020/day/8

# %%
data = get_data(year=2020, day=8).split("\n")
data = mapl(str.split, data)
data = mapl(lambda x: (x[0], int(x[1])), data)


# %%
def run(prog, return_loop=False):
    acc = 0
    ptr = 0
    seen = set()
    while 0 <= ptr < len(prog):
        if ptr in seen:
            return acc if return_loop else None

        seen.add(ptr)

        inst, arg = prog[ptr]
        if inst == "jmp":
            ptr += arg
            continue
        if inst == "acc":
            acc += arg
        if inst == "nop":
            pass

        ptr += 1

    return acc


# %% [markdown]
# Part 1

# %%
run(data, return_loop=True)

# %% [markdown]
# Part 2

# %%
for idx, (inst, arg) in enumerate(data):
    prog = data[:]

    if inst == "jmp":
        prog[idx] = ("nop", arg)
    if inst == "nop":
        prog[idx] = ("jmp", arg)

    if inst in ["jmp", "nop"]:
        acc = run(prog)
        if acc:
            break

acc

# %% [markdown]
# ## Day 09
# https://adventofcode.com/2020/day/9

# %%
data = mapl(int, get_data(year=2020, day=9).split("\n"))


# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 10
# https://adventofcode.com/2020/day/10

# %%
data = sorted(mapl(int, get_data(year=2020, day=10).split("\n")))

# %% [markdown]
# Part 1

# %%
diffs = collections.Counter(n2 - n1 for n1, n2 in zip(data, data[1:]))
(diffs[1] + 1) * (diffs[3] + 1)

# %% [markdown]
# Part 2

# %%
acc = collections.Counter({0: 1})
for n in data:
    acc[n] = acc[n - 3] + acc[n - 2] + acc[n - 1]

acc.most_common(1)[0][1]


# %% [markdown]
# ## Day 11
# https://adventofcode.com/2020/day/11

# %%
data = get_data(year=2020, day=11)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 12
# https://adventofcode.com/2020/day/12

# %%
data = get_data(year=2020, day=12)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 13
# https://adventofcode.com/2020/day/13

# %%
data = get_data(year=2020, day=13)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 14
# https://adventofcode.com/2020/day/14
# %%
data = get_data(year=2020, day=14)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 15
# https://adventofcode.com/2020/day/15
# %%
data = get_data(year=2020, day=15)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 16
# https://adventofcode.com/2020/day/16
# %%
data = get_data(year=2020, day=16)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 17
# https://adventofcode.com/2020/day/17
# %%
data = get_data(year=2020, day=17)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 18
# https://adventofcode.com/2020/day/18
# %%
data = get_data(year=2020, day=18)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 19
# https://adventofcode.com/2020/day/19
# %%
data = get_data(year=2020, day=19)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 20
# https://adventofcode.com/2020/day/20
# %%
data = get_data(year=2020, day=20)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 21
# https://adventofcode.com/2020/day/21
# %%
data = get_data(year=2020, day=21)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 22
# https://adventofcode.com/2020/day/22
# %%
data = get_data(year=2020, day=22)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 23
# https://adventofcode.com/2020/day/23
# %%
data = get_data(year=2020, day=23)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 24
# https://adventofcode.com/2020/day/24
# %%
data = get_data(year=2020, day=24)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 25
# https://adventofcode.com/2020/day/25
# %%
data = get_data(year=2020, day=25)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%
