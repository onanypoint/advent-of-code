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
import more_itertools
import networkx as nx
import numpy as np
import scipy


# %%
idefaultdict = lambda: collections.defaultdict(idefaultdict)

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
data = get_data(year=2022, day=3).split("\n")


def score(s):
    if s.isupper():
        return ord(s) - 65 + 27

    return ord(s) - 96


# %% [markdown]
# Part 1

# %%
total = 0
for entry in data:
    a = set(entry[: len(entry) // 2])
    b = set(entry[len(entry) // 2 :])
    total += sum(mapl(score, a & b))

total

# %% [markdown]
# Part 2

# %%
total = 0
for entries in more_itertools.chunked(data, 3):
    c = next(iter(set.intersection(*map(set, entries))))
    total += score(c)

total

# %% [markdown]
# ## Day 04
# https://adventofcode.com/2022/day/4

# %%
data = get_data(year=2022, day=4).split()
data = [mapl(int, re.search(r"(\d+)-(\d+),(\d+)-(\d+)", l).groups()) for l in data]


# %% [markdown]
# Part 1

# %%
sum((a1 <= b1 and b2 <= a2) or (b1 <= a1 and a2 <= b2) for a1, a2, b1, b2 in data)

# %% [markdown]
# Part 2

# %%
sum(a1 <= b2 and b1 <= a2 for a1, a2, b1, b2 in data)

# %% [markdown]
# ## Day 05
# https://adventofcode.com/2022/day/5

# %%
state, moves = get_data(year=2022, day=5).split("\n\n")


# %%
def get_crates(state):
    crates = collections.defaultdict(list)
    for row in state.splitlines()[-2::-1]:
        for index, value in enumerate(row[1::4], 1):
            if value.strip():
                crates[index].append(value)

    return crates


# %% [markdown]
# Part 1

# %%
crates = get_crates(state)
for move in moves.splitlines():
    a, b, c = mapl(int, move.split(" ")[1::2])
    for _ in range(a):
        crates[c].append(crates[b].pop())

"".join(val[-1] for val in crates.values())

# %% [markdown]
# Part 2

# %%
crates = get_crates(state)
for move in moves.splitlines():
    a, b, c = mapl(int, move.split(" ")[1::2])
    acc = [crates[b].pop() for _ in range(a)]
    crates[c].extend(acc[::-1])

"".join(val[-1] for val in crates.values())

# %% [markdown]
# ## Day 06
# https://adventofcode.com/2022/day/6

# %%
data = get_data(year=2022, day=6)


# %%
def get_after(n):
    for i in range(0, len(data)):
        if len(set(data[i : i + n])) == n:
            return i + n


# %% [markdown]
# Part 1

# %%
get_after(4)

# %% [markdown]
# Part 2

# %%
get_after(14)

# %% [markdown]
# ## Day 07
# https://adventofcode.com/2022/day/7

# %%
data = get_data(year=2022, day=7).split("\n")

# %%
sizes = []


def parse(iterator):
    total_size = 0
    for entry in iterator:
        match entry.split():
            case "$", "cd", "..":
                return total_size
            case "$", "cd", _:
                sizes.append(parse(iterator))
                total_size += sizes[-1]
            case "$", _:
                continue
            case a, b if a != "dir":
                total_size += int(a)

    return total_size


parse(iter(data))

# %% [markdown]
# Part 1

# %%
sum(v for v in sizes if v <= 100000)

# %% [markdown]
# Part 2

# %%
unused = 70000000 - total_size

min(v for v in sizes if unused + v >= 30000000)

# %% [markdown]
# ## Day 08
# https://adventofcode.com/2022/day/8

# %%
lines = get_data(year=2022, day=8).split("\n")


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
data = get_data(year=2022, day=9).split("\n")
data = [entry.split() for entry in data]
data = [(a, int(b)) for a, b in data]


# %%
MOVES = {"L": -1, "R": 1, "U": 1j, "D": -1j}

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
data = get_data(year=2022, day=10).split("\n")

# %%
acc = [1]
crt = [[" " for _ in range(40)] for _ in range(6)]


def step(value):
    acc.append(value)
    current_count = len(acc) - 1

    if abs(current_count % 40 - value) < 2:
        crt[current_count // 40][current_count % 40] = "#"


for entry in data:
    match entry.split():
        case ["noop"]:
            step(acc[-1])
        case "addx", a:
            step(acc[-1])
            step(acc[-1] + int(a))

# %% [markdown]
# Part 1

# %%
sum(ptr * acc[ptr - 1] for ptr in [20, 60, 100, 140, 180, 220])

# %% [markdown]
# Part 2

# %%
for row in crt:
    print("".join(row))


# %% [markdown]
# ## Day 11
# https://adventofcode.com/2022/day/11

# %%
data = get_data(year=2022, day=11).split("\n\n")

operators = {
    "*": operator.mul,
    "+": operator.add,
    "pow": operator.pow,
}

regex = re.compile(
    r"Monkey \d+:\n  Starting items: (.*?)\n  Operation: new = old ([*+]) (\d+|old)\n  Test: divisible by (\d+)\n    If true: throw to monkey (\d+)\n    If false: throw to monkey (\d+)",
)

monkeys = []
for entry in data:
    levels, op, *nums = re.match(regex, entry).groups()

    if nums[0] == "old":
        op = "pow"
        nums[0] = 2

    monkeys.append((mapl(int, levels.split(", ")), operators[op], *mapl(int, nums)))

# %%
lcm = math.lcm(*map(operator.itemgetter(3), monkeys))


# %%
def run(monkeys, turns=20, divide=3):
    monkeys = copy.deepcopy(monkeys)
    counts = [0 for _ in range(len(monkeys))]
    for _ in range(turns):
        for m, (levels, op, val, limit, t, f) in enumerate(monkeys):
            while levels:
                level = (op(levels.pop(0), val) // divide) % lcm
                next_monkey_idx = t if (level % limit) == 0 else f
                monkeys[next_monkey_idx][0].append(level)
                counts[m] += 1

    return operator.mul(*sorted(counts)[-2:])


# %% [markdown]
# Part 1

# %%
run(monkeys, turns=20)

# %% [markdown]
# Part 2

# %%
run(monkeys, turns=10000, divide=1)

# %% [markdown]
# ## Day 12
# https://adventofcode.com/2022/day/12

# %%
data = get_data(year=2022, day=12)
data = np.array(mapl(list, data.split("\n")))

h = len(data)
w = len(data[0])

# %%
[[sy], [sx]] = np.where(data == "S")
[[ey], [ex]] = np.where(data == "E")

data[sy, sx] = "a"
data[ey, ex] = "z"

# %%
G = nx.DiGraph()

for y, x in itertools.product(*map(range, data.shape)):
    for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        qy, qx = y + dy, x + dx

        if 0 < qy < len(data) and 0 < qx < len(data[0]) and ord(data[qy, qx]) <= ord(data[y, x]) + 1:
            G.add_edge((y, x), (qy, qx))


# %% [markdown]
# Part 1

# %%
len(nx.shortest_path(G, (sy, sx), (ey, ex))) - 1

# %% [markdown]
# Part 2

# %%
path_length = float("inf")
for pos in zip(*np.where(data == "a")):
    try:
        path_length = min(path_length, len(nx.shortest_path(G, pos, (ey, ex))) - 1)
    except:
        pass

path_length


# %% [markdown]
# ## Day 13
# https://adventofcode.com/2022/day/13

# %%
data = get_data(year=2022, day=13).split("\n\n")
data = [mapl(eval, entry.split("\n")) for entry in data]


# %%
def compare(left, right):
    match left, right:
        case int(), int():
            if left < right:
                return 1

            if left == right:
                return 0

            return -1

        case list(), list():
            for a, b in zip(left, right):
                if value := compare(a, b):
                    return value

            if len(left) < len(right):
                return 1

            if len(left) > len(right):
                return -1

            return 0

        case int(), list():
            return compare([left], right)
        case _:
            return compare(left, [right])


# %% [markdown]
# Part 1

# %%
sum(i for i, (left, right) in enumerate(data, 1) if compare(left, right) == 1)

# %% [markdown]
# Part 2

# %%
data = [
    [[2]],
    [[6]],
    *[entry for entries in data for entry in entries],
]

data = sorted(data2, key=ft.cmp_to_key(lambda a, b: -compare(a, b)))

# %%
count = 1
for i, k in enumerate(data2, 1):
    if k in [[[2]], [[6]]]:
        S *= i

count

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
