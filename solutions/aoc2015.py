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
# # AOC 2015

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
# https://adventofcode.com/2015/day/1

# %%
data = get_data(year=2015, day=1)

# %% [markdown]
# Part 1

# %%
count = collections.Counter(data)
count["("] - count[")"]

# %% [markdown]
# Part 2

# %%
current = 0
for idx, c in enumerate(data):
    current += 1 if c == "(" else -1
    if current == -1:
        break


# %%
idx + 1

# %% [markdown]
# ## Day 02
# https://adventofcode.com/2015/day/2

# %%
data = get_data(year=2015, day=2)


# %% [markdown]
# Part 1

# %%
def area(l, w, h):
    return 2 * l * w + 2 * w * h + 2 * h * l + min(w * l, w * h, h * l)


sum(area(l, w, h) for l, w, h in mapl(extract_ints, data.split()))


# %% [markdown]
# Part 2

# %%
def ribbon(l, w, h):
    return 2 * min(l + w, l + h, w + h) + l * w * h


sum(ribbon(l, w, h) for l, w, h in mapl(extract_ints, data.split()))

# %% [markdown]
# ## Day 03
# https://adventofcode.com/2015/day/3

# %%
data = get_data(year=2015, day=3)

# %%
directions = {
    ">": (0, 1),
    "<": (0, -1),
    "v": (1, 0),
    "^": (-1, 0),
}

# %% [markdown]
# Part 1

# %%
pos = (0, 0)
houses = collections.Counter([pos])

for val in mapl(lambda x: directions[x], data):
    pos = tuple(add(pos, val))
    houses[pos] += 1

len(houses)

# %% [markdown]
# Part 2

# %%
pos = [(0, 0), (0, 0)]
houses = collections.Counter(pos)

for idx, val in enumerate(mapl(lambda x: directions[x], data)):
    pos[idx % 2] = tuple(add(pos[idx % 2], val))
    houses[pos[idx % 2]] += 1

len(houses)

# %% [markdown]
# ## Day 04
# https://adventofcode.com/2015/day/4

# %%
data = get_data(year=2015, day=4)

# %% [markdown]
# Part 1

# %%
i = 0
for i in range(int(10e9)):
    if hashlib.md5(f"{data}{i}".encode()).hexdigest().startswith("00000"):
        break

print(i)

# %% [markdown]
# Part 2

# %%
i = 0
for i in range(int(10e8)):
    if hashlib.md5(f"{data}{i}".encode()).hexdigest().startswith("000000"):
        break

print(i)

# %% [markdown]
# ## Day 05
# https://adventofcode.com/2015/day/5

# %%
data = get_data(year=2015, day=5).split()


# %% [markdown]
# Part 1

# %%
def is_nice(text):
    return all([len(re.findall(r"[aeiou]", text)) >= 3, not re.search(r"ab|cd|pq|xy", text), re.search(r"(.)\1", text)])


sum(map(is_nice, data))


# %% [markdown]
# Part 2

# %%
def is_nice(text):
    return all(
        [
            re.search(r"(..).*\1", text),
            re.search(r"(.).\1", text),
        ]
    )


sum(map(is_nice, data))

# %% [markdown]
# ## Day 06
# https://adventofcode.com/2015/day/6

# %%
data = get_data(year=2015, day=6).split("\n")

# %% [markdown]
# Part 1

# %%
grid = np.zeros((1000, 1000)).astype(bool)

for value in data:
    dx1, dy1, dx2, dy2 = extract_ints(value)
    if "on" in value:
        grid[dx1 : dx2 + 1, dy1 : dy2 + 1] = True

    if "off" in value:
        grid[dx1 : dx2 + 1, dy1 : dy2 + 1] = False

    if "toggle" in value:
        grid[dx1 : dx2 + 1, dy1 : dy2 + 1] = ~grid[dx1 : dx2 + 1, dy1 : dy2 + 1]

grid.sum()

# %% [markdown]
# Part 2

# %%
grid = np.zeros((1000, 1000))

for value in data:
    dx1, dy1, dx2, dy2 = extract_ints(value)
    if "on" in value:
        grid[dx1 : dx2 + 1, dy1 : dy2 + 1] += 1

    if "off" in value:
        grid[dx1 : dx2 + 1, dy1 : dy2 + 1] -= 1

    if "toggle" in value:
        grid[dx1 : dx2 + 1, dy1 : dy2 + 1] += 2

    grid = np.clip(grid, 0, float("inf"))

grid.sum()

# %% [markdown]
# ## Day 07
# https://adventofcode.com/2015/day/7

# %%
OPERATORS = {
    "AND": operator.and_,
    "OR": operator.or_,
    "LSHIFT": operator.lshift,
    "RSHIFT": operator.rshift,
}


def apply(cmds):
    data = list(cmds)
    register = {}

    while data:

        next_data = []
        for entry in data:
            if re.search("RSHIFT|LSHIFT|AND|OR", entry):
                key1, cmd, key2, out = re.match(r"(\w+) (\w+) (\w+) -> (\w+)", entry).groups()

                value1 = int(key1) if re.match(r"\d+", key1) else register.get(key1)
                value2 = int(key2) if re.match(r"\d+", key2) else register.get(key2)

                if value1 is not None and value2 is not None:
                    register[out] = OPERATORS[cmd](value1, value2)
                    continue

            if "NOT" in entry:
                key, out = re.match(r"NOT (\w+) -> (\w+)", entry).groups()
                if key in register:
                    register[out] = ~register[key] & 0xFFFF
                    continue

            if re.match(r"(\w+) -> (\w+)", entry):
                key, out = re.match(r"(\w+) -> (\w+)", entry).groups()
                value = int(key) if re.match(r"\d+", key) else register.get(key)
                if value is not None:
                    register[out] = value
                    continue

            next_data.append(entry)

        if len(next_data) == len(data):
            raise

        data = next_data

    return register["a"]


# %% [markdown]
# Part 1

# %%
data = get_data(year=2015, day=7).split("\n")
apply(data)

# %% [markdown]
# Part 2

# %%
apply([re.sub(r"^\d+ -> b$", "956 -> b", value) for value in data])

# %% [markdown]
# ## Day 08
# https://adventofcode.com/2015/day/8

# %%
data = get_data(year=2015, day=8).split("\n")

# %% [markdown]
# Part 1

# %%
sum(len(s) - len(eval(s)) for s in data)

# %% [markdown]
# Part 2

# %%
sum(len(repr(s).replace('"', '\\"')) - len(s) for s in data)

# %% [markdown]
# ## Day 09
# https://adventofcode.com/2015/day/9

# %%
data = get_data(year=2015, day=9).split("\n")
data = [re.match(r"(\w+) to (\w+) = (\d+)", value).groups() for value in data]
data = {(k1, k2): int(v) for k1, k2, v in data}

# %%
towns = set(np.array(mapl(list, data.keys())).flatten())

# %%
permutations = list(itertools.permutations(towns))

# %%
distances = [sum(data.get((a, b), data.get((b, a), 999999)) for a, b in zip(perm, perm[1:])) for perm in permutations]

# %% [markdown]
# Part 1

# %%
min(distances)

# %% [markdown]
# Part 2

# %%
max(distances)

# %% [markdown]
# ## Day 10
# https://adventofcode.com/2015/day/10

# %%
data = get_data(year=2015, day=10)


# %% [markdown]
# [`itertools.groupby`](https://docs.python.org/3/library/itertools.html#itertools.groupby)
#
# > Make an iterator that returns consecutive keys and groups from the iterable.
#
#     ``` [k for k, g in groupby('AAAABBBCCDAABBB')] --> A B C D A B`
#

# %%
def apply(count):
    s = data
    for i in range(count):
        s = "".join([f"{len(list(value))}{key}" for key, value in itertools.groupby(s)])

    return s


# %% [markdown]
# Part 1

# %%
len(apply(40))

# %% [markdown]
# Part 2

# %%
len(apply(50))

# %% [markdown]
# ## Day 11
# https://adventofcode.com/2015/day/11

# %%
password = get_data(year=2015, day=11)


# %% [markdown]
# [`re.sub`](https://docs.python.org/3/library/re.html#re.sub)
#
# > If repl is a function, it is called for every non-overlapping occurrence of pattern. The function takes a single match object argument, and returns the replacement string.

# %%
def has_increasing_straight(password):
    return any(z - y == 1 and y - x == 1 for x, y, z in map(lambda x: map(ord, x), zip(password, password[1:], password[2:])))


def has_forbidden_letters(password):
    return bool(re.findall("[iol]", password))


def has_pairs(password):
    return len(re.findall(r"([a-z])\1", password)) >= 2


def increment(password):
    def wrap(pre, zs):
        return chr(ord(pre) + 1) + len(zs) * "a"

    return re.sub(r"([a-y])(z*)$", lambda x: wrap(*x.groups()), password)


def find_next(password):
    while True:
        password = increment(password)

        if has_increasing_straight(password) and not has_forbidden_letters(password) and has_pairs(password):
            break

    return password


# %% [markdown]
# Part 1

# %%
password = find_next(password)
password

# %% [markdown]
# Part 2

# %%
password = find_next(password)
password

# %% [markdown]
# ## Day 12
# https://adventofcode.com/2015/day/12

# %%
data = get_data(year=2015, day=12)


# %%
def get_sum(data):
    return sum(map(int, re.findall(r"-?\d+", data)))


# %% [markdown]
# Part 1

# %%
get_sum(data)

# %% [markdown]
# Part 2
#
# [`object_hook`](https://docs.python.org/3/library/json.html#json.load)
#
# > object_hook is an optional function that will be called with the result of any object literal decoded (a dict). The return value of object_hook will be used instead of the dict. This feature can be used to implement custom decoders (e.g. JSON-RPC class hinting).

# %%
get_sum(str(json.loads(data, object_hook=lambda obj: {} if "red" in obj.values() else obj)))

# %% [markdown]
# ## Day 13
# https://adventofcode.com/2015/day/13

# %%
data = get_data(year=2015, day=13).split("\n")

# %%
pattern = r"(\w+) would (\w+) (\d+) happiness units by sitting next to (\w+)."

people = set()
preferences = collections.defaultdict(int)
for entry in data:
    src, action, pts, tgt = re.match(pattern, entry).groups()
    people.update([src, tgt])
    preferences[tuple(sorted([src, tgt]))] += int(pts) * (-1 if action == "lose" else 1)


# %%
def get_best_points(people, preferences):
    totals = []
    for permutation in map(list, itertools.permutations(people)):
        totals.append(sum(preferences[tuple(sorted([x, y]))] for x, y in zip(permutation, permutation[-1:] + permutation[:-1])))

    return max(totals)


# %% [markdown]
# Part 1

# %%
get_best_points(people, preferences)

# %% [markdown]
# Part 2

# %%
for person in people:
    preferences[("0", person)] = 0

people.add("0")

get_best_points(people, preferences)

# %% [markdown]
# ## Day 14
# https://adventofcode.com/2015/day/14

# %%
data = get_data(year=2015, day=14).split("\n")

pattern = r"(\w+) can fly (\d+) km/s for (\d+) seconds, but then must rest for (\d+) seconds."
values = [re.match(pattern, entry).groups() for entry in data]

# %% [markdown]
# Part 1

# %%
history = collections.defaultdict()
for reindeer, speed, duration, rest_time in values:
    steps = itertools.cycle([int(speed)] * int(duration) + [0] * int(rest_time))
    history[reindeer] = list(itertools.accumulate(next(steps) for _ in range(2503)))

# %%
max(vals[-1] for vals in history.values())

# %% [markdown]
# Part 2

# %%
scored = [idx for entries in zip(*history.values()) for idx, v in enumerate(entries) if v == max(entries)]

# %%
max(collections.Counter(scored).values())

# %% [markdown]
# ## Day 15
# https://adventofcode.com/2015/day/15

# %%
data = get_data(year=2015, day=15).split("\n")

pattern = r".+: capacity (-?\d+), durability (-?\d+), flavor (-?\d+), texture (-?\d+), calories (-?\d+)"
data = np.array(mapl(lambda x: re.match(pattern, x).groups(), data)).astype(int)

# %%
scores = np.clip(
    np.array(
        [
            mapl(operator.mul, [i, j, k, l], data[:, :])
            for i in range(101)
            for j in range(0, 101 - i)
            for k in range(0, 101 - j - i)
            for l in [100 - i - j - k]
        ]
    ).sum(axis=1),
    0,
    np.inf,
)

# %% [markdown]
# Part 1

# %%
np.prod(scores[:, :-1], axis=1).max()

# %% [markdown]
# Part 2

# %%
np.prod(scores[scores[:, -1] == 500][:, :-1], axis=1).max()

# %% [markdown]
# ## Day 16
# https://adventofcode.com/2015/day/16

# %%
data = get_data(year=2015, day=16).split("\n")

pattern = r"Sue (\d+): (\w+): (\d+), (\w+): (\d+), (\w+): (\d+)"
data = [re.match(pattern, x).groups() for x in data]
data = {k: dict(zip(kv[::2], kv[1::2])) for k, *kv in data}

# %% [markdown]
# Part 1

# %%
aunt = {
    "children": 3,
    "cats": 7,
    "trees": 3,
    "samoyeds": 2,
    "akitas": 0,
    "vizslas": 0,
    "pomeranians": 3,
    "goldfish": 5,
    "cars": 2,
    "perfumes": 1,
}

next(i for i, values in data.items() if all(aunt[key] == int(value) for key, value in values.items()))

# %% [markdown]
# Part 2

# %%
aunt = {
    "children": lambda x: int(x) == 3,
    "cats": lambda x: int(x) >= 7,
    "trees": lambda x: int(x) >= 3,
    "samoyeds": lambda x: int(x) == 2,
    "akitas": lambda x: int(x) == 0,
    "vizslas": lambda x: int(x) == 0,
    "pomeranians": lambda x: int(x) <= 3,
    "goldfish": lambda x: int(x) <= 5,
    "cars": lambda x: int(x) == 2,
    "perfumes": lambda x: int(x) == 1,
}

next(i for i, values in data.items() if all(aunt[key](value) for key, value in values.items()))

# %% [markdown]
# ## Day 17
# https://adventofcode.com/2015/day/17

# %%
data = mapl(int, get_data(year=2015, day=17).split("\n"))

# %% [markdown]
# Part 1

# %%
combinations = [combination for i in range(len(data)) for combination in itertools.combinations(data, i) if sum(combination) == 150]
len(combinations)

# %% [markdown]
# Part 2

# %%
min_legnth = len(min(combinations, key=len))
len([c for c in combinations if len(c) == min_legnth])

# %% [markdown]
# ## Day 18
# https://adventofcode.com/2015/day/18

# %% [markdown]
# Definitely not the fastest. An excellent and almost instantaneous solution can be found [here](https://www.reddit.com/r/adventofcode/comments/3xb3cj/comment/cy368tv/?utm_source=share&utm_medium=web2x&context=3).
#
# To be faster, you are better off not doing a stupid convolution (easy but slow). Instead you can create an array where each position indicates the count of neighbors `on`. Then you can (based on the position status (`on` / `off`) check for constrains and modifiy the status of the light accordingly. This is much faster because it relies only on numpy functions.

# %%
from scipy import ndimage


# %%
data = np.array(mapl(list, get_data(year=2015, day=18).split())) == "#"


# %%
def kernel(x):
    return sum(x) - 1 in [2, 3] if x[4] else sum(x) == 3


# %% [markdown]
# Part 1
#
# [`scipy.ndimage.generic_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html)
#
# > Calculate a multidimensional filter using the given function. At each element the provided function is called. The input values within the filter footprint at that element are passed to the function as a 1-D array of double values.

# %%
grid = np.copy(data)
for _ in range(100):
    grid = scipy.ndimage.generic_filter(
        grid,
        kernel,
        size=(3, 3),
        mode="constant",
    )

grid.sum()

# %% [markdown]
# Part 2

# %%
grid = np.copy(data)
grid[[0, 99, 0, 99], [0, 0, 99, 99]] = True

for _ in range(100):
    grid = ndimage.generic_filter(
        grid,
        kernel,
        size=(3, 3),
        mode="constant",
    )

    grid[[0, 99, 0, 99], [0, 0, 99, 99]] = True

grid.sum()

# %% [markdown]
# ## Day 19
# https://adventofcode.com/2015/day/19

# %%
*replacements, _, medicine = get_data(year=2015, day=19).split("\n")
replacements = [entry.split(" => ") for entry in replacements]

# %% [markdown]
# Part 1

# %%
solutions = set()
for frm, to in replacements:
    for i in range(len(medicine)):
        if medicine[i : i + len(frm)] == frm:
            solutions.add(medicine[:i] + to + medicine[i + len(frm) :])

len(solutions)

# %% [markdown]
# Part 2

# %%
count = 0
mol = medicine
while len(mol) > 1:
    start = mol
    for frm, to in replacements:
        while to in mol:
            count += mol.count(to)
            mol = mol.replace(to, frm)

count

# %% [markdown]
# ## Day 20
# https://adventofcode.com/2015/day/20

# %%
data = int(get_data(year=2015, day=20))

# %% [markdown]
# Part 1

# %%
houses = np.zeros(1000000)
for i in range(1, 1000000):
    houses[i::i] += 10 * i

np.nonzero(houses >= data)[0][0]

# %% [markdown]
# Part 2

# %%
houses = np.zeros(1000000)
for i in range(1, 1000000):
    houses[i : (i + 1) * 50 : i] += 11 * i

np.nonzero(houses >= data)[0][0]

# %% [markdown]
# ## Day 21
# https://adventofcode.com/2015/day/21

# %%
data = get_data(year=2015, day=21)
hit_points, damage, armor = map(int, re.findall(r"(\d+)", data, re.M))

# %%
weapons = [
    (8, 4, 0),
    (10, 5, 0),
    (25, 6, 0),
    (40, 7, 0),
    (74, 8, 0),
]

armours = [
    (0, 0, 0),
    (13, 0, 1),
    (31, 0, 2),
    (53, 0, 3),
    (75, 0, 4),
    (102, 0, 5),
]

rings = [
    (0, 0, 0),
    (0, 0, 0),
    (25, 1, 0),
    (50, 2, 0),
    (100, 3, 0),
    (20, 0, 1),
    (40, 0, 2),
    (80, 0, 3),
]

# %% [markdown]
# Part 1

# %%
min(
    wc + ac + r1c + r2c
    for wc, wd, wa in weapons
    for ac, ad, aa in armours
    for (r1c, r1d, r1a), (r2c, r2d, r2a) in itertools.combinations(rings, 2)
    if (hit_points // max(1, wd + ad + r1d + r2d - armor)) <= 100 // max(1, damage - wa - aa - r1a - r2a)
)

# %% [markdown]
# Part 2

# %%
max(
    wc + ac + r1c + r2c
    for wc, wd, wa in weapons
    for ac, ad, aa in armours
    for (r1c, r1d, r1a), (r2c, r2d, r2a) in itertools.combinations(rings, 2)
    if (hit_points // max(1, wd + ad + r1d + r2d - armor)) > 100 // max(1, damage - wa - aa - r1a - r2a)
)

# %% [markdown]
# ## Day 22
# https://adventofcode.com/2015/day/22

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%

# %% [markdown]
# ## Day 23
# https://adventofcode.com/2015/day/23

# %%
data = get_data(year=2015, day=23)
instructions = re.findall(r"^(\w+) (\S+)(?:, (\S+))?$", data, re.M)


# %%
def execute(a, b):
    reg = {
        "a": a,
        "b": b,
    }

    idx = 0
    while 0 <= idx < len(instructions):
        inst, a, b = instructions[idx]

        match inst:
            case "hlf":
                reg[a] //= 2
            case "tpl":
                reg[a] *= 3
            case "inc":
                reg[a] += 1
            case "jmp":
                idx += int(a) - 1
            case "jie":
                idx += (int(b) - 1) * (reg[a] % 2 == 0)
            case "jio":
                idx += (int(b) - 1) * (reg[a] == 1)

        idx += 1

    return reg


# %% [markdown]
# Part 1

# %%
execute(0, 0)["b"]

# %% [markdown]
# Part 2

# %%
execute(1, 0)["b"]

# %% [markdown]
# ## Day 24
# https://adventofcode.com/2015/day/24

# %%
data = mapl(int, get_data(year=2015, day=24).split("\n"))

# %% [markdown]
# Part 1

# %%
size_goal = sum(data) // 3
min(functools.reduce(lambda x, y: x * y, combo) for i in range(10) for combo in itertools.combinations(data, i) if sum(combo) == size_goal and combo)

# %% [markdown]
# Part 2

# %%
size_goal = sum(data) // 4
min(functools.reduce(lambda x, y: x * y, combo) for i in range(10) for combo in itertools.combinations(data, i) if sum(combo) == size_goal and combo)

# %% [markdown]
# ## Day 25
# https://adventofcode.com/2015/day/25

# %%
start = 20151125
row, column = (3010, 3019)

# %% [markdown]
# Part 1

# %%
code_count = sum(range(row + column)) - row

current = start
for i in range(code_count):
    current = (current * 252533) % 33554393

current
