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
# # AOC 2019

from collections import abc

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
import more_itertools
import networkx as nx
import numpy as np
import scipy


# %% [markdown]
# -----

# %%
def get_instructions(year, day):
    return mapl(int, get_data(year=year, day=day).split(","))


# %%
class IntCode(abc.MutableMapping):
    def __init__(self, instructions, inputs=None, relative_base=0):
        self.memory = collections.defaultdict(int, enumerate(instructions))
        self.inputs = inputs or []
        self.relative_base = relative_base
        self.ptr = 0

    def __getitem__(self, index):
        return self.memory[index]

    def __setitem__(self, index, value):
        self.memory[index] = value

    def __delitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        yield from self.step()

    def __len__(self):
        return len(self.memory)

    def _value(self, ptr, mode):
        match mode:
            case 0:
                return self[ptr]
            case 1:
                return ptr
            case 2:
                return self.relative_base + self[ptr]

    def get_instruction(self):
        value = self[self.ptr]
        return (
            value % 100,
            self._value(self.ptr + 1, (value % 1000) // 100),
            self._value(self.ptr + 2, (value % 10000) // 1000),
            self._value(self.ptr + 3, (value % 100000) // 10000),
        )

    def step(self, stop_on_empty_inputs=False):
        while True:
            opcode, a, b, c = self.get_instruction()
            # print(f"{self.ptr: >2d}", self.memory)
            match opcode:
                case 1:
                    self[c] = self[a] + self[b]
                    self.ptr += 4
                case 2:
                    self[c] = self[a] * self[b]
                    self.ptr += 4
                case 3:
                    if not self.inputs and stop_on_empty_inputs:
                        return

                    self[a] = self.inputs.pop(0)
                    self.ptr += 2
                case 4:
                    self.ptr += 2
                    yield self[a]
                case 5:
                    self.ptr = self[b] if self[a] else self.ptr + 3
                case 6:
                    self.ptr = self[b] if not self[a] else self.ptr + 3
                case 7:
                    self[c] = int(self[a] < self[b])
                    self.ptr += 4
                case 8:
                    self[c] = int(self[a] == self[b])
                    self.ptr += 4
                case 9:
                    self.relative_base += self[a]
                    self.ptr += 2
                case 99:
                    return

    def accumulate(self):
        return list(self.step())


# %% [markdown]
# -----

# %% [markdown]
# ## Day 01
# https://adventofcode.com/2019/day/1

# %%
data = mapl(int, get_data(year=2019, day=1).split("\n"))


# %% [markdown]
# Part 1

# %% code_folding=[0]
def get_fuel(mass):
    return max(0, math.floor(mass / 3) - 2)


sum(map(get_fuel, data))


# %% [markdown]
# Part 2

# %%
def get_over_all_fuel(mass):
    if mass <= 0:
        return 0

    fuel = get_fuel(mass)
    return fuel + get_over_all_fuel(fuel)


sum(map(get_over_all_fuel, data))

# %% [markdown]
# ## Day 02
# https://adventofcode.com/2019/day/2

# %%
data = get_instructions(year=2019, day=2)

# %% [markdown]
# Part 1

# %%
prog = IntCode(data)
prog[1] = 12
prog[2] = 2

prog.accumulate()

prog[0]

# %% [markdown]
# Part 2

# %%
for i in range(100):
    for j in range(100):
        prog = IntCode(data)
        prog[1] = i
        prog[2] = j

        prog.accumulate()

        if prog[0] == 19690720:
            print(100 * i + j)
            break

# %% [markdown]
# ## Day 03
# https://adventofcode.com/2019/day/3

# %%
path1, path2 = get_data(year=2019, day=3).split("\n")
path1 = path1.split(",")
path2 = path2.split(",")


# %%
directions = {
    "R": (1, 0),
    "L": (-1, 0),
    "U": (0, 1),
    "D": (0, -1),
}


def get_positions(path):
    positions = list()
    current_position = (0, 0)
    for step in path:
        direction, number = step[0], int(step[1:])
        for i in range(number):
            current_position = tuple(add(current_position, directions[direction]))
            positions.append(current_position)

    return positions


# %%
positions_path1 = get_positions(path1)
positions_path2 = get_positions(path2)

# %%
commun = set(positions_path1).intersection(set(positions_path2))

# %% [markdown]
# Part 1

# %%
min(abs(x) + abs(y) for x, y in commun)

# %% [markdown]
# Part 2

# %%
steps = []
for c in commun:
    idx1 = positions_path1.index(c) + 1
    idx2 = positions_path2.index(c) + 1
    steps.append(idx1 + idx2)

min(steps)

# %% [markdown]
# ## Day 04
# https://adventofcode.com/2019/day/4

# %%
low, high = mapl(int, get_data(year=2019, day=4).split("-"))


# %%
def never_decrease(numbers):
    return all(x <= y for x, y in zip(numbers, numbers[1:]))


def contains_a_pair(numbers):
    return any(x == y for x, y in zip(numbers, numbers[1:]))


# %%
pwds = []
for pwd in range(low, high + 1):
    numbers = mapl(int, str(pwd))
    if never_decrease(numbers) and contains_a_pair(numbers):
        pwds.append(numbers)

# %% [markdown]
# Part 1

# %%
len(pwds)

# %% [markdown]
# Part 2

# %%
sum(2 in collections.Counter(pwd).values() for pwd in pwds)

# %% [markdown]
# ## Day 05
# https://adventofcode.com/2019/day/5

# %%
data = get_instructions(year=2019, day=5)


# %% [markdown]
# Part 1

# %%
prog = IntCode(data, [1])
prog.accumulate()[-1]

# %% [markdown]
# Part 2

# %%
prog = IntCode(data, [5])
prog.accumulate()[-1]

# %% [markdown]
# ## Day 06
# https://adventofcode.com/2019/day/6

# %%
data = get_data(year=2019, day=6).split("\n")
data = [entry.split(")") for entry in data]


# %%
G = nx.DiGraph(incoming_graph_data=data)

# %% [markdown]
# Part 1

# %%
sum(len(nx.ancestors(G, node)) for node in G.nodes())

# %% [markdown]
# Part 2

# %%
nx.shortest_path_length(G.to_undirected(), "YOU", "SAN") - 2

# %% [markdown]
# ## Day 07
# https://adventofcode.com/2019/day/7

# %%
data = get_instructions(year=2019, day=7)


# %% [markdown]
# Part 1

# %%
max_amp = 0
for phases in itertools.permutations(range(5)):
    amp = 0
    for p in phases:
        amp = next(IntCode(data, [p, amp]).step())

    max_amp = max(max_amp, amp)

max_amp

# %% [markdown]
# Part 2

# %%
max_amp = 0
for phases in itertools.permutations(range(5, 10)):
    progs = [IntCode(data.copy(), [p]) for p in phases]
    progs[0].inputs.append(0)

    while True:
        for i in range(5):
            amp = next(progs[i].step())
            progs[(i + 1) % 5].inputs.append(amp)

        if amp is None:
            break

        max_amp = max(amp, max_amp)

max_amp

# %% [markdown]
# ## Day 08
# https://adventofcode.com/2019/day/8

# %%
data = get_data(year=2019, day=8)


# %%
width = 25
height = 6
layers_count = (len(data) // width) // height

# %%
layers = np.array(list(data)).reshape(layers_count, height, width).astype(int)

# %% [markdown]
# Part 1

# %%
min_zeros = sys.maxsize
output = None
for l in range(layers_count):
    counter = collections.Counter(layers[l].flatten())
    if counter[0] < min_zeros:
        output = counter[1] * counter[2]
        min_zeros = counter[0]

output

# %% [markdown]
# Part 2

# %%
for y in range(height):
    for x in range(width):
        val = next(e for e in np.array(layers)[:, y, x] if e != 2)

        if val == 2:
            print("?", end="")
        elif val == 1:
            print("#", end="")
        elif val == 0:
            print(" ", end="")
    print()

# %% [markdown]
# ## Day 09
# https://adventofcode.com/2019/day/9

# %%
data = get_instructions(year=2019, day=9)


# %% [markdown]
# Part 1

# %%
next(IntCode(data, [1]).step())

# %% [markdown]
# Part 2

# %%
next(IntCode(data, [2]).step())

# %% [markdown]
# ## Day 10
# https://adventofcode.com/2019/day/10

# %%
data = get_data(year=2019, day=10).split("\n")

asteroids_map = np.array(mapl(list, data))
asteroids_positions = list(zip(*np.where(asteroids_map == "#")))

# %%
stations2neighbors = collections.defaultdict(set)

for sy, sx in asteroids_positions:
    for ay, ax in asteroids_positions:
        if (sy, sx) == (ay, ax):
            continue

        y, x = (ay - sy, ax - sx)
        gcd = abs(math.gcd(y, x))
        stations2neighbors[(sx, sy)].add((x // gcd, y // gcd))

# %% [markdown]
# Part 1

# %%
max_value, station = max((len(x), pos) for pos, x in stations2neighbors.items())
max_value

# %% [markdown]
# Part 2

# %%
to_zap = sorted(((math.atan2(dx, dy), (dx, dy)) for dx, dy in stations2neighbors[station]), reverse=True)

_, (dx, dy) = to_zap[200 - 1]

x, y = station[0] + dx, station[1] + dy
while (x, y) not in asteroids_positions:
    x, y = x + dx, y + dy

x * 100 + y


# %% [markdown]
# ## Day 11
# https://adventofcode.com/2019/day/11

# %%
data = get_instructions(year=2019, day=11)

# %%
directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def paint(instructions, start_color=0):
    position = (0, 0)
    direction = 0

    panels = collections.defaultdict(int, [(position, start_color)])

    prog = IntCode(instructions, [panels[position]])
    iterator = prog.step()

    for o1, o2 in zip(iterator, iterator):
        panels[position] = o1

        match o2:
            case 0:
                direction = (direction - 1) % 4
            case 1:
                direction = (direction + 1) % 4

        position = tuple(add(position, directions[direction]))

        prog.inputs.append(panels[position])

    return panels


len(paint(data, start_color=0))

# %% [markdown]
# Part 1

# %%
panels = paint(data, start_color=1)

xs = [x for x, y in panels]
ys = [y for x, y in panels]

for y in range(min(ys), max(ys) + 1):
    for x in range(min(xs), max(xs) + 1):
        print("#" if panels[(x, y)] else " ", end="")
    print()


# %% [markdown]
# ## Day 12
# https://adventofcode.com/2019/day/12

# %%
data = get_data(year=2019, day=12).split("\n")

# %%
planets = np.array([extract_ints(line) for line in data])
combinations = np.array(list(itertools.combinations(range(len(planets)), 2)))

# %%
base_state = np.zeros((len(planets), 6), dtype=int)
base_state[:, :3] = planets


# %%
def step(state):
    np.add.at(state[:, 3:], combinations, np.sign(state[combinations, :3][:, [1, 0]] - state[combinations, :3][:, [0, 1]]))
    state[:, :3] += state[:, 3:]


# %% [markdown]
# Part 1

# %%
state = base_state.copy()

for _ in range(1000):
    step(state)

(np.abs(state[:, :3]).sum(axis=1) * np.abs(state[:, 3:]).sum(axis=1)).sum()

# %% [markdown]
# Part 2

# %%
state = base_state.copy()

loops = {}
count = 1
while True:
    step(state)
    count += 1

    for i in range(3):
        if i not in loops and np.array_equal(base_state[:, i], state[:, i]):
            loops[i] = count

    if len(loops) == 3:
        break

lcm(loops.values())

# %% [markdown]
# ## Day 13
# https://adventofcode.com/2019/day/13

# %%
data = get_instructions(year=2019, day=13)

# %% [markdown]
# Part 1

# %%
prog = IntCode(data)
collections.Counter(list(prog)[2::3])[2]

# %% [markdown]
# Part 2

# %%
prog = IntCode(data)
prog[0] = 2

move = 0
score = 0
ball = (0, 0)
paddle = (0, 0)

pixels = collections.defaultdict(int)

while True:
    prog.inputs.append(move)

    iterator = prog.step(stop_on_empty_inputs=True)
    for x, y, tile_id in more_itertools.grouper(iterator, 3, incomplete="strict"):

        if (x, y) == (-1, 0):
            score = tile_id
        else:
            pixels[(x, y)] = tile_id

            match tile_id:
                case 4:
                    ball = (x, y)
                case 3:
                    paddle = (x, y)

    if ball[0] < paddle[0]:
        move = -1
    elif ball[0] > paddle[0]:
        move = 1
    else:
        move = 0

    if collections.Counter(pixels.values()).get(2, 0) == 0:
        break

# %%
score

# %% [markdown]
# ## Day 14
# https://adventofcode.com/2019/day/14
# %%
data = get_data(year=2019, day=14).split("\n")
data = [mapl(lambda x: (int(x[0]), x[1]), re.findall(r"(\d+) (\w+)", line)) for line in data]

# %%
reactions = {}
for *inputs, (out_qty, out_elem) in data:
    reactions[out_elem] = (out_qty, inputs)


# %% [markdown]
# Part 1

# %%
def get_required_ore(fuel_amount):

    needed = {"FUEL": fuel_amount}

    def get_requirements():
        return [elem for elem, qty in needed.items() if qty > 0 and elem != "ORE"]

    while get_requirements():
        for element in get_requirements():
            qty, inputs = reactions[element]
            scaling = (needed[element] + qty - 1) // qty

            for (src_qty, src_elem) in inputs:
                needed[src_elem] = needed.get(src_elem, 0) + scaling * src_qty

            needed[element] -= scaling * qty

    return needed["ORE"]


# %%
get_required_ore(1)


# %% [markdown]
# Part 2

# %%
ore_low = 1
ore_high = 10
ore_amount = 10e11

while get_required_ore(ore_high) <= ore_amount:
    ore_high *= 10

# Use binary search
while ore_high - ore_low > 1:
    ore_middle = (ore_high + ore_low) // 2
    if get_required_ore(ore_middle) <= ore_amount:
        ore_low = ore_middle
    else:
        ore_high = ore_middle

# %%
print(ore_low)

# %% [markdown]
# ## Day 15
# https://adventofcode.com/2019/day/15
# %%
data = get_data(year=2019, day=15)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 16
# https://adventofcode.com/2019/day/16
# %%
data = mapl(int, get_data(year=2019, day=16))

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 17
# https://adventofcode.com/2019/day/17
# %%
data = get_data(year=2019, day=17)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 18
# https://adventofcode.com/2019/day/18
# %%
data = get_data(year=2019, day=18)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 19
# https://adventofcode.com/2019/day/19
# %%
data = get_data(year=2019, day=19)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 20
# https://adventofcode.com/2019/day/20
# %%
data = get_data(year=2019, day=20)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 21
# https://adventofcode.com/2019/day/21
# %%
data = get_data(year=2019, day=21)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 22
# https://adventofcode.com/2019/day/22
# %%
data = get_data(year=2019, day=22)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 23
# https://adventofcode.com/2019/day/23
# %%
data = get_data(year=2019, day=23)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 24
# https://adventofcode.com/2019/day/24
# %%
data = get_data(year=2019, day=24)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%


# %% [markdown]
# ## Day 25
# https://adventofcode.com/2019/day/25
# %%

# %%
data = get_instructions(year=2019, day=25)

# %% [markdown]
# Part 1

# %%

# %% [markdown]
# Part 2

# %%
