import functools
import itertools
import math
import re


def mapl(func, *args):
    return list(map(func, *args))


def first(l):
    return next(iter(l))


def is_unique(l):
    return len(set(l)) == len(l)


def is_prime(n):
    return n > 1 and all(n % i for i in itertools.islice(itertools.count(2), int(math.sqrt(n) - 1)))


def sort(items):
    cast = "".join if isinstance(items, str) else tuple
    return cast(sorted(items))


def add(x, y):
    return mapl(sum, zip(x, y))


def mult(x, value):
    return [i * value for i in x]


def extract_ints(line):
    return [int(x) for x in re.findall(r"-?\d+", line)]


def diff(a, b):
    return abs(a - b)


def lcm(lst):
    return functools.reduce(lambda a, b: (a * b) // math.gcd(a, b), lst)
