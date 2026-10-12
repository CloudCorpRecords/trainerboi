"""
math_gym.py - an unlimited, self-checking maths gym.

Why this matters: every problem here is generated with its exact answer, so you need NO dataset and NO human labels. The model can
practise forever, we know instantly whether it was right, and we can aim practice at the skills it is weakest at (adaptive curriculum).

  problem(skill, level, rng)  -> {"question", "answer", "skill", "level"}
  make_set(n, levels, skills, weights, seed) -> list of problems
  extract_answer(text) / is_correct(text, gold)   robust checking (\\boxed{}, ####, "answer is", fractions, decimals, %, commas)
  SKILLS, INSTRUCTION
"""

import math
import random
import re
from fractions import Fraction

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

NAMES = ["Ava", "Ben", "Chen", "Dana", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jade", "Kofi", "Lena", "Mateo", "Nia", "Omar", "Priya"]
ITEMS = [("apple", "apples"), ("pencil", "pencils"), ("marble", "marbles"), ("sticker", "stickers"), ("book", "books"),
         ("cookie", "cookies"), ("ticket", "tickets"), ("card", "cards")]


def _n(rng, level, lo=1):
    """A number whose size grows with the level."""
    return rng.randint(lo, 10 ** (level + 0) + 5 * level)


def _fmt(fr):
    fr = Fraction(fr)
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


# ------------------------------------------------------------------------------------------------ generators
def g_addsub(rng, lv):
    a, b = _n(rng, lv + 1), _n(rng, lv + 1)
    if rng.random() < 0.5:
        return f"What is {a} + {b}?", a + b
    a, b = max(a, b), min(a, b)
    return f"What is {a} - {b}?", a - b


def g_mul(rng, lv):
    a, b = rng.randint(2, 9 + 8 * lv), rng.randint(2, 9 + 6 * lv)
    return f"Compute {a} × {b}.", a * b


def g_div(rng, lv):
    b, q = rng.randint(2, 9 + 4 * lv), rng.randint(2, 12 + 10 * lv)
    return f"What is {b * q} ÷ {b}?", q


def g_order(rng, lv):
    a, b, c = rng.randint(2, 9 * lv + 5), rng.randint(2, 9 * lv + 5), rng.randint(2, 9 * lv + 5)
    forms = [(f"{a} + {b} × {c}", a + b * c), (f"({a} + {b}) × {c}", (a + b) * c), (f"{a} × {b} - {c}", a * b - c)]
    if lv >= 3:
        d = rng.randint(2, 9)
        forms += [(f"({a} + {b}) × ({c} - {d})", (a + b) * (c - d)), (f"{a} × {b} + {c} × {d}", a * b + c * d)]
    e, v = rng.choice(forms)
    return f"Evaluate {e}.", v


def g_fraction(rng, lv):
    d1, d2 = rng.randint(2, 4 + 3 * lv), rng.randint(2, 4 + 3 * lv)
    f1, f2 = Fraction(rng.randint(1, d1), d1), Fraction(rng.randint(1, d2), d2)
    op = rng.choice(["+", "-", "×"] if lv >= 2 else ["+"])
    val = f1 + f2 if op == "+" else f1 - f2 if op == "-" else f1 * f2
    return (f"Compute {f1.numerator}/{f1.denominator} {op} {f2.numerator}/{f2.denominator}. "
            "Give the answer as a fraction in lowest terms (or an integer)."), val


def g_percent(rng, lv):
    p = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75] if lv < 3 else list(range(2, 96, 3)))
    base = rng.choice([40, 60, 80, 120, 200, 250, 400, 500, 800] if lv < 3 else [rng.randint(20, 900) * 4])
    if rng.random() < 0.6:
        return f"What is {p}% of {base}?", Fraction(p * base, 100)
    price = base
    return f"A jacket costs ${price}. It is on sale for {p}% off. What is the sale price in dollars?", Fraction(price * (100 - p), 100)


def g_linear(rng, lv):
    x = rng.randint(-5 - 3 * lv, 8 + 5 * lv)
    a = rng.choice([v for v in range(-3 - lv, 4 + 2 * lv) if v not in (0, 1)] or [2])
    b = rng.randint(-10 * lv, 10 * lv)
    return f"Solve for x: {a}x {'+' if b >= 0 else '-'} {abs(b)} = {a * x + b}.", x


def g_twostep_system(rng, lv):
    x, y = rng.randint(1, 4 + 2 * lv), rng.randint(1, 4 + 2 * lv)
    return (f"Two numbers x and y satisfy x + y = {x + y} and x - y = {x - y}. What is x × y?"), x * y


def g_gcdlcm(rng, lv):
    g = rng.randint(2, 6 + 2 * lv)
    a, b = g * rng.randint(2, 6 + 3 * lv), g * rng.randint(2, 6 + 3 * lv)
    if rng.random() < 0.5:
        return f"What is the greatest common divisor of {a} and {b}?", math.gcd(a, b)
    return f"What is the least common multiple of {a} and {b}?", a * b // math.gcd(a, b)


def g_mod_pow(rng, lv):
    if rng.random() < 0.5:
        a, m = rng.randint(10 ** lv, 10 ** (lv + 1) * 5), rng.randint(3, 6 + 4 * lv)
        return f"What is the remainder when {a} is divided by {m}?", a % m
    b, e = rng.randint(2, 5 + lv), rng.randint(2, 3 + lv)
    return f"Compute {b}^{e}.", b ** e


def g_sequence(rng, lv):
    a0 = rng.randint(1, 10 * lv)
    if rng.random() < 0.6 or lv < 3:
        d = rng.randint(2, 4 + 2 * lv)
        seq = [a0 + d * i for i in range(5)]
        return f"What is the next number in the sequence {', '.join(map(str, seq))}, ...?", a0 + 5 * d
    r = rng.randint(2, 4)
    seq = [a0 * r ** i for i in range(4)]
    return f"What is the next number in the sequence {', '.join(map(str, seq))}, ...?", a0 * r ** 4


def g_stats(rng, lv):
    k = rng.randint(3, 4 + lv)
    n = rng.randint(2, 6)
    xs = [rng.randint(1, 10 * lv + 9) for _ in range(k * n)][:k]
    # force an integer mean
    xs[-1] += (-sum(xs)) % k
    kind = rng.choice(["mean", "median", "range"])
    s = ", ".join(map(str, xs))
    if kind == "mean":
        return f"What is the mean of {s}?", Fraction(sum(xs), k)
    if kind == "range":
        return f"What is the range of {s}?", max(xs) - min(xs)
    ys = sorted(xs)
    med = Fraction(ys[k // 2]) if k % 2 else Fraction(ys[k // 2 - 1] + ys[k // 2], 2)
    return f"What is the median of {s}?", med


def g_counting(rng, lv):
    n = rng.randint(4, 6 + 2 * lv)
    r = rng.randint(2, min(4, n - 1))
    if rng.random() < 0.5:
        return f"In how many ways can you choose {r} items from {n} different items (order does not matter)?", math.comb(n, r)
    return f"How many different ways can {r} of {n} people stand in a line (order matters)?", math.perm(n, r)


def g_probability(rng, lv):
    red, blue = rng.randint(2, 4 + 2 * lv), rng.randint(2, 4 + 2 * lv)
    return (f"A bag has {red} red and {blue} blue marbles. One marble is drawn at random. "
            "What is the probability it is red? Give a fraction in lowest terms."), Fraction(red, red + blue)


def g_geometry(rng, lv):
    a, b = rng.randint(2, 6 + 4 * lv), rng.randint(2, 6 + 4 * lv)
    kind = rng.choice(["rect_area", "rect_perim", "tri_area"] + (["circle"] if lv >= 4 else []))
    if kind == "rect_area":
        return f"A rectangle is {a} cm by {b} cm. What is its area in square centimetres?", a * b
    if kind == "rect_perim":
        return f"A rectangle is {a} cm by {b} cm. What is its perimeter in centimetres?", 2 * (a + b)
    if kind == "tri_area":
        a += a % 2
        return f"A triangle has base {a} cm and height {b} cm. What is its area in square centimetres?", Fraction(a * b, 2)
    return f"A circle has radius {a} cm. What is its area divided by π (that is, give the number multiplying π)?", a * a


def g_word(rng, lv):
    name, (s1, p1) = rng.choice(NAMES), rng.choice(ITEMS)
    name2 = rng.choice([n for n in NAMES if n != name])
    a, b, c = rng.randint(3, 12 * lv + 8), rng.randint(2, 9 * lv + 5), rng.randint(2, 6 + 2 * lv)
    forms = [
        (f"{name} has {a} {p1}. {name2} gives {name} {b} more, then {name} gives away {min(c, a + b - 1)}. How many {p1} does {name} have now?", a + b - min(c, a + b - 1)),
        (f"{name} buys {c} packs of {p1}. Each pack has {b} {p1}. {name} already had {a}. How many {p1} does {name} have in total?", c * b + a),
        (f"A train travels at {b * 10} km/h for {c} hours. How many kilometres does it travel?", b * 10 * c),
        (f"{name} earns ${b} per hour and works {c} hours a day for {rng.randint(2, 6)} days. How many dollars does {name} earn in total?", None),
    ]
    q, ans = forms[rng.randrange(len(forms))]
    if ans is None:
        days = int(re.search(r"for (\d+) days", q).group(1))
        ans = b * c * days
    return q, ans


def g_multistep_word(rng, lv):
    name = rng.choice(NAMES)
    price, qty, disc, cash = rng.randint(3, 9 + 3 * lv), rng.randint(3, 6 + 2 * lv), rng.randint(1, 4 + lv), None
    total = price * qty - disc
    cash = total + rng.randint(1, 20)
    return (f"{name} buys {qty} notebooks at ${price} each and uses a ${disc} coupon. {name} pays with ${cash}. "
            "How much change does the cashier give back, in dollars?"), cash - total


SKILLS = {
    "addition & subtraction": (g_addsub, 1), "multiplication": (g_mul, 1), "division": (g_div, 1),
    "order of operations": (g_order, 1), "fractions": (g_fraction, 2), "percentages": (g_percent, 2),
    "linear equations": (g_linear, 2), "systems": (g_twostep_system, 2), "gcd & lcm": (g_gcdlcm, 2),
    "remainders & powers": (g_mod_pow, 1), "sequences": (g_sequence, 1), "statistics": (g_stats, 2),
    "counting": (g_counting, 3), "probability": (g_probability, 2), "geometry": (g_geometry, 1),
    "word problems": (g_word, 1), "multi-step word problems": (g_multistep_word, 2),
}


def problem(skill, level, rng):
    fn, _ = SKILLS[skill]
    q, a = fn(rng, max(1, min(5, level)))
    return {"question": q, "answer": _fmt(a), "skill": skill, "level": level}


def make_set(n, levels=(1, 2, 3), skills=None, weights=None, seed=0):
    """n unique problems. `weights` ({skill: weight}) focuses practice on weak skills."""
    rng = random.Random(seed)
    skills = list(skills or SKILLS)
    w = [max(0.02, (weights or {}).get(s, 1.0)) for s in skills]
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 30:
        tries += 1
        s = rng.choices(skills, w)[0]
        p = problem(s, rng.choice(list(levels)), rng)
        if p["question"] in seen:
            continue
        seen.add(p["question"])
        out.append(p)
    return out


def prompt_text(p):
    return f"{p['question']}\n{INSTRUCTION}"


# ------------------------------------------------------------------------------------------------ checking answers
_FRAC_LATEX = re.compile(r"\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}")


def _last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7:j - 1] if depth == 0 else None


def extract_answer(text):
    """Pull the final answer out of a model's reply (None if there isn't one)."""
    if not text:
        return None
    b = _last_boxed(text)
    if b is not None:
        return b.strip()
    m = re.findall(r"####\s*([^\n]+)", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|the answer is|answer:)\s*[:=]?\s*\$?([^\n]+)", text, re.I)
    if m:
        return m[-1].strip(" .$")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?", text)
    return nums[-1] if nums else None


def to_fraction(s):
    if s is None:
        return None
    s = str(s).strip().replace("$", "").replace(",", "").replace("\\!", "").replace("\\,", "").replace("%", "")
    s = re.sub(r"\\text\{[^}]*\}|\\mbox\{[^}]*\}", "", s).strip(" .")
    s = _FRAC_LATEX.sub(lambda m: f"{m.group(1)}/{m.group(2)}", s)
    s = re.sub(r"(?<=\d)\s*(cm|km|m|kg|g|hours?|hrs?|dollars?|units?|ways?|square.*)$", "", s).strip()
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    try:
        if m:
            return Fraction(int(m.group(1)), int(m.group(2)))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
        try:
            return Fraction(m.group(0)) if m else None
        except (ValueError, ZeroDivisionError):
            return None


def is_correct(text, gold):
    pred = to_fraction(extract_answer(text))
    g = to_fraction(gold)
    if pred is None or g is None:
        return False
    return pred == g or abs(float(pred) - float(g)) <= 1e-6 * max(1.0, abs(float(g)))


def has_boxed(text):
    return _last_boxed(text or "") is not None
