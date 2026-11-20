"""Постановка, решение, независимая проверка и жадная эвристика."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

TOL = 1e-7
STATUSES = {0: 'optimal', 1: 'time_or_iteration_limit', 2: 'infeasible',
            3: 'unbounded', 4: 'solver_error'}


@dataclass
class Problem:
    data: dict
    scenario: dict

    def __post_init__(self):
        self.people = self.data['candidates']
        self.teams = self.data['teams']
        self.n, self.k = len(self.people), len(self.teams)
        self.roles = self.data['roles']
        self.scores = np.asarray(self.data['scores'])
        self.hours = np.array([person['hours'] for person in self.people])
        self.required = np.array([team['required_hours'] + self.scenario.get('extra_required_hours', 0)
                                  for team in self.teams])
        self.caps = np.array([team['reserve_cap'] for team in self.teams])
        self.fixed = np.array([team['reserve_fixed'] for team in self.teams])
        self.hourly = np.array([team['reserve_hourly'] for team in self.teams])
        self.unavailable = set(self.scenario.get('unavailable', []))
        self.available = np.array([p['id'] not in self.unavailable for p in self.people])
        self.conflicts = self.data['conflicts'] + self.scenario.get('extra_conflicts', [])

    def triples(self):
        """Допустимые по ролям и попарной совместимости составы."""
        forbidden = {frozenset(pair) for pair in self.conflicts}
        per_role = [[i for i, person in enumerate(self.people)
                     if self.available[i] and person['role'] == role] for role in self.roles]
        for triple in product(*per_role):
            ids = [self.people[i]['id'] for i in triple]
            if all(frozenset((ids[a], ids[b])) not in forbidden
                   for a, b in ((0, 1), (0, 2), (1, 2))):
                yield triple

    def pack(self, x, y, h):
        return np.r_[x.ravel(), y, h]

    def unpack(self, vector):
        count = self.n * self.k
        return vector[:count].reshape(self.n, self.k), vector[count:count+self.k], vector[count+self.k:]


def build_model(problem, m_factor=1, tightened=False):
    """Порядок переменных: назначения x, активации y, часы h."""
    p = problem
    x = np.arange(p.n * p.k).reshape(p.n, p.k)
    y = p.n * p.k + np.arange(p.k)
    h = y + p.k
    size = p.n * p.k + 2*p.k
    upper = p.pack(np.broadcast_to(p.available[:, None], x.shape), np.ones(p.k), p.caps)
    cost = p.pack(-p.scores, p.fixed, p.hourly)
    equal, eq_rhs, inequalities, ub_rhs = [], [], [], []

    def add(indices, values, rhs, equality=False):
        row = np.zeros(size)
        row[indices] = values
        if equality:
            equal.append(row)
            eq_rhs.append(rhs)
        else:
            inequalities.append(row)
            ub_rhs.append(rhs)

    for person_assignments in x:
        add(person_assignments, 1, 1)
    indices = {person['id']: i for i, person in enumerate(p.people)}
    for j in range(p.k):
        for role in p.roles:
            members = [i for i, person in enumerate(p.people) if person['role'] == role]
            add(x[members, j], 1, 1, equality=True)
        for left, right in p.conflicts:
            add(x[[indices[left], indices[right]], j], 1, 1)
        add(np.r_[x[:, j], h[j]], np.r_[-p.hours, -1], -p.required[j])
        add([h[j], y[j]], [1, -m_factor*p.caps[j]], 0)
        if tightened:
            for triple in p.triples():
                members = list(triple)
                if p.hours[members].sum() < p.required[j]:
                    add(np.r_[x[members, j], y[j]], [1, 1, 1, -1], 2)
    return cost, upper, np.asarray(equal), np.asarray(eq_rhs), np.asarray(inequalities), np.asarray(ub_rhs)


def solve(problem, *, integer, m_factor=1, tightened=False, display=False):
    cost, upper, equal, eq_rhs, inequalities, ub_rhs = build_model(problem, m_factor, tightened)
    if integer:
        # Абсолютные и относительные допуски ветвления применяет HiGHS.
        integrality = np.r_[np.ones(problem.n*problem.k + problem.k), np.zeros(problem.k)]
        return milp(cost, integrality=integrality, bounds=Bounds(0, upper),
                    constraints=(LinearConstraint(equal, eq_rhs, eq_rhs),
                                 LinearConstraint(inequalities, -np.inf, ub_rhs)),
                    options={'presolve': True, 'disp': display, 'time_limit': 15.0, 'mip_rel_gap': 0.0})
    return linprog(cost, A_ub=inequalities, b_ub=ub_rhs, A_eq=equal, b_eq=eq_rhs,
                   bounds=list(zip(np.zeros_like(upper), upper)), method='highs-ds',
                   options={'presolve': True, 'disp': display, 'time_limit': 15.0,
                            'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})


def verify(problem, vector, *, integer, m_factor=1, tightened=False):
    """Пересчёт ограничений и цели по данным, без матрицы и результата HiGHS."""
    p = problem
    x, y, h = p.unpack(vector)
    rows = []

    def record(rule, where, violation):
        rows.append({'constraint': rule, 'index': where, 'violation': max(0.0, float(violation))})

    for i, person in enumerate(p.people):
        name = person['id']
        record('single_team', name, x[i].sum() - 1)
        for j, team in enumerate(p.teams):
            key = f"{name}/{team['id']}"
            record('assignment_lower_bound', key, -x[i, j])
            record('availability_upper_bound', key, x[i, j] - (name not in p.unavailable))
            if integer:
                record('assignment_integrality', key, abs(x[i, j] - round(x[i, j])))
    total = 0.0
    for j, team in enumerate(p.teams):
        key = team['id']
        for role in p.roles:
            count = sum(x[i, j] for i, person in enumerate(p.people) if person['role'] == role)
            record('one_person_per_role', f'{key}/{role}', abs(count - 1))
        assignments = {person['id']: x[i, j] for i, person in enumerate(p.people)}
        for a, b in p.conflicts:
            record('compatibility', f'{key}/{a}/{b}', assignments[a] + assignments[b] - 1)
        supplied = sum(person['hours'] * x[i, j] for i, person in enumerate(p.people))
        required = team['required_hours'] + p.scenario.get('extra_required_hours', 0)
        record('hours_requirement', key, required - supplied - h[j])
        record('reserve_lower_bound', key, -h[j])
        record('reserve_upper_bound', key, h[j] - team['reserve_cap'])
        record('activation_bounds', key, max(-y[j], y[j] - 1))
        record('big_m_link', key, h[j] - m_factor * team['reserve_cap'] * y[j])
        if integer:
            record('activation_integrality', key, abs(y[j] - round(y[j])))
        if tightened:
            for triple in p.triples():
                if p.hours[list(triple)].sum() < required:
                    group = '/'.join(p.people[i]['id'] for i in triple)
                    record('shortfall_cut', f'{key}/{group}', sum(x[i, j] for i in triple) - 2 - y[j])
        total += sum(p.data['scores'][i][j] * x[i, j] for i in range(p.n))
        total -= team['reserve_fixed'] * y[j] + team['reserve_hourly'] * h[j]
    return float(total), rows


def greedy(p):
    """Берём лучший допустимый состав для каждой команды по порядку, без возврата."""
    triples = list(p.triples())
    used = set()
    x, y, h = np.zeros((p.n, p.k)), np.zeros(p.k), np.zeros(p.k)
    for j in range(p.k):
        best = None
        best_score = -np.inf
        for triple in triples:
            if used.intersection(triple):
                continue
            members = list(triple)
            extra = max(0, p.required[j] - p.hours[members].sum())
            if extra > p.caps[j]:
                continue
            score = p.scores[members, j].sum() - p.fixed[j]*(extra > 0) - p.hourly[j]*extra
            if score > best_score:
                best, best_score = (members, extra), score
        if best is None:
            return None
        members, hours = best
        used.update(members)
        x[members, j], y[j], h[j] = 1, hours > 0, hours
    return p.pack(x, y, h)
