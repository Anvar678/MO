"""Синтетические данные для варианта 14 и эксперимента с масштабом."""

import json
from pathlib import Path

import numpy as np

SEED = 10031
SCALES = (12, 24, 48)
ROLES = ('Разработчик', 'Аналитик', 'Координатор')
DATA_DIR = Path(__file__).resolve().parent / 'data'


def generate(size):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, size]))
    team_count = size // 4
    candidates = [
        {'id': f'P{i + 1:02}', 'role': ROLES[i % 3],
         'hours': int(rng.integers(5, 11))}
        for i in range(size)
    ]
    teams = [
        {'id': f'T{j + 1:02}', 'required_hours': 23 + j % 3,
         'reserve_cap': 10, 'reserve_fixed': 8 + j % 3, 'reserve_hourly': 2}
        for j in range(team_count)
    ]
    scores = rng.integers(42, 86, (size, team_count))
    for j in range(team_count):
        scores[3*j:3*j + 3, j] += 8
    conflicts = [
        [candidates[i]['id'], candidates[k]['id']]
        for i in range(size) for k in range(i + 1, size)
        if i % 3 != k % 3 and (i // 3 != k // 3 or i >= team_count * 3)
        and rng.random() < 0.17
    ]
    result = {
        'variant': 14, 'seed': SEED, 'roles': ROLES,
        'candidates': candidates, 'teams': teams, 'scores': scores.tolist(),
        'conflicts': conflicts,
    }
    if size == 12:
        result['scenarios'] = [
            {'id': 'base', 'name': 'Базовый', 'unavailable': [], 'extra_conflicts': [],
             'extra_required_hours': 0},
            {'id': 'deficit', 'name': 'Дефицитный', 'unavailable': ['P01'],
             'extra_conflicts': [], 'extra_required_hours': 0},
            {'id': 'stress', 'name': 'Стрессовый', 'unavailable': ['P01', 'P06'],
             'extra_conflicts': [['P02', 'P04']], 'extra_required_hours': 1},
        ]
    return result


if __name__ == '__main__':
    DATA_DIR.mkdir(exist_ok=True)
    for size in SCALES:
        path = DATA_DIR / f'teams_{size}.json'
        path.write_text(json.dumps(generate(size), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(path.name)
