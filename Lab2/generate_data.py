"""Создание воспроизводимого учебного набора для варианта 14."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(10031)
coordinates = [(0, 3), (0, 2), (0, 1), (0, 0), (4, 3), (4, 2), (4, 1), (4, 0)]
balances = [35, 30, 25, 30, -25, -35, -30, -30]
arcs = []
for i in range(4):
    for j in range(4, 8):
        cost = int(round(8 + 4 * abs(coordinates[i][1] - coordinates[j][1]) + rng.integers(0, 5)))
        arcs.append({'from': i + 1, 'to': j + 1, 'capacity': 12 if (i, j) == (0, 4) else 40, 'cost': cost})
data = {'variant': 14, 'seed': 10031, 'units': 'автомобили за период',
        'nodes': [{'id': i + 1, 'balance': balances[i], 'x': xy[0], 'y': xy[1]} for i, xy in enumerate(coordinates)],
        'arcs': arcs}
path = Path(__file__).resolve().parent / 'data' / 'taxi.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
print(path)
