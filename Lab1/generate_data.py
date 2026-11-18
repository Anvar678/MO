"""Воспроизведение учебных исходных данных; явно запускать только для их обновления."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(10031)
ash = np.round(np.array([6.7, 9.3, 12.5, 16.0, 20.2]) + rng.uniform(-0.5, 0.5, 5), 2)
sulfur = np.round(np.array([0.75, 1.65, 1.05, 0.52, 2.10]) + rng.uniform(-0.08, 0.08, 5), 3)
prices = np.array([5300, 4400, 4240, 3770, 3080]) + rng.integers(-90, 91, 5)
capacities = np.array([250, 300, 300, 390, 400]) + rng.integers(-20, 21, 5)
data = {
    'variant': 14, 'seed': 10031,
    'source': 'Учебная синтетическая генерация; диапазоны выбраны для данной ЛР',
    'demand_t': 1000.0, 'ash_max_pct': 12.3, 'sulfur_max_pct': 1.25,
    'suppliers': [
        {'name': f'Поставщик {i + 1}', 'price_rub_t': int(prices[i]),
         'ash_pct': float(ash[i]), 'sulfur_pct': float(sulfur[i]),
         'capacity_t': int(capacities[i])} for i in range(5)
    ]
}
path = Path(__file__).resolve().parent / 'data' / 'coal_data.json'
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Данные записаны: {path}')
