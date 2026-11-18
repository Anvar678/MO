"""Балансировка такси: ЛП, сценарии и независимая проверка."""
from pathlib import Path
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
data = json.loads((ROOT / 'data' / 'taxi.json').read_text(encoding='utf-8'))
nodes, arcs = data['nodes'], data['arcs']
n, m = len(nodes), len(arcs)
balance = np.array([v['balance'] for v in nodes], dtype=float)
cost = np.array([a['cost'] for a in arcs], dtype=float)
capacity = np.array([a['capacity'] for a in arcs], dtype=float)
A = np.zeros((n, m))
for k, arc in enumerate(arcs):
    A[arc['from'] - 1, k] = 1
    A[arc['to'] - 1, k] = -1

def solve(b, caps):
    return linprog(cost, A_eq=A, b_eq=b, bounds=list(zip(np.zeros(m), caps)),
                   method='highs-ds', options={'presolve': True,
                   'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})

def verify(x, b, caps):
    # Проверка строит потоки непосредственно по списку дуг, не используя матрицу модели.
    net = np.zeros(n)
    for amount, arc in zip(x, arcs):
        net[arc['from'] - 1] += amount
        net[arc['to'] - 1] -= amount
    residual = net - b
    violation = max(float(np.max(np.abs(residual))), float(np.max(np.maximum(-x, 0))),
                    float(np.max(np.maximum(x - caps, 0))))
    return net, residual, violation, sum(float(x[k]) * arc['cost'] for k, arc in enumerate(arcs))

def greedy(b, caps):
    supply, demand = np.maximum(b, 0).copy(), np.maximum(-b, 0).copy()
    x = np.zeros(m)
    for k in sorted(range(m), key=lambda k: (cost[k], k)):
        i, j = arcs[k]['from'] - 1, arcs[k]['to'] - 1
        x[k] = min(supply[i], demand[j], caps[k])
        supply[i] -= x[k]
        demand[j] -= x[k]
    return x

base = solve(balance, capacity)
if not base.success:
    raise RuntimeError(f'Базовая модель не решена: {base.message}')
net, residual, violation, independent_cost = verify(base.x, balance, capacity)
assert violation <= 1e-7 and abs(independent_cost - base.fun) <= 1e-7
closed = int(np.argmax(base.x))
closed_caps = capacity.copy()
closed_caps[closed] = 0
reduced_caps = capacity.copy()
reduced_caps[closed] *= 0.5
growth_balance = balance.copy()
growth_balance[growth_balance < 0] *= 1.2
scenarios = [('Базовый', balance, capacity, base),
             ('Перегрузка: ёмкость дуги снижена на 50%', balance, reduced_caps, solve(balance, reduced_caps)),
             ('Отказ самой загруженной дуги', balance, closed_caps, solve(balance, closed_caps)),
             ('Рост спроса на 20%', growth_balance, capacity, solve(growth_balance, capacity))]
status = {0: 'optimal', 1: 'time_limit / iteration_limit', 2: 'infeasible', 3: 'unbounded', 4: 'solver_error'}
rows, checks, log = [], [], [f'Python {platform.python_version()}, SciPy {scipy.__version__}',
                            'HiGHS через scipy.optimize.linprog: highs-ds, presolve=True; протокол интерфейса SciPy.']
for name, b, caps, result in scenarios:
    rows.append({'Сценарий': name, 'Статус': status.get(result.status, str(result.status)),
                 'Стоимость': result.fun if result.success else np.nan,
                 'Изменение стоимости': result.fun - base.fun if result.success else np.nan,
                 'Итерации': result.nit})
    log.extend([f'\n{name}', f'Статус: {status.get(result.status, result.status)}',
                f'Сообщение: {result.message}', f'Итерации: {result.nit}',
                f'Цель: {result.fun if result.success else "нет"}'])
    if result.success:
        _, _, err, value = verify(result.x, b, caps)
        assert err <= 1e-7 and abs(value - result.fun) <= 1e-7
        checks.append({'Сценарий': name, 'Максимальное нарушение': err,
                       'Независимая стоимость': value, 'Расхождение стоимости': abs(value - result.fun)})
        plan = pd.DataFrame(arcs)
        plan['flow'] = result.x
        plan['scenario'] = name
        plan.to_csv(OUT / ('plan_' + str(len(checks)) + '.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(rows).to_csv(OUT / 'scenarios.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(checks).to_csv(OUT / 'verification_scenarios.csv', index=False, encoding='utf-8-sig')
(OUT / 'solver.log').write_text('\n'.join(log) + '\n', encoding='utf-8')
pd.DataFrame({'Зона': range(1, n+1), 'Баланс': balance, 'Фактический баланс': net,
              'Невязка': residual}).to_csv(OUT / 'verification_nodes.csv', index=False, encoding='utf-8-sig')
flows = pd.DataFrame(arcs)
flows['flow'] = base.x
flows['slack'] = capacity - base.x
flows['shadow_price'] = base.upper.marginals
flows['lower_violation'] = np.maximum(-base.x, 0)
flows['upper_violation'] = np.maximum(base.x-capacity, 0)
flows.to_csv(OUT / 'solution.csv', index=False, encoding='utf-8-sig')

started = time.perf_counter()
gx = greedy(balance, capacity)
greedy_time = time.perf_counter() - started
_, _, greedy_violation, greedy_cost = verify(gx, balance, capacity)
assert greedy_violation <= 1e-7
gap = (greedy_cost / base.fun - 1) * 100
pd.DataFrame([{'Метод': 'Жадная по стоимости', 'Стоимость': greedy_cost,
               'Зазор, %': gap, 'Нарушение': greedy_violation, 'Время, с': greedy_time}]).to_csv(
                   OUT / 'greedy.csv', index=False, encoding='utf-8-sig')

active = np.where(capacity - base.x < 1e-7)[0]
shadow_rows = []
for k in active:
    changed = capacity.copy()
    changed[k] += 1
    r = solve(balance, changed)
    shadow_rows.append({'Дуга': f"{arcs[k]['from']}→{arcs[k]['to']}", 'Маржинальная цена': base.upper.marginals[k],
                        'Прогноз изменения при +1': base.upper.marginals[k],
                        'Фактическое изменение': r.fun - base.fun if r.success else np.nan})
pd.DataFrame(shadow_rows, columns=['Дуга', 'Маржинальная цена', 'Прогноз изменения при +1', 'Фактическое изменение']).to_csv(
    OUT / 'shadow_prices.csv', index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(13, 9))
for k, arc in enumerate(arcs):
    a, b = nodes[arc['from']-1], nodes[arc['to']-1]
    ax.annotate('', xy=(b['x']-.15, b['y']), xytext=(a['x']+.15, a['y']),
                arrowprops={'arrowstyle': '->', 'color': '#286aa0' if base.x[k] > 1e-7 else '#bbbbbb',
                            'alpha': .8, 'lw': 1.8 if base.x[k] > 1e-7 else .7})
    # Разнесение подписей вдоль дуг уменьшает наложение.
    t = .30 + .12 * (arc['from']-1)
    ax.text((1-t)*a['x']+t*b['x'], (1-t)*a['y']+t*b['y'],
            f"{base.x[k]:.0f}/{capacity[k]:.0f}", fontsize=7,
            bbox={'facecolor':'white', 'alpha':.75, 'edgecolor':'none', 'pad':.3})
for v in nodes:
    ax.scatter(v['x'], v['y'], s=650, color='#82c99d' if v['balance'] > 0 else '#ffc078', zorder=3)
    ax.text(v['x'], v['y'], str(v['id']), ha='center', va='center', weight='bold', zorder=4)
    ax.text(v['x'], v['y']+.23, f"b={v['balance']:+}", ha='center', fontsize=9)
ax.set(xlim=(-.8, 4.8), ylim=(-.5, 3.6), title='Потоки такси: подписи поток / пропускная способность')
ax.axis('off')
fig.savefig(OUT / 'network.png', dpi=180, bbox_inches='tight')
plt.close(fig)

closure = scenarios[2][3]
closure_text = (f'стоимость {closure.fun:.2f}, рост {closure.fun-base.fun:.2f} руб.' if closure.success
                else 'решение отсутствует: при закрытии дуги сеть не может удовлетворить все балансы')
active_text = ', '.join(f"{arcs[k]['from']}→{arcs[k]['to']} (оценка {base.upper.marginals[k]:.2f} руб./авто)" for k in active) or 'нет'
zero_arcs = sum(base.x < 1e-7)
integer = milp(cost, integrality=np.ones(m), bounds=Bounds(np.zeros(m), capacity),
               constraints=LinearConstraint(A, balance, balance), options={'presolve': True})
if not integer.success:
    raise RuntimeError(f'Целочисленная модель не решена: {integer.message}')
_, _, integer_violation, integer_cost = verify(integer.x, balance, capacity)
integrality_violation = float(np.max(np.abs(integer.x - np.round(integer.x))))
assert integer_violation <= 1e-7 and integrality_violation <= 1e-7
pd.DataFrame([{'Статус': integer.message, 'Стоимость': integer_cost,
               'Разница с LP': integer_cost-base.fun, 'Невязка': integer_violation,
               'Невязка целочисленности': integrality_violation}]).to_csv(
                   OUT / 'integer_comparison.csv', index=False, encoding='utf-8-sig')
report = f'''# Лабораторная работа № 2. Вариант 14: балансировка такси

## Постановка и данные
За один период требуется перебросить свободные автомобили из четырёх зон избытка в четыре зоны дефицита. Вершины сети — восемь зон, дуги — допустимые перемещения из зоны избытка в зону дефицита. Исходные данные синтетические, seed 10031; `data/taxi.json` содержит балансы, координаты, стоимости и пропускные способности всех 16 дуг. `generate_data.py` воспроизводит файл. Балансы: {balance.tolist()}; положительное значение — избыток, отрицательное — потребность. Общий избыток и спрос равны {sum(balance[balance>0]):.0f} автомобилей.

## Формальная модель
Множество V — зоны 1–8, E — дуги из JSON. xᵢⱼ — непрерывное число автомобилей за период; cᵢⱼ — стоимость перемещения, руб./автомобиль; uᵢⱼ — пропускная способность, автомобилей за период; bᵢ — избыток (положительный) или потребность (отрицательная), автомобилей. Модель минимизирует Σ₍ᵢ,ⱼ₎ cᵢⱼxᵢⱼ при Σⱼ xᵢⱼ − Σⱼ xⱼᵢ = bᵢ для каждой зоны и 0 ≤ xᵢⱼ ≤ uᵢⱼ для каждой дуги. Цель измеряется в рублях за период. Непрерывность — учебное приближение, хотя машины неделимы.

## Решение и проверка
Использован открытый HiGHS через SciPy linprog, highs-ds (двойственный симплекс), presolve включён, допуски солвера 1e-9. Статус optimal означает найденный оптимум с численными допусками; infeasible — система ограничений несовместна, unbounded — цель не ограничена, а достижение лимита или ошибка не подтверждают оптимальность. Базовый статус: {status[base.status]}, стоимость {base.fun:.2f} руб., итераций {base.nit}. Таблица потоков и границ: `solution.csv`; схема `network.png` показывает поток/ёмкость на каждой дуге. Затраты указаны в исходном JSON и таблице решения.

Независимый код обходит исходный список дуг, не используя матрицу модели или объект ограничений солвера. Он проверяет баланс в каждой вершине и обе границы каждой дуги, отдельно пересчитывает стоимость. Максимальное нарушение {violation:.3g} при допуске 1e-7, независимая стоимость {independent_cost:.2f} руб. Подробности — `verification_nodes.csv`, `solution.csv`, `verification_scenarios.csv`.

## Сценарии
```
{pd.DataFrame(rows).to_string(index=False)}
```
Самая загруженная дуга {arcs[closed]['from']}→{arcs[closed]['to']} несла {base.x[closed]:.0f} автомобилей; при полном отказе {closure_text}. При росте спроса на 20% нужно {sum(-growth_balance[growth_balance<0]):.0f} автомобилей при наличии {sum(growth_balance[growth_balance>0]):.0f}; дефицит {abs(growth_balance.sum()):.0f} делает баланс несовместным независимо от пропускных способностей. Это не оценка стоимости неудовлетворённого спроса: модель не допускает недоставки.

Перегрузка смоделирована уменьшением пропускной способности той же дуги на 50%; результаты и планы допустимых сценариев приведены в CSV. Это имитация снижения доступной пропускной способности, а не модель очереди.

## Узкие места и теневые цены
Насыщенные дуги: {active_text}. Маржинальная цена — изменение минимальной стоимости в рублях при увеличении ёмкости на один автомобиль локально; отрицательное значение показывает экономию. Насыщенная дуга с нулевой оценкой не обязательно ограничивает стоимость. Повторные решения при увеличении ёмкости на единицу и сравнение с прогнозом сохранены в `shadow_prices.csv`; при смене базиса прогноз может отличаться. Число дуг с нулевым потоком: {zero_arcs}; более дорогие или ненужные перемещения исключаются оптимизатором.

## Сравнение с жадной эвристикой
Эвристика рассматривает дуги по возрастанию стоимости и сразу отправляет максимально возможный поток; результат отдельно проверен. Стоимость {greedy_cost:.2f} руб., относительный зазор к оптимуму {gap:.2f}%, максимальная невязка {greedy_violation:.3g}; результат в `greedy.csv`. Жадный выбор не гарантирует оптимума, поскольку раннее распределение меняет доступные варианты для остальных зон.

## Дополнительное задание: целочисленная модель
Модель решена повторно через SciPy milp / HiGHS с требованием целочисленности каждого потока. Статус: {integer.message}; стоимость {integer_cost:.2f} руб., отличие от LP {integer_cost-base.fun:.2f} руб. Независимая проверка ограничений: {integer_violation:.3g}, целочисленности: {integrality_violation:.3g}. Сравнение сохранено в `integer_comparison.csv`. Совпадение целей относится к этим данным и не является результатом округления LP-решения.

## Допущения и практическая применимость
План применим как ориентир для короткого периода с известными избытком, спросом и постоянными тарифами. В реальности нужны целочисленные машины, прогнозный и меняющийся спрос, время движения, пробки, отказы водителей и ограничения по времени. Модель не учитывает промежуточные перемещения и неудовлетворённый спрос; оценить устойчивость к ним без расширения модели нельзя.
'''
(OUT / 'report.md').write_text(report, encoding='utf-8')
print(f'Базовый оптимум: {base.fun:.2f}; проверка: {violation:.3g}; результат: {OUT}')
