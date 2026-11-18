from pathlib import Path
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from IPython.display import display, Markdown

SEED = 10031
VARIANT = 14
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'lab1_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 110, 'savefig.dpi': 160, 'axes.grid': True, 'grid.alpha': 0.22})
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
print(f'Вариант {VARIANT}; seed {SEED}; Python {platform.python_version()}; SciPy {scipy.__version__}')

dataset = json.loads((ROOT / 'data' / 'coal_data.json').read_text(encoding='utf-8'))
names = [s['name'] for s in dataset['suppliers']]
price = np.array([s['price_rub_t'] for s in dataset['suppliers']])
ash_pct = np.array([s['ash_pct'] for s in dataset['suppliers']])
sulfur_pct = np.array([s['sulfur_pct'] for s in dataset['suppliers']])
capacity = np.array([s['capacity_t'] for s in dataset['suppliers']])
supplier_df = pd.DataFrame(dataset['suppliers']).rename(columns={
    'name': 'Поставщик', 'price_rub_t': 'Цена, ₽/т', 'ash_pct': 'Зольность, %',
    'sulfur_pct': 'Сера, %', 'capacity_t': 'Лимит, т'
})
display(supplier_df)
print(f"Потребность: {dataset['demand_t']:.0f} т; зольность ≤ {dataset['ash_max_pct']}%; сера ≤ {dataset['sulfur_max_pct']}%")
print(f"Совокупный доступный объём: {supplier_df['Лимит, т'].sum():.0f} т")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
idx = np.arange(1, 6)
ax[0].bar(idx, supplier_df['Цена, ₽/т'], color=plt.cm.viridis(np.linspace(0.2, 0.85, 5)))
ax[0].set(title='Цена угля по поставщикам', xlabel='Поставщик', ylabel='₽ за тонну', xticks=idx)
ax[0].set_ylim(0, max(price) * 1.18)
for i, value in enumerate(price):
    ax[0].text(i + 1, value + 55, f'{value:,.0f}', ha='center', fontsize=9)
ax[1].scatter(ash_pct, sulfur_pct, s=95 + 0.5 * capacity, c=price, cmap='viridis', edgecolors='black', linewidths=0.6)
for i in range(5):
    ax[1].annotate(str(i + 1), (ash_pct[i], sulfur_pct[i]), xytext=(6, 5), textcoords='offset points', fontsize=10)
ax[1].axvline(dataset['ash_max_pct'], ls='--', color='tomato', label='Максимум зольности смеси')
ax[1].axhline(dataset['sulfur_max_pct'], ls='--', color='royalblue', label='Максимум серы смеси')
ax[1].set(title='Качество отдельных углей', xlabel='Зольность, %', ylabel='Сера, %')
ax[1].legend(fontsize=8, loc='upper left')
fig.savefig(OUTPUT_DIR / '01_suppliers.png', bbox_inches='tight')
plt.show()

prices = np.array([s['price_rub_t'] for s in dataset['suppliers']], dtype=float)
ashes = np.array([s['ash_pct'] for s in dataset['suppliers']], dtype=float)
sulfurs = np.array([s['sulfur_pct'] for s in dataset['suppliers']], dtype=float)
limits = np.array([s['capacity_t'] for s in dataset['suppliers']], dtype=float)
D = float(dataset['demand_t'])
A_MAX = float(dataset['ash_max_pct'])
S_MAX = float(dataset['sulfur_max_pct'])


def solve_mix(caps=None, ash_limit=None, sulfur_limit=None):
    cap = np.asarray(limits if caps is None else caps, dtype=float)
    a_limit = A_MAX if ash_limit is None else float(ash_limit)
    s_limit = S_MAX if sulfur_limit is None else float(sulfur_limit)
    return linprog(
        c=prices,
        A_ub=np.vstack((ashes, sulfurs)),
        b_ub=np.array([a_limit * D, s_limit * D]),
        A_eq=np.ones((1, len(prices))),
        b_eq=np.array([D]),
        bounds=list(zip(np.zeros(len(cap)), cap)),
        method='highs-ds',
        options={'presolve': True, 'primal_feasibility_tolerance': 1e-9,
                 'dual_feasibility_tolerance': 1e-9}
    )


status_names = {0: 'optimal', 1: 'time_limit / iteration_limit', 2: 'infeasible',
                3: 'unbounded', 4: 'solver_error'}
base = solve_mix()
print('Статус:', status_names.get(base.status, str(base.status)))
print('Сообщение:', base.message)
print('Итераций:', base.nit)
if base.success:
    print(f'Минимальная стоимость: {base.fun:,.2f} ₽')

if not base.success:
    raise RuntimeError(f'Базовая модель не решена: {base.message}')

base_solution = pd.DataFrame({
    'Поставщик': names,
    'Закупка, т': base.x,
    'Доступно, т': limits,
    'Использование лимита, %': base.x / limits * 100,
    'Цена, ₽/т': prices,
    'Затраты, ₽': base.x * prices
})
base_solution.to_csv(OUTPUT_DIR / 'solution.csv', index=False, encoding='utf-8-sig')
display(base_solution)

ash_base = float(ashes @ base.x / D)
sulfur_base = float(sulfurs @ base.x / D)
print(f'Итоговая смесь: {D:.0f} т; стоимость {base.fun:,.2f} ₽; {base.fun / D:,.2f} ₽/т')
print(f'Средняя зольность: {ash_base:.6f}% (лимит {A_MAX:.2f}%)')
print(f'Средняя сера: {sulfur_base:.6f}% (лимит {S_MAX:.2f}%)')
print('Активные верхние границы:', ', '.join(names[i] for i in range(5) if abs(base.x[i] - limits[i]) < 1e-6) or 'нет')
print('Нулевые закупки:', ', '.join(names[i] for i in range(5) if base.x[i] < 1e-7) or 'нет')

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
ax[0].bar(names, base.x, color=colors)
ax[0].plot(names, limits, 'ko--', label='Лимит поставщика', markersize=5)
ax[0].set(title='Оптимальный объём закупки', ylabel='Тонны')
ax[0].tick_params(axis='x', rotation=25)
ax[0].legend()
ax[1].pie(base.x * prices, labels=names, autopct='%1.1f%%', startangle=95, colors=colors,
          wedgeprops={'linewidth': 0.7, 'edgecolor': 'white'})
ax[1].set_title('Доли поставщиков в стоимости закупки')
fig.savefig(OUTPUT_DIR / '02_optimal_mix.png', bbox_inches='tight')
plt.show()

def independent_check(x, source, caps=None, ash_limit=None, sulfur_limit=None):
    suppliers = source['suppliers']
    x = np.asarray(x, dtype=float)
    c = np.array([v['price_rub_t'] for v in suppliers], dtype=float)
    a = np.array([v['ash_pct'] for v in suppliers], dtype=float)
    s = np.array([v['sulfur_pct'] for v in suppliers], dtype=float)
    cap = np.array([v['capacity_t'] for v in suppliers], dtype=float) if caps is None else np.asarray(caps, dtype=float)
    mass = float(source['demand_t'])
    ash_lim = float(source['ash_max_pct']) if ash_limit is None else float(ash_limit)
    sulfur_lim = float(source['sulfur_max_pct']) if sulfur_limit is None else float(sulfur_limit)
    residuals = {
        'Баланс массы, т': abs(np.sum(x) - mass),
        'Превышение зольности, %·т': max(float(a @ x - ash_lim * mass), 0.0),
        'Превышение серы, %·т': max(float(s @ x - sulfur_lim * mass), 0.0),
        'Нарушение x ≥ 0, т': float(np.max(np.maximum(-x, 0.0))),
        'Превышение поставок, т': float(np.max(np.maximum(x - cap, 0.0)))
    }
    return {
        'остатки': residuals,
        'максимальное_нарушение': max(residuals.values()),
        'стоимость_независимая': float(c @ x),
        'средняя_зольность': float(a @ x / mass),
        'средняя_сера': float(s @ x / mass),
        'допустимо': max(residuals.values()) <= 1e-7
    }


check = independent_check(base.x, dataset)
check_df = pd.DataFrame([{'Проверка': key, 'Нарушение': value} for key, value in check['остатки'].items()])
check_df.to_csv(OUTPUT_DIR / 'verification.csv', index=False, encoding='utf-8-sig')
display(check_df)
print('Допустимость подтверждена:', check['допустимо'])
print(f'Независимая стоимость: {check["стоимость_независимая"]:,.6f} ₽')
print(f'Разность с результатом HiGHS: {abs(check["стоимость_независимая"] - base.fun):.9f} ₽')
assert check['допустимо']
assert np.isclose(check['стоимость_независимая'], base.fun, rtol=1e-12, atol=1e-7)

quality_slack = np.array([A_MAX * D - ashes @ base.x, S_MAX * D - sulfurs @ base.x])
upper_slack = limits - base.x
q_shadow = np.asarray(base.ineqlin.marginals, dtype=float)
u_shadow = np.asarray(base.upper.marginals, dtype=float)

economic_table = pd.DataFrame({
    'Ограничение': ['Зольность смеси', 'Сера смеси'] + [f'Лимит: {v}' for v in names],
    'Остаток': np.r_[quality_slack, upper_slack],
    'Активно': np.abs(np.r_[quality_slack, upper_slack]) < 1e-6,
    'Маржинальная оценка': np.r_[q_shadow, u_shadow],
    'Выгода от +1 единицы RHS, ₽': -np.r_[q_shadow, u_shadow]
})
display(economic_table)
print('Единицы: качество — ₽/(%·т), лимиты поставщиков — ₽/т дополнительной поставки.')
print(f'Увеличение допустимой зольности на 0,01 п.п.: локальный прогноз {q_shadow[0] * D * 0.01:,.2f} ₽')
print(f'Увеличение допустимой серы на 0,01 п.п.: локальный прогноз {q_shadow[1] * D * 0.01:,.2f} ₽')

A_ub = np.vstack((ashes, sulfurs))
A_eq = np.ones((1, 5))
m_eq = np.atleast_1d(base.eqlin.marginals)
m_lb = np.asarray(base.lower.marginals)
m_ub = np.asarray(base.upper.marginals)
m_ineq = np.asarray(base.ineqlin.marginals)
stationarity = prices - A_eq.T @ m_eq - A_ub.T @ m_ineq - m_lb - m_ub
kkt = {
    'Макс. невязка стационарности': float(np.max(abs(stationarity))),
    'Макс. произведение оценка × остаток (качество)': float(np.max(abs(m_ineq * quality_slack))),
    'Макс. произведение оценка × остаток (верхние границы)': float(np.max(abs(m_ub * upper_slack))),
    'Макс. произведение оценка × остаток (нижние границы)': float(np.max(abs(m_lb * base.x)))
}
display(pd.DataFrame([{'Проверка': k, 'Значение': v} for k, v in kkt.items()]))
print('marginals качества ≤ 0:', np.all(m_ineq <= 1e-8))
print('marginals верхних границ ≤ 0:', np.all(m_ub <= 1e-8))
print('marginals нижних границ ≥ 0:', np.all(m_lb >= -1e-8))
assert max(kkt.values()) < 1e-6

ash_step = 0.01
sulfur_step = 0.01
cap_step = 5.0
cap_candidates = np.where(np.abs(upper_slack) < 1e-7)[0]
cap_idx = int(cap_candidates[0]) if len(cap_candidates) else int(np.argmin(upper_slack))
caps_more = limits.copy()
caps_more[cap_idx] += cap_step

local_tests = [
    ('Зольность: +0,01 п.п.', base.fun + q_shadow[0] * D * ash_step,
     solve_mix(ash_limit=A_MAX + ash_step)),
    ('Сера: +0,01 п.п.', base.fun + q_shadow[1] * D * sulfur_step,
     solve_mix(sulfur_limit=S_MAX + sulfur_step)),
    (f'{names[cap_idx]}: +5 т лимита', base.fun + u_shadow[cap_idx] * cap_step,
     solve_mix(caps=caps_more))
]
local_rows = []
for title, prediction, result in local_tests:
    local_rows.append({'Изменение': title, 'Прогноз стоимости, ₽': prediction,
                       'Фактическая стоимость, ₽': result.fun if result.success else np.nan,
                       'Разница прогноза, ₽': (result.fun - prediction) if result.success else np.nan})
display(pd.DataFrame(local_rows))
pd.DataFrame(local_rows).to_csv(OUTPUT_DIR / 'shadow_comparison.csv', index=False, encoding='utf-8-sig')

cap_deficit = limits.copy()
cap_deficit[1] *= 0.8
cap_stress = limits.copy()
cap_stress[1] *= 0.8
cap_stress[2] *= 0.8

scenario_specs = [
    {'Название': 'Базовый', 'caps': limits, 'ash': A_MAX, 'sulfur': S_MAX},
    {'Название': 'Дефицитный', 'caps': cap_deficit, 'ash': A_MAX, 'sulfur': S_MAX},
    {'Название': 'Стрессовый', 'caps': cap_stress, 'ash': 10.5, 'sulfur': 0.9},
    {'Название': 'Жёсткий допустимый', 'caps': cap_stress, 'ash': 11.3, 'sulfur': 1.0}
]
scenario_results = {}
scenario_rows = []
scenario_check_rows = []
for spec in scenario_specs:
    name = spec['Название']
    result = solve_mix(caps=spec['caps'], ash_limit=spec['ash'], sulfur_limit=spec['sulfur'])
    scenario_results[name] = result
    verified = independent_check(result.x, dataset, caps=spec['caps'],
                                 ash_limit=spec['ash'], sulfur_limit=spec['sulfur']) if result.success else None
    scenario_rows.append({
        'Сценарий': name,
        'Статус': status_names.get(result.status, str(result.status)),
        'Стоимость, ₽': result.fun if result.success else np.nan,
        'Изменение, ₽': result.fun - base.fun if result.success else np.nan,
        'Зольность, %': verified['средняя_зольность'] if verified else np.nan,
        'Сера, %': verified['средняя_сера'] if verified else np.nan,
        'Итерации HiGHS': result.nit
    })
    if result.success:
        scenario_check_rows.append({'Сценарий': name, 'Проверка допустимости': verified['допустимо'],
                                    'Максимальная невязка': verified['максимальное_нарушение'],
                                    'Расхождение стоимости, ₽': abs(verified['стоимость_независимая'] - result.fun)})
scenario_df = pd.DataFrame(scenario_rows)
scenario_df.to_csv(OUTPUT_DIR / 'scenarios.csv', index=False, encoding='utf-8-sig')
display(scenario_df)
display(pd.DataFrame(scenario_check_rows))
pd.DataFrame(scenario_check_rows).to_csv(OUTPUT_DIR / 'scenario_verification.csv', index=False, encoding='utf-8-sig')
assert all(row['Проверка допустимости'] for row in scenario_check_rows)

feasible_names = [spec['Название'] for spec in scenario_specs if scenario_results[spec['Название']].success]
xmat = np.vstack([scenario_results[name].x for name in feasible_names])
positions = np.arange(len(feasible_names))
fig, ax = plt.subplots(figsize=(12, 5.1), constrained_layout=True)
bottom = np.zeros(len(feasible_names))
for i, nm in enumerate(names):
    ax.bar(positions, xmat[:, i], bottom=bottom, label=nm, color=colors[i], width=0.62)
    bottom += xmat[:, i]
ax.set(xticks=positions, xticklabels=feasible_names, ylabel='Тонны',
       title='Оптимальная структура закупок в допустимых сценариях')
ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)
fig.savefig(OUTPUT_DIR / '03_scenario_composition.png', bbox_inches='tight')
plt.show()

display(pd.DataFrame({'Поставщик': names, **{name: scenario_results[name].x for name in feasible_names}}))
pd.DataFrame({'Поставщик': names, **{name: scenario_results[name].x for name in feasible_names}}).to_csv(OUTPUT_DIR / 'scenario_plans.csv', index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
vals = [scenario_results[n].fun / 1e6 for n in feasible_names]
bar = ax[0].bar(np.arange(len(feasible_names)), vals, color=['#4472c4', '#e09f3e', '#d1495b'])
ax[0].set(xticks=np.arange(len(feasible_names)), xticklabels=feasible_names,
          title='Минимальная стоимость закупки', ylabel='Млн ₽')
ax[0].tick_params(axis='x', rotation=12)
for b, v in zip(bar, vals):
    ax[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f'{v:.3f}', ha='center')
ax[0].set_ylim(0, max(vals) * 1.15)
positions = np.arange(len(feasible_names))
width = 0.34
for i, name in enumerate(feasible_names):
    r = scenario_results[name]
    ax[1].bar(i - width/2, (ashes @ r.x / D) / scenario_specs[[s['Название'] for s in scenario_specs].index(name)]['ash'] * 100,
              width, color='#4472c4', label='Зольность / лимит' if i == 0 else None)
    ax[1].bar(i + width/2, (sulfurs @ r.x / D) / scenario_specs[[s['Название'] for s in scenario_specs].index(name)]['sulfur'] * 100,
              width, color='#e09f3e', label='Сера / лимит' if i == 0 else None)
ax[1].axhline(100, color='crimson', linestyle='--', label='100% допустимого')
ax[1].set(xticks=positions, xticklabels=feasible_names, title='Использование нормативов качества', ylabel='% от лимита')
ax[1].tick_params(axis='x', rotation=12)
ax[1].legend(fontsize=8)
fig.savefig(OUTPUT_DIR / '04_scenario_comparison.png', bbox_inches='tight')
plt.show()

stress = scenario_results['Стрессовый']
assert stress.status == 2
ash_aux = linprog(
    c=ashes,
    A_ub=np.array([sulfurs]),
    b_ub=np.array([0.9 * D]),
    A_eq=np.ones((1, 5)),
    b_eq=np.array([D]),
    bounds=list(zip(np.zeros(5), cap_stress)),
    method='highs-ds'
)
if not ash_aux.success:
    print('Даже баланс массы и требование по сере без ограничения зольности несовместны.')
else:
    min_ash_achievable = float(ashes @ ash_aux.x / D)
    print(f'При сере ≤ 0,9% и сокращённых поставках минимальная зольность = {min_ash_achievable:.6f}%')
    print('Стрессовый лимит зольности = 10,500000%')
    print(f'Неустранимое превышение по зольности = {min_ash_achievable - 10.5:.6f} процентного пункта')
    print('Следовательно, одновременно выполнить все ограничения стрессового сценария нельзя.')
    display(pd.DataFrame({'Поставщик': names, 'Минимизация зольности, т': ash_aux.x,
                          'Доступно при стрессе, т': cap_stress}))
    assert min_ash_achievable > 10.5

ash_grid = np.linspace(A_MAX - 2.0, A_MAX + 2.0, 41)
actual_costs = []
for a in ash_grid:
    rr = solve_mix(ash_limit=float(a))
    actual_costs.append(rr.fun / 1e6 if rr.success else np.nan)
predicted_costs = (base.fun + q_shadow[0] * D * (ash_grid - A_MAX)) / 1e6
fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
ax.plot(ash_grid, actual_costs, marker='o', markersize=3.2, label='Результат HiGHS')
ax.plot(ash_grid, predicted_costs, '--', linewidth=1.4, label='Локальная теневая оценка')
ax.scatter([A_MAX], [base.fun / 1e6], s=70, color='crimson', zorder=5, label='Базовая модель')
ax.set(title='Зависимость минимальной стоимости от допустимой зольности',
       xlabel='Лимит средней зольности, %', ylabel='Минимальная стоимость, млн ₽')
ax.legend()
fig.savefig(OUTPUT_DIR / '05_sensitivity.png', bbox_inches='tight')
plt.show()

base_active = [names[i] for i in range(5) if abs(base.x[i] - limits[i]) < 1e-7]
base_zero = [names[i] for i in range(5) if base.x[i] < 1e-7]
deficit_res = scenario_results['Дефицитный']
tight_res = scenario_results['Жёсткий допустимый']

# Дополнительное задание: сопоставление алгоритмов на одной и той же модели.
method_rows = []
for method in ('highs-ds', 'highs-ipm'):
    started = time.perf_counter()
    result = linprog(prices, A_ub=np.vstack((ashes, sulfurs)),
                     b_ub=[A_MAX * D, S_MAX * D], A_eq=np.ones((1, len(prices))),
                     b_eq=[D], bounds=list(zip(np.zeros(len(limits)), limits)),
                     method=method, options={'presolve': True})
    elapsed = time.perf_counter() - started
    assert result.success and np.isclose(result.fun, base.fun, atol=1e-5)
    method_rows.append({'Метод': method, 'Статус': status_names[result.status],
                        'Стоимость, ₽': result.fun, 'Итерации': result.nit,
                        'Время, с': elapsed})
methods = pd.DataFrame(method_rows)
methods.to_csv(OUTPUT_DIR / 'methods.csv', index=False, encoding='utf-8-sig')

conclusion = f"""**Базовый результат.** Оптимальная смесь массой {D:,.0f} т обходится в **{base.fun:,.2f} ₽** ({base.fun/D:,.2f} ₽/т). Средняя зольность — **{ash_base:.3f}%**, сера — **{sulfur_base:.3f}%**. Оба норматива качества активны.

**Ограниченность поставок.** Полностью загружены: **{', '.join(base_active) if base_active else 'нет'}**. Нулевые закупки: **{', '.join(base_zero) if base_zero else 'нет'}**. Все положительные закупки могут сосуществовать в оптимуме: даже сравнительно дорогой уголь позволяет разбавлять более загрязнённый дешёвый уголь.

**Дефицит.** Сокращение мощности поставщика № 2 на 20% приводит к стоимости **{deficit_res.fun:,.2f} ₽**, что на **{deficit_res.fun-base.fun:,.2f} ₽** больше базовой. Рост цены обусловлен необходимостью изменить состав смеси, сохранив экологические требования.

**Стресс.** Основной стрессовый сценарий имеет статус **{status_names[stress.status]}**. При одновременно заданных сокращениях отгрузки и сере ≤ 0,9% даже минимально достижимая зольность равна **{min_ash_achievable:.3f}%**, то есть выше допустимых 10,5%. Для восстановления допустимости придётся ослабить хотя бы одно условие или найти дополнительные поставки.

**Жёсткий, но выполнимый сценарий.** При нормативе зольности 11,3% и серы 1,0% стоимость составляет **{tight_res.fun:,.2f} ₽**, увеличение относительно базы — **{tight_res.fun-base.fun:,.2f} ₽**.

**Теневые цены.** Вблизи базового решения повышение допустимой зольности на 0,01 процентного пункта изменяет стоимость примерно на **{q_shadow[0]*D*0.01:,.2f} ₽**; повышение допустимой серы на 0,01 процентного пункта — на **{q_shadow[1]*D*0.01:,.2f} ₽**. Отрицательный знак означает экономию при ослаблении норматива.

**Проверка.** Все допустимые сценарии повторно проверены кодом NumPy с допуском 10⁻⁷; стоимость в базовом сценарии независимо пересчитана. Найденный план применим как агрегированный план закупки, но для реального ТЭС необходим учёт теплоты сгорания, влажности, логистики, условий контракта, качества партий и целочисленных ограничений перевозки."""
display(Markdown(conclusion))

import importlib.metadata

report = f"""# Лабораторная работа № 1. Вариант 14. Seed 10031

## Постановка
Смешение углей для ТЭС из пяти источников. Минимизируется стоимость закупки C = Σ cᵢxᵢ при Σxᵢ=1000 т, Σaᵢxᵢ≤12,3·1000, Σsᵢxᵢ≤1,25·1000 и 0≤xᵢ≤uᵢ. Зольность и сера заданы в процентах. Объёмы непрерывны.

Множество I = {{1,…,5}} — поставщики; xᵢ — закупка, т; cᵢ — цена, ₽/т; aᵢ и sᵢ — зольность и содержание серы, %; uᵢ — доступный объём, т. Цель измеряется в рублях, ограничения качества — в %·т. Баланс фиксирует спрос, остальные ограничения задают качество и доступность.

Решатель SciPy linprog / HiGHS, двойственный симплекс highs-ds, presolve включён, допуски прямой и двойственной допустимости 1e-9. Статус optimal означает найденный оптимум с учётом численных допусков; infeasible — отсутствие допустимого решения; unbounded — неограниченность цели; при лимите времени или ошибке оптимальность не подтверждена.

## Происхождение данных
Учебные параметры с seed 10031 сгенерированы программой generate_data.py; методичка не фиксирует численные исходные данные. Входной файл: data/coal_data.json.

## Результаты
```
{base_solution.to_string(index=False)}
```

Стоимость: {base.fun:,.2f} ₽; зольность: {ash_base:.5f}%; сера: {sulfur_base:.5f}%.

## Проверка
Максимальное нарушение: {check['максимальное_нарушение']:.3e}; допустимо: {check['допустимо']}. Цена пересчитана отдельно: {check['стоимость_независимая']:,.2f} ₽.

## Сценарии
```
{scenario_df.to_string(index=False)}
```

## Теневые оценки
```
{economic_table.to_string(index=False)}
```

Единицы маржинальных оценок качества — ₽/(%·т), ограничений поставщиков — ₽/т. Приращения неодинаковых единиц нельзя сравнивать напрямую. Для увеличения поставок при прочих равных выгоднее лимит с наибольшей положительной выгодой на тонну; см. таблицу. Теневые цены локальны и могут измениться при смене оптимального базиса.

Проверка локального прогноза (изменение нормы на 0,01 п.п. или лимита на 5 т):
```
{pd.DataFrame(local_rows).to_string(index=False)}
```
Для дефицитного сценария также сопоставим прогноз от исходной теневой цены с фактом: прогноз изменения {u_shadow[1] * (cap_deficit[1] - limits[1]):,.2f} ₽, фактически {deficit_res.fun - base.fun:,.2f} ₽. Отличие возможно при смене базиса; теневая цена гарантирует лишь локальный прогноз.

## Дополнительное задание: сравнение алгоритмов
```
{methods.to_string(index=False)}
```
Время одного запуска зависит от системы и не является устойчивым бенчмарком. Оба метода дали совпадающую целевую функцию; highs-ipm использует метод внутренней точки, highs-ds — двойственный симплекс.

## Выводы
{conclusion.replace('**', '')}

## Допущения
Линейное смешивание по массе, фиксированные цены, однородное сырьё, непрерывность закупок. Не учитываются калорийность, влажность, транспорт, потери и сезонность.
"""
(OUTPUT_DIR / 'report.md').write_text(report, encoding='utf-8')

requirements = '\n'.join([
    f'numpy=={np.__version__}', f'pandas=={pd.__version__}',
    f'scipy=={scipy.__version__}', f'matplotlib=={matplotlib.__version__}',
    f'ipython=={importlib.metadata.version("ipython")}'
]) + '\n'
(OUTPUT_DIR / 'requirements.txt').write_text(requirements, encoding='utf-8')

log_lines = [f'ЛР1 | Вариант {VARIANT} | seed {SEED}', f'Python {platform.python_version()}',
             f'SciPy {scipy.__version__}; HiGHS через linprog, method=highs-ds, presolve=True',
             'Протокол статусов и числа итераций, формируемый интерфейсом SciPy (не нативный stdout HiGHS).', '']
for spec in scenario_specs:
    res = scenario_results[spec['Название']]
    log_lines.extend([f"Сценарий: {spec['Название']}", f'Status: {status_names.get(res.status, str(res.status))}',
                      f'Message: {res.message}', f'Iterations: {res.nit}',
                      f'Objective: {res.fun if res.success else "N/A"}', ''])
(OUTPUT_DIR / 'solver.log').write_text('\n'.join(log_lines), encoding='utf-8')

for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name, '|', f'{file.stat().st_size:,} байт')
