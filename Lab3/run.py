"""Воспроизведение всех экспериментов ЛР № 3 одной командой."""

from pathlib import Path
import csv
import json
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
REPEATS = 5


def run_with_log():
    OUT.mkdir(exist_ok=True)
    log = OUT / 'solver.log'
    with log.open('wb') as stream:
        process = subprocess.run(
            [sys.executable, '-B', '-X', 'utf8', '-u', str(Path(__file__).resolve()), '--solver-worker'],
            stdout=stream, stderr=subprocess.STDOUT,
        )
    lines = log.read_text(encoding='utf-8').splitlines()
    log.write_text('\n'.join(line.rstrip() for line in lines) + '\n', encoding='utf-8')
    print(f"Lab3: {'OK' if process.returncode == 0 else 'ERROR'}. See Lab3/outputs/solver.log")
    return process.returncode


if __name__ == '__main__' and '--solver-worker' not in sys.argv:
    raise SystemExit(run_with_log())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from model import Problem, STATUSES, TOL, greedy, solve, verify
from report import write_report


def load_data(size):
    return json.loads((ROOT / 'data' / f'teams_{size}.json').read_text(encoding='utf-8'))


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_optimal(result, label):
    if not result.success:
        raise RuntimeError(f'{label}: {result.message}')
    return result


def logged_solve(label, p, **options):
    print(f'\n=== {label} ===', flush=True)
    result = require_optimal(solve(p, display=True, **options), label)
    print(f"status={STATUSES[result.status]}, score={-result.fun:.6f}, "
          f"mip_gap={result.get('mip_gap')}, nodes={result.get('mip_node_count')}")
    return result


def timed(call):
    samples = []
    for _ in range(REPEATS):
        start = perf_counter()
        result = call()
        samples.append(perf_counter() - start)
    return result, median(samples)


def integration_gap(lp, milp):
    return max(0.0, (lp - milp) / abs(milp)) * 100


def plan_rows(case, method, p, vector):
    x, y, h = p.unpack(vector)
    members, teams = [], []
    for j, team in enumerate(p.teams):
        selected = []
        for i, person in enumerate(p.people):
            if x[i, j] > TOL:
                selected.append(person['id'] if abs(x[i, j]-1) < TOL else f"{person['id']} ({x[i,j]:.4f})")
                members.append({'scenario': case, 'method': method, 'team': team['id'],
                                'candidate': person['id'], 'role': person['role'],
                                'assignment': x[i, j], 'score': p.scores[i, j],
                                'available_hours': person['hours']})
        teams.append({'scenario': case, 'method': method, 'team': team['id'],
                      'members': ', '.join(selected), 'internal_hours': p.hours @ x[:, j],
                      'required_hours': p.required[j], 'reserve_on': y[j], 'reserve_hours': h[j],
                      'gross_score': p.scores[:, j] @ x[:, j],
                      'reserve_cost': p.fixed[j]*y[j] + p.hourly[j]*h[j]})
    return members, teams


def verified(case, method, p, vector, *, expected=None, **options):
    if not np.all(np.isfinite(vector)):
        raise RuntimeError(f'Некорректный вектор: {case}, {method}')
    score, details = verify(p, vector, **options)
    maximum = max(row['violation'] for row in details)
    discrepancy = abs(score - expected) if expected is not None else None
    if maximum > TOL or (discrepancy is not None and discrepancy > TOL):
        raise RuntimeError(f'Проверка не пройдена: {case}, {method}, {maximum}, {discrepancy}')
    summary = {'scenario': case, 'method': method, 'score_recalculated': score,
               'objective_error': discrepancy, 'max_violation': maximum, 'tolerance': TOL}
    return score, summary, [{'scenario': case, 'method': method, **row} for row in details]


def compare_formulations(base, lp, milp):
    results = []
    for name, factor, tightened in (('tight_M', 1, False), ('large_M', 100, False),
                                    ('tight_M_with_cuts', 1, True)):
        options = {'m_factor': factor, 'tightened': tightened}
        label = f'Формулировка {name}'
        relaxed = lp if name == 'tight_M' else logged_solve(f'{label}, LP', base, integer=False, **options)
        integer = milp if name == 'tight_M' else logged_solve(f'{label}, MILP', base, integer=True, **options)
        for is_integer, result in ((False, relaxed), (True, integer)):
            verified('base', name, base, result.x, expected=-result.fun, integer=is_integer, **options)
        if abs(integer.fun - milp.fun) > TOL:
            raise RuntimeError(f'Целочисленные формулировки дали разные цели: {label}')
        measured, elapsed = timed(lambda: solve(base, integer=True, **options))
        require_optimal(measured, label)
        _, activation, reserve = base.unpack(relaxed.x)
        results.append({'formulation': name, 'M': factor * int(base.caps[0]),
                        'lp_score': -relaxed.fun, 'milp_score': -integer.fun,
                        'integrality_gap_pct': integration_gap(-relaxed.fun, -integer.fun),
                        'mip_gap_pct': 100 * integer.mip_gap, 'mip_nodes': integer.mip_node_count,
                        'median_milp_ms': 1000 * elapsed, 'lp_reserve_T02': reserve[1],
                        'lp_activation_T02': activation[1], 'repeats': REPEATS})
    return results


def scaling_experiment():
    rows = []
    for size in (12, 24, 48):
        p = Problem(load_data(size), {})
        lp = logged_solve(f'Масштаб {size}, LP', p, integer=False)
        milp = logged_solve(f'Масштаб {size}, MILP', p, integer=True)
        verified(f'scale_{size}', 'MILP', p, milp.x, expected=-milp.fun, integer=True)
        measured, lp_seconds = timed(lambda: solve(p, integer=False))
        require_optimal(measured, f'Масштаб {size}, LP')
        measured, milp_seconds = timed(lambda: solve(p, integer=True))
        require_optimal(measured, f'Масштаб {size}, MILP')
        rows.append({'candidates': size, 'teams': p.k, 'binary_variables': p.n*p.k+p.k,
                     'continuous_variables': p.k, 'lp_status': STATUSES[lp.status],
                     'milp_status': STATUSES[milp.status], 'lp_score': -lp.fun,
                     'milp_score': -milp.fun, 'lp_median_ms': 1000*lp_seconds,
                     'milp_median_ms': 1000*milp_seconds, 'integrality_gap_pct': integration_gap(-lp.fun, -milp.fun),
                     'mip_gap_pct': 100*milp.mip_gap, 'mip_nodes': milp.mip_node_count, 'repeats': REPEATS})
    return rows


def plot_scaling(rows):
    sizes = [r['candidates'] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3), layout='constrained')
    for key, title in (('lp_median_ms', 'LP'), ('milp_median_ms', 'MILP')):
        ax[0].plot(sizes, [r[key] for r in rows], 'o-', label=title)
    ax[0].set(title='Время построения и решения', ylabel='Медиана, мс', yscale='log')
    for key, title, marker in (('integrality_gap_pct', 'Интегральный зазор', 'o'),
                               ('mip_gap_pct', 'Итоговый зазор HiGHS', 's')):
        ax[1].plot(sizes, [r[key] for r in rows], marker + '-', label=title)
    ax[1].set(title='Зазоры при росте задачи', ylabel='Зазор, %')
    for panel in ax:
        panel.set(xlabel='Число кандидатов', xticks=sizes)
        panel.grid(alpha=0.25)
        panel.legend()
    fig.savefig(OUT / 'scaling.png', dpi=160)
    plt.close(fig)


def main():
    print(f'Lab3: Python {platform.python_version()}, SciPy {scipy.__version__}, '
          f'NumPy {np.__version__}, Matplotlib {matplotlib.__version__}')
    data = load_data(12)
    base = Problem(data, data['scenarios'][0])
    lp = logged_solve('Базовый сценарий, LP-релаксация', base, integer=False)
    chosen, teams, checks, violations, scenarios, heuristics = [], [], [], [], [], []

    def save_plan(case, method, p, vector, expected, integer):
        score, summary, detail = verified(case, method, p, vector, expected=expected, integer=integer)
        selection, capacity = plan_rows(case, method, p, vector)
        chosen.extend(selection)
        teams.extend(capacity)
        checks.append(summary)
        violations.extend(detail)
        return score

    save_plan('base', 'LP', base, lp.x, -lp.fun, False)
    for scenario in data['scenarios']:
        case = scenario['id']
        p = Problem(data, scenario)
        result = logged_solve(f"{scenario['name']} сценарий, MILP", p, integer=True)
        if case == 'base':
            base_milp = result
        score = save_plan(case, 'MILP', p, result.x, -result.fun, True)
        scenarios.append({'scenario': scenario['name'], 'id': case, 'status': STATUSES[result.status],
                          'score': score, 'change_from_base': score + base_milp.fun,
                          'best_bound': -result.mip_dual_bound, 'mip_gap_pct': 100*result.mip_gap,
                          'mip_nodes': result.mip_node_count,
                          'unavailable': ', '.join(scenario['unavailable']),
                          'extra_conflicts': ', '.join('/'.join(pair) for pair in scenario['extra_conflicts']),
                          'extra_required_hours': scenario['extra_required_hours']})
        measured, milp_seconds = timed(lambda: solve(p, integer=True))
        require_optimal(measured, case)
        vector, greedy_seconds = timed(lambda: greedy(p))
        greedy_score = None if vector is None else save_plan(case, 'greedy', p, vector, None, True)
        heuristics.append({'scenario': scenario['name'], 'milp_score': score, 'greedy_score': greedy_score,
                           'greedy_status': 'failed_to_construct' if vector is None else 'feasible',
                           'shortfall_pct': None if greedy_score is None else 100*(score-greedy_score)/abs(score),
                           'milp_median_ms': 1000*milp_seconds, 'greedy_median_ms': 1000*greedy_seconds,
                           'repeats': REPEATS})

    comparisons = compare_formulations(base, lp, base_milp)
    scaling = scaling_experiment()
    for name, rows in (('solution.csv', chosen), ('team_plans.csv', teams),
                       ('verification.csv', checks), ('verification_details.csv', violations),
                       ('scenarios.csv', scenarios), ('greedy.csv', heuristics),
                       ('formulations.csv', comparisons), ('scaling.csv', scaling)):
        write_csv(name, rows)
    plot_scaling(scaling)
    write_report(OUT / 'report.md', data, scenarios, teams, checks, heuristics, comparisons, scaling, REPEATS)


if __name__ == '__main__':
    main()
