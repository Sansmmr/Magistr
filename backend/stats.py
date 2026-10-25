# -*- coding: utf-8 -*-
"""
Статистичний аналіз глобального тесту (M1–M4), регламент розд. 7.3.

Вхід: індекс прогонів JSON  {"M1": {"A02": "<каталог>", ...}, "M2": {...}, ...}
      (каталог — результат lecture_graph.py з metrics.csv і arbiter_run.json від arb_run.py/arbiter).
Одиниця аналізу — питання; питання з однієї лекції залежні, тому довірчі інтервали подаються двома способами:
  (а) Вілсон (за незалежності питань; нижня межа обережності), (б) кластерний bootstrap за лекціями (≥ 3 лекцій; інакше — за питаннями).
Порівняння конфігурацій (питання різні, тому НЕ парні): точний тест Фішера для часток (прийнятність, бали F/A/P/S),
Манна–Уітні для CV і косинуса; розміри ефекту: різниця часток з ІД bootstrap, відношення шансів, h Коена; r (rank-biserial).
Поправка Холма на множинні порівняння. Окремо: Cohen's kappa арбітра з розміткою викладача (функція kappa).

Використання:  python stats.py index.json [out.json]
"""
import csv, json, math, os, random, sys
from itertools import combinations
import numpy as np
from scipy import stats as st

CRIT = ("factuality", "no_ambiguity", "plausibility", "no_shortcuts")


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_items(index):
    """Повертає список питань: {config, lecture, id, accepted, F, A, P, S, cv_words, cv_chars, key_longest, mean_cosine}."""
    rows = []
    for cfg, lects in index.items():
        for lec, d in lects.items():
            d = os.path.expanduser(d)
            mi = {r["id"]: r for r in csv.DictReader(open(os.path.join(d, "metrics_items.csv"), encoding="utf-8"))}
            ar = json.load(open(os.path.join(d, "arbiter_run.json"), encoding="utf-8"))
            for id_, m in mi.items():
                o = (ar.get(id_) or {}).get("0")
                if not o or "error" in o: continue
                rows.append({"config": cfg, "lecture": lec, "id": id_, "accepted": int(o["is_accepted"]),
                             "factuality": o["factuality"], "no_ambiguity": o["no_ambiguity"], "plausibility": o["plausibility"], "no_shortcuts": o["no_shortcuts"],
                             "cv_words": float(m["cv_words"]), "cv_chars": float(m["cv_chars"]),
                             "key_longest": int(m["key_longest"] == "True"), "mean_cosine": float(m["mean_cosine"]),
                             "solver_agrees": int(bool((o.get("blind_solver") or {}).get("agrees_with_key"))), "bloom_match": int(o.get("bloom_match", -1))})
    return rows


def cluster_boot(rows, key, n_boot=4000, seed=1):
    """Bootstrap-ІД середнього за лекціями (кластерами); якщо лекцій < 3 — за питаннями."""
    rnd = random.Random(seed); by = {}
    for r in rows: by.setdefault(r["lecture"], []).append(r[key])
    cl = list(by.values())
    if len(cl) >= 3:
        means = []
        for _ in range(n_boot):
            pick = [cl[rnd.randrange(len(cl))] for _ in cl]; v = [x for c in pick for x in c]; means.append(sum(v) / len(v))
        mode = "лекції"
    else:
        v = [r[key] for r in rows]; means = [sum(rnd.choices(v, k=len(v))) / len(v) for _ in range(n_boot)]; mode = "питання"
    means.sort(); return means[int(.025 * n_boot)], means[int(.975 * n_boot) - 1], mode


def boot_diff(a, b, key, n_boot=4000, seed=2):
    rnd = random.Random(seed)
    ga, gb = {}, {}
    for r in a: ga.setdefault(r["lecture"], []).append(r[key])
    for r in b: gb.setdefault(r["lecture"], []).append(r[key])
    common = sorted(set(ga) & set(gb))
    if len(common) >= 3:
        ds = []
        for _ in range(n_boot):
            pick = [common[rnd.randrange(len(common))] for _ in common]
            va = [x for l in pick for x in ga[l]]; vb = [x for l in pick for x in gb[l]]
            ds.append(sum(va) / len(va) - sum(vb) / len(vb))
        mode = "лекції"
    else:
        va = [r[key] for r in a]; vb = [r[key] for r in b]
        ds = [sum(rnd.choices(va, k=len(va))) / len(va) - sum(rnd.choices(vb, k=len(vb))) / len(vb) for _ in range(n_boot)]; mode = "питання"
    ds.sort(); return ds[int(.025 * n_boot)], ds[int(.975 * n_boot) - 1], mode


def cohen_h(p1, p2): return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i]); m = len(pvals); adj = [0.0] * m; prev = 0.0
    for r, i in enumerate(idx):
        prev = max(prev, min(1.0, (m - r) * pvals[i])); adj[i] = prev
    return adj


def kappa(a, b):
    """Cohen's kappa для двох бінарних/категоріальних розміток однакової довжини (арбітр vs викладач)."""
    n = len(a); cats = sorted(set(a) | set(b)); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def analyze(rows):
    cfgs = sorted({r["config"] for r in rows}); by = {c: [r for r in rows if r["config"] == c] for c in cfgs}
    out = {"configs": {}, "pairs": []}
    for c in cfgs:
        R = by[c]; n = len(R); d = {"n": n, "lectures": sorted({r["lecture"] for r in R})}
        for k in ("accepted",) + CRIT + ("key_longest", "solver_agrees") + (("bloom_match",) if all(r["bloom_match"] >= 0 for r in R) else ()):
            kk = sum(r[k] for r in R); lo, hi = wilson(kk, n); blo, bhi, mode = cluster_boot(R, k)
            d[k] = {"k": kk, "pct": round(100 * kk / n, 1), "wilson": [round(100 * lo, 1), round(100 * hi, 1)], "cluster_boot": [round(100 * blo, 1), round(100 * bhi, 1)], "boot_unit": mode}
        for k in ("cv_words", "cv_chars", "mean_cosine"):
            v = [r[k] for r in R]; d[k] = {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v, ddof=1)) if n > 1 else 0.0, 3), "median": round(float(np.median(v)), 3)}
        out["configs"][c] = d
    raw = []
    for a, b in combinations(cfgs, 2):
        A, B = by[a], by[b]
        pr = {"pair": f"{a} vs {b}", "tests": {}}
        for k in ("accepted",) + CRIT:
            ka, kb = sum(r[k] for r in A), sum(r[k] for r in B)
            _, p = st.fisher_exact([[ka, len(A) - ka], [kb, len(B) - kb]])
            pa, pb = ka / len(A), kb / len(B); lo, hi, mode = boot_diff(A, B, k)
            odds = ((ka + .5) * (len(B) - kb + .5)) / ((len(A) - ka + .5) * (kb + .5))
            pr["tests"][k] = {"p_fisher": p, "diff_pct": round(100 * (pa - pb), 1), "diff_ci": [round(100 * lo, 1), round(100 * hi, 1)], "boot_unit": mode, "odds_ratio": round(odds, 2), "h": round(cohen_h(pa, pb), 2)}
            raw.append((pr, k, p))
        for k in ("cv_words", "cv_chars", "mean_cosine"):
            va, vb = [r[k] for r in A], [r[k] for r in B]
            u, p = st.mannwhitneyu(va, vb, alternative="two-sided")
            rb = 1 - 2 * u / (len(va) * len(vb))      # rank-biserial: >0 — значення A менші
            pr["tests"][k] = {"p_mwu": p, "median_a": round(float(np.median(va)), 3), "median_b": round(float(np.median(vb)), 3), "rank_biserial": round(rb, 2)}
            raw.append((pr, k, p))
        out["pairs"].append(pr)
    for (pr, k, p), q in zip(raw, holm([p for _, _, p in raw])): pr["tests"][k]["p_holm"] = q
    return out


def report(res):
    L = ["Конфігурації (n питань; % з ІД Вілсона | кластерний bootstrap):"]
    for c, d in res["configs"].items():
        a = d["accepted"]; L.append(f"  {c}: n={d['n']}, лекції {','.join(d['lectures'])}; прийнято арбітром {a['k']} ({a['pct']}%; Вілсон {a['wilson']}; bootstrap {a['cluster_boot']} за {a['boot_unit']}); "
                                    f"CV слів {d['cv_words']['mean']}±{d['cv_words']['sd']}; CV символів {d['cv_chars']['mean']}; косинус {d['mean_cosine']['mean']}")
    L.append("Пари (прийнятність арбітром; p за Фішером і Холмом; різниця, п.п.):")
    for pr in res["pairs"]:
        t = pr["tests"]["accepted"]; L.append(f"  {pr['pair']}: різниця {t['diff_pct']} п.п. (ІД {t['diff_ci']}), OR {t['odds_ratio']}, h {t['h']}, p={t['p_fisher']:.3f}, p_Holm={t['p_holm']:.3f}")
    return "\n".join(L)


if __name__ == "__main__":
    idx = json.load(open(sys.argv[1], encoding="utf-8")); rows = load_items(idx); res = analyze(rows)
    print(report(res))
    if len(sys.argv) > 2: json.dump(res, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
