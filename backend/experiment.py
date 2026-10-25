# -*- coding: utf-8 -*-
"""
Покроковий запуск глобального тесту (регламент розд. 10): 9 лекцій (dataset/manifest.json, split=test) × конфігурації M1–M4.
Стан визначається за файлами в results/<конфігурація>/<лекція>/ (відновлюється після будь-якої перерви):
  gen      — summary.json із complete=true (граф конвеєра lecture_graph.run_lecture для M1–M4);
  arbiter  — arbiter_run.json має оцінку (без тексту лекції, незалежне розв'язання) і рівень Блума для всіх питань;
  metrics  — metrics_items.csv (CV, косинус).
M3 і M4 однієї лекції використовують спільний передаліз Аналітика (results/_shared/<лекція>/analyst.json, alloc.json).

  python experiment.py status
  python experiment.py run M1 [--budget 100] [--lec W01]    # виконує наступні етапи конфігурації в межах бюджету, с
  python experiment.py index                                   # results/index.json для stats.py (лише завершені)
"""
import csv, json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, HERE)
RES = os.path.join(ROOT, "results"); CONFIGS = ["M1", "M2", "M3", "M4"]; N = 12
MODE = {"M1": "m1", "M2": "m2", "M3": "m3", "M4": "m4"}
# Модель арбітра за конфігурацією (рішення викладача): M1 — claude-sonnet-4-6 (оцінено до переходу), решта — з .env (ARBITER_MODEL=claude-sonnet-5-5)
ARBITER_OVERRIDE = {}      # усі конфігурації (нові запуски) — claude-haiku-4-5; M1 і M2 уже оцінено Sonnet (поле arbiter_model)


def lectures():
    m = json.load(open(os.path.join(ROOT, "dataset", "manifest.json"), encoding="utf-8"))
    return [x["id"] for x in m if x["split"] == "test"]

def out(cfg, lec): return os.path.join(RES, cfg, lec)
def pdf(lec): return os.path.join(ROOT, "dataset", "pdf", lec + ".pdf")

def done_gen(cfg, lec):
    f = os.path.join(out(cfg, lec), "summary.json")
    return os.path.exists(f) and json.load(open(f, encoding="utf-8")).get("complete") is True

def items(cfg, lec):
    import metrics
    return metrics.final_mcqs(out(cfg, lec))

def done_arbiter(cfg, lec):
    f = os.path.join(out(cfg, lec), "arbiter_run.json")
    if not os.path.exists(f) or not done_gen(cfg, lec): return False
    A = json.load(open(f, encoding="utf-8")); ids = [r["id"] for r in items(cfg, lec)]
    return all(i in A and "0" in A[i] and "error" not in A[i]["0"] and "bloom_by_judge" in A[i]["0"] for i in ids)

def done_metrics(cfg, lec): return os.path.exists(os.path.join(out(cfg, lec), "metrics_items.csv"))


def stage_gen(cfg, lec, budget):
    import lecture_graph as LG
    d = out(cfg, lec); os.makedirs(d, exist_ok=True); sh = os.path.join(RES, "_shared", lec)
    if cfg in ("M3", "M4"):
        for f in ("analyst.json", "alloc.json"):
            if os.path.exists(os.path.join(sh, f)) and not os.path.exists(os.path.join(d, f)): shutil.copy(os.path.join(sh, f), os.path.join(d, f))
    r = LG.run_lecture(pdf(lec), d, n=N, mode=MODE[cfg], budget_s=budget, workers=6)
    if cfg in ("M3", "M4"):
        os.makedirs(sh, exist_ok=True)
        for f in ("analyst.json", "alloc.json"):
            if os.path.exists(os.path.join(d, f)) and not os.path.exists(os.path.join(sh, f)): shutil.copy(os.path.join(d, f), os.path.join(sh, f))
    return r


def stage_arbiter(cfg, lec, budget):
    import arbiter as A, critic as C, lecture_graph as LG
    A.ARBITER_MODEL = ARBITER_OVERRIDE.get(cfg, A.ARBITER_MODEL)
    d = out(cfg, lec); f = os.path.join(d, "arbiter_run.json"); t0 = time.time()
    R = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    text = None; blocks = {}
    if cfg in ("M1", "M2"): text = C.truncate_source_text(LG.read_lecture(pdf(lec)), C.RL.limit("context_chars"))
    else: blocks = {b["id"]: b for b in json.load(open(os.path.join(d, "analyst.json"), encoding="utf-8"))["blocks"]}
    todo = []
    for it in items(cfg, lec):
        o = (R.get(it["id"]) or {}).get("0")
        if o and "error" not in o and "bloom_by_judge" in o: continue
        todo.append(it)
    def job(it):
        if time.time() - t0 > budget: return it["id"], None
        m = it["mcq"]; k = m["key"]; ds = [x["text"] if isinstance(x, dict) else x for x in m["distractors"]]
        ctx = text if text is not None else blocks[json.load(open(os.path.join(d, it["id"] + ".json"), encoding="utf-8"))["block"]]["text"]
        try:
            o = A.judge_item(ctx, m["question"], k, ds); o.update(A.classify_bloom(ctx, m["question"], k, ds, share_ctx=text is not None))
            o["arbiter_model"] = A.ARBITER_MODEL; o["target_level"] = it["level"]; o["bloom_match"] = int(o["bloom_by_judge"] == int(it["level"]))
            return it["id"], o
        except Exception as e:
            return it["id"], {"error": repr(e)[:200]}
    with ThreadPoolExecutor(12) as ex:
        if todo and text is not None:      # прогрів кешу: перше питання окремо, решта читають кеш лекції
            i0, o0 = job(todo[0])
            if o0 is not None: R[i0] = {"0": o0}
            todo = todo[1:]
        for id_, o in ex.map(job, todo):
            if o is not None: R[id_] = {"0": o}
    print(f"  арбітр {cfg}/{lec}: токени {A.USAGE}, ≈${A.usage_cost():.3f}")
    json.dump(R, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def stage_metrics(cfg, lec):
    import metrics; metrics.run(out(cfg, lec))


def status():
    L = lectures(); print("конфіг  " + "  ".join(f"{l:>4}" for l in L))
    for c in CONFIGS:
        row = []
        for l in L:
            g, a, m = done_gen(c, l), done_arbiter(c, l), done_metrics(c, l)
            row.append(f"{'G' if g else '.'}{'A' if a else '.'}{'M' if m else '.'}".rjust(4))
        print(f"{c:6}  " + "  ".join(row))
    print("G — генерація (і цикл Критика), A — арбітр+Блум, M — метрики")


def run(cfg, budget=100.0, only=None):
    t0 = time.time(); rest = lambda: budget - (time.time() - t0)
    for lec in lectures():
        if only and lec != only: continue
        for stage in ("gen", "arbiter", "metrics"):
            if rest() < 8: print("бюджет вичерпано"); return
            if stage == "gen" and not done_gen(cfg, lec):
                r = stage_gen(cfg, lec, rest()); print(cfg, lec, "gen", r.get("statuses"), "complete" if r.get("complete") else "НЕ завершено")
                if not r.get("complete"): return
            elif stage == "arbiter" and done_gen(cfg, lec) and not done_arbiter(cfg, lec):
                stage_arbiter(cfg, lec, rest()); print(cfg, lec, "arbiter", "готово" if done_arbiter(cfg, lec) else "частково")
                if not done_arbiter(cfg, lec): return
            elif stage == "metrics" and done_arbiter(cfg, lec) and not done_metrics(cfg, lec):
                stage_metrics(cfg, lec); print(cfg, lec, "metrics готово")
    print(cfg, "усі етапи виконано" if all(done_metrics(cfg, l) for l in lectures()) else "є незавершені етапи")


def index():
    idx = {c: {l: out(c, l) for l in lectures() if done_arbiter(c, l) and done_metrics(c, l)} for c in CONFIGS}
    idx = {c: v for c, v in idx.items() if v}
    json.dump(idx, open(os.path.join(RES, "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps({c: sorted(v) for c, v in idx.items()}, ensure_ascii=False))


if __name__ == "__main__":
    a = sys.argv[1:]; opt = lambda n, d=None: a[a.index(n) + 1] if n in a else d
    if not a or a[0] == "status": status()
    elif a[0] == "run": run(a[1], float(opt("--budget", 100)), opt("--lec"))
    elif a[0] == "index": index()
