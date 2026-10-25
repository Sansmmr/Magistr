# -*- coding: utf-8 -*-
"""
Конвеєр агентів у вигляді графа станів LangGraph — для всіх конфігурацій M1–M4.

Спільний для двох точок входу:
  - вебзастосунок (main.py, POST /api/generate-lecture): текст лекції користувача, конфігурація M4, журнал подій через WebSocket;
  - модуль експериментів (experiment.py): лекції набору, конфігурації M1–M4.

Граф лекції (LectureState):
    check_size -> analyst -> allocate -> generate -> verify -> summarize -> END      (M3, M4)
    check_size -> generate -> verify -> summarize -> END                             (M2: без аналітика, уся лекція)
    check_size -> generate -> summarize -> END                                       (M1: без аналітика й без критика)
Граф завдання (ItemState), який вузол verify запускає для кожного завдання:
    critic --(схвалено | M3 | ліміт ітерацій | вичерпано бюджет)--> END
    critic --(відхилено)--> refine --(нова версія)--> critic
    refine --(генератор повідомив про неможливість виправлення: hard_reject, формула 2.8)--> END

Мовні судження виконують агенти (аналітик і критик — critic.py, генератор — generator.py); код лише рахує, розподіляє
й маршрутизує. Кожен вузол зберігає результат у каталозі прогону (analyst.json, alloc.json, gen.json, bNN_i.json, summary.json),
тому перерваний прогін відновлюється з того самого місця. У M1 і M2 уся лекція є блоком 0 (файли b00_i.json).

Використання:  python lecture_graph.py <лекція.pdf|.txt> <каталог_результатів> [--n 12] [--mode m1|m2|m3|m4]
"""
import json, math, os, re, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import critic as C
import generator as PL
import rules_loader as RL

LogFn = Optional[Callable[[Dict[str, Any]], None]]
FLAT = ("m1", "m2")          # конфігурації без аналітика

# ---------------------------------------------------------------- вхідні дані (вузол check_size)
def read_lecture(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    return open(path, encoding="utf-8").read()


def add_markers(text: str, step: int):
    """Позначки ⟦n⟧ приблизно через кожні `step` символів (на межі рядка). Повертає (текст із позначками, список зміщень)."""
    cuts = [0]
    for m in re.finditer(r"\n", text):
        if m.start() - cuts[-1] >= step:
            cuts.append(m.start() + 1)
    marked = "".join(f"⟦{k + 1}⟧" + text[cuts[k]:(cuts[k + 1] if k + 1 < len(cuts) else len(text))] for k in range(len(cuts)))
    return marked, cuts


# ---------------------------------------------------------------- агент-аналітик: сегментація лекції (вузол analyst)
def _validate(blocks, m):
    errs, exp = [], 1
    for b in blocks:
        a, z = int(b["start_marker"]), int(b["end_marker"])
        if a != exp or z < a:
            errs.append(f"блок {a}-{z}: очікувалась початкова позначка {exp}")
        exp = z + 1
    if exp != m + 1:
        errs.append(f"блоки закінчуються на {exp - 1}, а остання позначка {m}")
    return errs


def _repair(blocks, m):
    """Арифметичне впорядкування: сортування, усунення перекриттів і прогалин, продовження останнього блока до кінця."""
    out, exp = [], 1
    for b in sorted(blocks, key=lambda b: (int(b["start_marker"]), int(b["end_marker"]))):
        z = int(b["end_marker"])
        if z < exp:
            continue
        b = dict(b); b["start_marker"] = exp; b["end_marker"] = min(z, m)
        out.append(b); exp = b["end_marker"] + 1
    if out and exp <= m:
        out[-1]["end_marker"] = m
    return out


def _split_big(blocks, cuts, text, max_chars):
    """Блок, довший за max_chars, ділиться за позначками на рівні частини (арифметика)."""
    res = []
    for b in blocks:
        a, z = int(b["start_marker"]), int(b["end_marker"])
        n = z - a + 1
        size = (cuts[z] if z < len(cuts) else len(text)) - cuts[a - 1]
        parts = min(n, max(1, math.ceil(size / max_chars)))
        if parts == 1:
            res.append(b); continue
        for i in range(parts):
            lo, hi = a + i * n // parts, a + (i + 1) * n // parts - 1
            nb = dict(b); nb.update(start_marker=lo, end_marker=hi, title=f'{b["title"]} (частина {i + 1})')
            res.append(nb)
    return res


def analyze_lecture(text: str) -> dict:
    C.check_lecture_size(text)
    step = RL.limit("marker_step_chars")
    marked, cuts = add_markers(text, step)
    m = len(cuts)
    msgs = [{"role": "user", "content": f'Усього позначок: {m}.\nТекст:\n"""\n{marked}\n"""'}]
    errs_log = []
    for _ in range(3):
        out = C.analyst_chat(C.RL.prompt("analyst_classify"), msgs, 0.0)
        errs = _validate(out["blocks"], m)
        if not errs:
            break
        errs_log.append(errs)
        msgs += [{"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
                 {"role": "user", "content": "Помилки: " + "; ".join(errs) + ". Виправ і поверни JSON."}]
    repaired = bool(errs)
    if repaired:
        out["blocks"] = _repair(out["blocks"], m)
    out["blocks"] = _split_big(out["blocks"], cuts, text, RL.limit("block_max_chars"))
    for i, b in enumerate(out["blocks"], 1):
        a, z = int(b["start_marker"]), int(b["end_marker"])
        b["id"] = i
        b["text"] = text[cuts[a - 1]:(cuts[z] if z < m else len(text))].strip()
    out["markers"] = m; out["analyst_errors"] = errs_log; out["repaired_by_code"] = repaired
    return out


# ---------------------------------------------------------------- розподіл завдань і рівнів, довідка блока (вузли allocate, generate)
def allocate(blocks, n):
    content = [b for b in blocks if b.get("type") == "content"]
    pick = content if len(content) <= n else [content[int(i * len(content) / n)] for i in range(n)]
    cnt = {b["id"]: 1 for b in pick}
    while sum(cnt.values()) < n:
        b = min(pick, key=lambda b: cnt[b["id"]] / max(1, len(b["text"]))); cnt[b["id"]] += 1
    base = n // 3
    need = {1: base, 2: base, 3: n - 2 * base}
    lv = {b["id"]: [] for b in pick}
    for lvl in (3, 2, 1):
        for _ in range(need[lvl]):
            free = lambda b: len(lv[b["id"]]) < cnt[b["id"]]
            el = [b for b in pick if (b.get("bloom") or 1) >= lvl and free(b) and lvl not in lv[b["id"]]] \
                or [b for b in pick if (b.get("bloom") or 1) >= lvl and free(b)] or [b for b in pick if free(b)]
            b = max(el, key=lambda b: (cnt[b["id"]] - len(lv[b["id"]]), -b["id"]))
            lv[b["id"]].append(lvl)
    return {str(k): sorted(v) for k, v in lv.items() if v}


def passport(analysis, block) -> str:
    t = block["text"].lower(); seen, rows = set(), []
    for g in analysis.get("glossary") or []:
        k = g["term"].lower()
        if k in t and k not in seen:
            seen.add(k); rows.append(f'- {g["term"]}: {g["explanation"]}')
    return f'Предметна галузь: {analysis.get("domain")}\nТерміни, що трапляються в цьому блоці:\n' + ("\n".join(rows) or "- (немає)")


# ---------------------------------------------------------------- підсумок прогону (вузол summarize)
def summarize_lecture(A, AL, states, outdir, mode, cap, done):
    from collections import Counter
    cv = []
    for s in states:
        m = s["iterations"][-1]["mcq"]; L = [C.word_count(m["key"])] + [C.word_count(d["text"]) for d in m["distractors"]]
        cv.append(statistics.pstdev(L) / statistics.mean(L))
    sm = {"mode": mode, "complete": done, "domain": A.get("domain"), "blocks": len(A["blocks"]),
          "content_blocks": sum(b["type"] == "content" for b in A["blocks"]), "markers": A.get("markers"),
          "analyst_repaired_by_code": A.get("repaired_by_code"), "questions": len(states), "max_iterations": cap,
          "statuses": dict(Counter(s["status"] for s in states)), "cv_length_mean": round(statistics.mean(cv), 3) if cv else None}
    json.dump(sm, open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return sm


def summarize_flat(states, outdir, mode, cap, done):
    """Підсумок M1/M2 (без аналітика): той самий формат summary.json, що й до переходу на граф."""
    from collections import Counter
    cv = []
    for s in states:
        m = s["iterations"][-1]["mcq"]; L = [C.word_count(m["key"])] + [C.word_count(d["text"]) for d in m["distractors"]]
        cv.append(statistics.pstdev(L) / statistics.mean(L))
    sm = {"mode": mode, "complete": done, "questions": len(states),
          "max_iterations": cap if mode == "m2" else 0, "statuses": dict(Counter(s["status"] for s in states)),
          "cv_length_mean": round(statistics.mean(cv), 3) if cv else None}
    json.dump(sm, open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return sm


# ---------------------------------------------------------------- стани
class LectureState(TypedDict, total=False):
    path: str                 # файл лекції (.pdf | .txt)
    outdir: str               # каталог прогону
    n: int                    # кількість завдань
    mode: str                 # "m1" | "m2" | "m3" | "m4"
    budget_s: Optional[float] # обмеження часу (для експериментів), None — без обмеження
    t0: float
    workers: int
    text: str
    analysis: Dict[str, Any]  # результат аналітика: domain, glossary, blocks
    alloc: Dict[str, List[int]]   # id блока -> цільові рівні Блума
    gen: Any                  # M3, M4: id блока -> згенеровані завдання; M1, M2: список завдань для всієї лекції
    states: List[Dict]        # підсумкові стани завдань (bNN_i.json)
    summary: Dict[str, Any]


class ItemState(TypedDict, total=False):
    file: str                 # bNN_i.json
    block_text: str
    passport: str
    mode: str
    cap: int                  # ліміт ітерацій (rules.json -> refinement_loop.max_iterations)
    deadline: Optional[float]
    item: Dict[str, Any]      # {"block", "level", "status", "iterations": [{"iter", "mcq", "report"}]}


# ---------------------------------------------------------------- допоміжне
def _j(outdir, name): return os.path.join(outdir, name + ".json")

def _load(outdir, name):
    f = _j(outdir, name)
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None

def _save(outdir, name, obj):
    json.dump(obj, open(_j(outdir, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

def _over(deadline): return deadline is not None and time.time() > deadline

def _emit(log: LogFn, agent: str, message: str, **kw):
    if log:
        try:
            log({"type": "log", "agent": agent, "message": message, **kw})
        except Exception:
            pass        # збій доставки журналу не перериває конвеєр


def _targets(report):
    return report.get("target_elements") or [report.get("target_element")]


# ---------------------------------------------------------------- граф одного завдання (цикл доопрацювання)
def build_item_graph(log: LogFn = None):
    def critic(s: ItemState):
        it = s["item"]
        if it["status"] != "in_progress" or _over(s.get("deadline")):
            return {}
        cur = it["iterations"][-1]
        if "report" not in cur:
            cur["report"] = C.critique(s["block_text"], cur["mcq"], iteration=cur["iter"], passport=s.get("passport"))
        r = cur["report"]
        name = os.path.basename(s["file"])[:-5]
        if r["verdict"] == "APPROVED":
            it["status"] = "passed_at_iter_1" if cur["iter"] == 1 else f"fixed_at_iter_{cur['iter']}"
            _emit(log, "Critic", f"{name}: схвалено на ітерації {cur['iter']}", iteration=cur["iter"], item=name, verdict="APPROVED")
        else:
            _emit(log, "Critic", f"{name}: відхилено ({r.get('failed_stage') or 'етап'}), дефектні елементи: "
                                 f"{', '.join(map(str, _targets(r)))}. {r.get('defect_explanation', '')}",
                  iteration=cur["iter"], item=name, verdict="REJECT_FOR_REFINEMENT")
            if s["mode"] == "m3":
                it["status"] = "m3_rejected_by_critic"
            elif cur["iter"] >= s["cap"]:
                it["status"] = "unresolved_after_cap"
                _emit(log, "System", f"{name}: вичерпано ліміт у {s['cap']} ітерацій", item=name)
        json.dump(it, open(s["file"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return {"item": it}

    def refine(s: ItemState):
        it = s["item"]; cur = it["iterations"][-1]; r = cur["report"]
        name = os.path.basename(s["file"])[:-5]
        new = C.local_regenerate(s["block_text"], cur["mcq"], r, passport=s.get("passport"))
        if "_fix_failed" in new:            # формула 2.8: доопрацювання завершується
            it["status"] = "hard_reject"; cur["fail"] = new.pop("_fix_failed")
            _emit(log, "Generator", f"{name}: виправлення неможливе без вигадування фактів — завдання відхилено", item=name)
        else:
            it["iterations"].append({"iter": cur["iter"] + 1, "mcq": new})
            _emit(log, "Generator", f"{name}: нова версія (змінено лише {', '.join(map(str, _targets(r)))})",
                  iteration=cur["iter"] + 1, item=name)
        json.dump(it, open(s["file"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return {"item": it}

    def route(s: ItemState):
        return "refine" if s["item"]["status"] == "in_progress" and not _over(s.get("deadline")) else "end"

    g = StateGraph(ItemState)
    g.add_node("critic", critic)
    g.add_node("refine", refine)
    g.set_entry_point("critic")
    g.add_conditional_edges("critic", route, {"refine": "refine", "end": END})
    g.add_edge("refine", "critic")
    return g.compile()


# ---------------------------------------------------------------- граф лекції
def build_lecture_graph(log: LogFn = None):
    item_graph = build_item_graph(log)

    def deadline(s): return None if s.get("budget_s") is None else s["t0"] + s["budget_s"]

    def check_size(s: LectureState):
        text = read_lecture(s["path"])
        n = C.check_lecture_size(text)          # LectureTooLarge, якщо > limits.lecture_max_chars
        os.makedirs(s["outdir"], exist_ok=True)
        _emit(log, "System", f"Лекція: {n} символів; завдань: {s['n']}; режим: {s['mode'].upper()}", stage="check")
        return {"text": text}

    def analyst_node(s: LectureState):
        A = _load(s["outdir"], "analyst")
        if A is None:
            _emit(log, "Analyst", "Сегментація лекції на тематичні блоки, стеля рівня Блума, довідка…", stage="analyst")
            A = analyze_lecture(s["text"]); _save(s["outdir"], "analyst", A)
        content = [b for b in A["blocks"] if b.get("type") == "content"]
        _emit(log, "Analyst", f"Галузь: {A.get('domain')}; блоків: {len(A['blocks'])}, з них змістових: {len(content)}; "
                              f"термінів у глосарії: {len(A.get('glossary') or [])}", stage="analyst",
              blocks=[{"id": b["id"], "title": b.get("title"), "type": b.get("type"), "bloom": b.get("bloom")} for b in A["blocks"]])
        return {"analysis": A}

    def allocate_node(s: LectureState):
        AL = _load(s["outdir"], "alloc")
        if AL is None:
            AL = allocate(s["analysis"]["blocks"], s["n"]); _save(s["outdir"], "alloc", AL)
        _emit(log, "System", "Розподіл завдань і цільових рівнів Блума: " +
              "; ".join(f"блок {k}: {', '.join(map(str, v))}" for k, v in AL.items()), stage="allocate")
        return {"alloc": AL}

    def generate(s: LectureState):
        if s["mode"] in FLAT:
            return generate_flat(s)
        A, AL = s["analysis"], s["alloc"]; byid = {b["id"]: b for b in A["blocks"]}
        G = _load(s["outdir"], "gen") or {}
        dl = deadline(s)
        def one(k):
            if k in G or _over(dl): return
            b, levels = byid[int(k)], AL[k]
            items = PL.plan_m2(b["text"], levels, passport=passport(A, b))[:len(levels)]
            for it, l in zip(items, levels): it["target_level"] = l
            G[k] = items
            _emit(log, "Generator", f"Блок {k} «{b.get('title', '')}»: створено {len(items)} завд.", stage="generate", block=int(k))
        with ThreadPoolExecutor(s.get("workers", 6)) as ex: list(ex.map(one, AL.keys()))
        _save(s["outdir"], "gen", G)
        return {"gen": G}

    def generate_flat(s: LectureState):
        """M1, M2: без аналітика — уся лекція одним запитом, рівні Блума порівну (n/3 кожного)."""
        n = s["n"]; base = n // 3; levels = [1] * base + [2] * base + [3] * (n - 2 * base)
        G = _load(s["outdir"], "gen")
        if G is None:
            ctx = C.truncate_source_text(s["text"], RL.limit("context_chars"))
            G = (PL.plan_m1(ctx, levels) if s["mode"] == "m1" else PL.plan_m2(ctx, levels))[:n]
            for it, l in zip(G, levels): it["target_level"] = l
            _save(s["outdir"], "gen", G)
        _emit(log, "Generator", f"Уся лекція одним запитом: створено {len(G)} завд.", stage="generate")
        if s["mode"] != "m1":
            return {"gen": G}
        states = []                      # M1: критика немає — завдання фіксуються як згенеровані
        for i, it in enumerate(G):
            f = os.path.join(s["outdir"], f"b00_{i}.json")
            st = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else \
                {"block": 0, "level": it.get("target_level"), "status": "in_progress", "iterations": [{"iter": 1, "mcq": PL._norm_mcq(it)}]}
            st["status"] = "m1_generated"
            json.dump(st, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            states.append(st)
        return {"gen": G, "states": states}

    def verify(s: LectureState):
        flat = s["mode"] in FLAT; G = s["gen"]
        cap = RL.max_iterations(); dl = deadline(s)
        _emit(log, "Critic", f"Двоетапна перевірка; цикл доопрацювання до {cap} ітерацій" if s["mode"] in ("m2", "m4")
              else "Двоетапна перевірка (лише вимірювання, без доопрацювання)", stage="verify")
        if flat:                         # M2: уся лекція — блок 0, довідки аналітика немає
            jobs = [(0, i, it) for i, it in enumerate(G)]
            ctx = lambda k: (s["text"], None)
        else:
            A = s["analysis"]; byid = {b["id"]: b for b in A["blocks"]}
            jobs = [(int(k), i, it) for k, v in G.items() for i, it in enumerate(v)]
            ctx = lambda k: (byid[k]["text"], passport(A, byid[k]))
        def run_item(a):
            k, i, it = a; f = os.path.join(s["outdir"], f"b{k:02d}_{i}.json"); text, pp = ctx(k)
            st = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else \
                {"block": k, "level": it.get("target_level"), "status": "in_progress", "iterations": [{"iter": 1, "mcq": PL._norm_mcq(it)}]}
            out = item_graph.invoke({"file": f, "block_text": text, "passport": pp, "mode": s["mode"],
                                     "cap": cap, "deadline": dl, "item": st}, {"recursion_limit": 4 * cap + 10})
            st = out["item"]
            json.dump(st, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            return st
        with ThreadPoolExecutor(s.get("workers", 6) * 2) as ex: states = list(ex.map(run_item, jobs))
        return {"states": states}

    def summarize(s: LectureState):
        done = not _over(deadline(s)) and all(x["status"] != "in_progress" for x in s["states"])
        if s["mode"] in FLAT:
            sm = summarize_flat(s["states"], s["outdir"], s["mode"], RL.max_iterations(), done=done)
        else:
            sm = summarize_lecture(s["analysis"], s["alloc"], s["states"], s["outdir"], s["mode"], RL.max_iterations(), done=done)
        _emit(log, "System", f"Завершено: {sm['statuses']}", stage="done")
        return {"summary": sm}

    g = StateGraph(LectureState)
    for name, fn in (("check_size", check_size), ("analyst", analyst_node), ("allocate", allocate_node),
                     ("generate", generate), ("verify", verify), ("summarize", summarize)):
        g.add_node(name, fn)
    g.set_entry_point("check_size")
    # M1 і M2 пропускають analyst і allocate; M1 також пропускає verify
    g.add_conditional_edges("check_size", lambda s: "generate" if s["mode"] in FLAT else "analyst",
                            {"analyst": "analyst", "generate": "generate"})
    g.add_edge("analyst", "allocate"); g.add_edge("allocate", "generate")
    g.add_conditional_edges("generate", lambda s: "summarize" if s["mode"] == "m1" else "verify",
                            {"verify": "verify", "summarize": "summarize"})
    g.add_edge("verify", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


def run_lecture(path, outdir, n=12, mode="m4", budget_s=None, workers=6, log: LogFn = None) -> Dict[str, Any]:
    """Запуск графа лекції; повертає summary.json прогону."""
    if mode not in ("m1", "m2", "m3", "m4"):
        raise ValueError("lecture_graph: режим m1, m2, m3 або m4")
    app = build_lecture_graph(log)
    out = app.invoke({"path": path, "outdir": outdir, "n": int(n), "mode": mode, "budget_s": budget_s,
                      "t0": time.time(), "workers": workers})
    return out["summary"]


def collect(outdir) -> Dict[str, Any]:
    """Результат прогону для вебзастосунку: блоки аналітика і кожне завдання з усіма ітераціями (версія + звіт критика)."""
    A = _load(outdir, "analyst") or {}; sm = _load(outdir, "summary") or {}
    blocks = {b["id"]: b for b in A.get("blocks", [])}
    items = []
    for f in sorted(x for x in os.listdir(outdir) if x.startswith("b") and x.endswith(".json") and "_" in x):
        st = json.load(open(os.path.join(outdir, f), encoding="utf-8")); b = blocks.get(st["block"], {})
        its = []
        for it in st["iterations"]:
            r = it.get("report") or {}
            its.append({"iter": it["iter"], "mcq": it["mcq"], "verdict": r.get("verdict"), "failed_stage": r.get("failed_stage"),
                        "targets": [t for t in _targets(r) if t and t != "none"] if r else [],
                        "defect_explanation": r.get("defect_explanation", ""),
                        "instruction": r.get("actionable_refinement_instruction", "")})
        items.append({"id": f[:-5], "block": st["block"], "block_title": b.get("title"), "block_bloom": b.get("bloom"),
                      "target_level": st["level"], "status": st["status"], "final": st["iterations"][-1]["mcq"],
                      "iterations": its, "fail": next((it.get("fail") for it in st["iterations"] if it.get("fail")), None)})
    return {"summary": sm, "domain": A.get("domain"),
            "blocks": [{"id": b["id"], "title": b.get("title"), "type": b.get("type"), "bloom": b.get("bloom")} for b in A.get("blocks", [])],
            "items": items}


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    opt = lambda name, d: (sys.argv[sys.argv.index(name) + 1] if name in sys.argv else d)
    try:
        bs = float(os.environ["LECTURE_BUDGET_S"]) if os.environ.get("LECTURE_BUDGET_S") else None
        r = run_lecture(a[0], a[1], n=int(opt("--n", 12)), mode=opt("--mode", "m4"), budget_s=bs,
                        log=lambda e: print(f"[{e.get('agent')}] {e.get('message')}"))
    except C.LectureTooLarge as e:
        print("ПОМИЛКА:", e); sys.exit(2)
    print(json.dumps(r, ensure_ascii=False, indent=1))
