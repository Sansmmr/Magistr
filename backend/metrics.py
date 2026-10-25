# -*- coding: utf-8 -*-
"""
Метрики якості дистракторів і довжини варіантів (регламент, розділ 7; гіпотеза H2).

 - cv_words, cv_chars — коефіцієнт варіації довжини чотирьох варіантів (ключ + 3 дистрактори) у словах і в символах;
 - mean_cosine, corridor_ratio_pct — косинусна подібність дистракторів до ключа (sentence-transformers) і частка
   дистракторів у коридорі [low, high] (rules.json → parameters.semantic_corridor);
 - key_longest — чи ключ є найдовшим варіантом (за символами).
Використання: python metrics.py <каталог_результатів> [<каталог2> ...]  — рахує за фінальними MCQ кожного завдання
(останньою ітерацією b*_*.json) і записує metrics.json та metrics_items.csv у каталог.
"""
import csv, glob, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules_loader as RL
from arbiter import word_count

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(RL.param("semantic_corridor", "model"))
    return _MODEL


def cv(values):
    """Коефіцієнт варіації = σ / середнє (генеральне стандартне відхилення)."""
    m = statistics.mean(values)
    return statistics.pstdev(values) / m if m else 0.0


def item_metrics(mcq: dict, with_embeddings: bool = True) -> dict:
    key = mcq["key"]; ds = [d["text"] for d in mcq["distractors"]]
    words = [word_count(t) for t in [key] + ds]
    chars = [len(t.strip()) for t in [key] + ds]
    out = {"cv_words": cv(words), "cv_chars": cv(chars), "key_chars": chars[0],
           "distractor_mean_chars": statistics.mean(chars[1:]), "key_longest": chars[0] > max(chars[1:])}
    if with_embeddings:
        import numpy as np
        e = _model().encode([key] + ds, normalize_embeddings=True)
        sims = [float(np.dot(e[0], v)) for v in e[1:]]
        lo, hi = RL.param("semantic_corridor", "low"), RL.param("semantic_corridor", "high")
        out.update(cosines=sims, mean_cosine=statistics.mean(sims), in_corridor=[lo <= s <= hi for s in sims])
    return out


def aggregate(items: list) -> dict:
    n = len(items); res = {"n_items": n,
        "cv_words_mean": statistics.mean(i["cv_words"] for i in items),
        "cv_chars_mean": statistics.mean(i["cv_chars"] for i in items),
        "key_longest_pct": 100.0 * sum(i["key_longest"] for i in items) / n}
    if all("cosines" in i for i in items):
        allsims = [s for i in items for s in i["cosines"]]
        res.update(mean_cosine=statistics.mean(allsims), cosine_sd=statistics.pstdev(allsims),
                   corridor_ratio_pct=100.0 * sum(c for i in items for c in i["in_corridor"]) / len(allsims),
                   n_distractors=len(allsims))
    return res


def final_mcqs(outdir: str) -> list:
    res = []
    for f in sorted(glob.glob(os.path.join(outdir, "b*_*.json"))):
        s = json.load(open(f, encoding="utf-8"))
        res.append({"id": os.path.basename(f)[:-5], "status": s["status"], "level": s.get("level"), "mcq": s["iterations"][-1]["mcq"]})
    return res


def run(outdir: str, with_embeddings: bool = True) -> dict:
    rows = final_mcqs(outdir)
    items = []
    for r in rows:
        m = item_metrics(r["mcq"], with_embeddings); m.update(id=r["id"], status=r["status"], level=r["level"]); items.append(m)
    agg = aggregate(items)
    for st in sorted({i["status"] for i in items}):
        pass
    json.dump({"aggregate": agg, "items": items}, open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cols = ["id", "level", "status", "cv_words", "cv_chars", "key_longest", "mean_cosine", "n_in_corridor"]
    with open(os.path.join(outdir, "metrics_items.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for i in items:
            w.writerow([i["id"], i["level"], i["status"], round(i["cv_words"], 3), round(i["cv_chars"], 3), i["key_longest"],
                        round(i.get("mean_cosine", float("nan")), 3), sum(i.get("in_corridor", []))])
    return agg


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d, json.dumps(run(d), ensure_ascii=False, indent=1))
