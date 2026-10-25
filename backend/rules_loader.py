"""Завантаження правил із rules/rules.json: параметри, шаблони, файли промптів. Єдине джерело для Критика та Арбітра."""
import json, os
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")
_PATH = os.getenv("RULES_PATH", os.path.join(_DIR, "rules.json"))
with open(_PATH, encoding="utf-8") as _f:
    RULES = json.load(_f)
_RULES_DIR = os.path.dirname(_PATH)

def prompt(name: str) -> str:
    with open(os.path.join(_RULES_DIR, RULES["prompts"][name]), encoding="utf-8", newline="") as f:
        return f.read()

def param(*path):
    v = RULES["parameters"]
    for k in path: v = v[k]
    return v

def limit(name: str):
    return RULES["limits"][name]

def template(name: str) -> dict:
    return RULES["templates"][name]

def length_limit(mean_words: float) -> int:
    """Цільова довжина ключа при скороченні."""
    r, e = param("length_parity", "ratio"), param("length_parity", "min_extra_words")
    return max(int(r * mean_words), int(mean_words) + e)

def render(name: str, which: str, **kw) -> str:
    return template(name)[which].format(**kw)

def render_forms(name: str, which: str, count: int, **kw) -> str:
    forms = template(name)["forms"]["one" if count == 1 else "many"]
    return template(name)[which].format(**forms, **kw)

def max_iterations() -> int:
    return int(RULES["refinement_loop"]["max_iterations"])

def categorical_candidates(key: str, distractors) -> str:
    """Проста лексична підказка для моделі: де в ключі/дистракторах є слова зі списку категоричних і чи стоїть перед ними «не». Вердикт (категоричне вживання чи заперечення) виносить модель."""
    import re
    rx = re.compile(r"(?<!\w)(?P<neg>не\s+)?(?P<w>" + param("categorical_words", "words") + r")(?!\w)", re.IGNORECASE)
    def find(t):
        out = {}
        for m in rx.finditer(t):
            out.setdefault(m.group("w").lower(), []).append(bool(m.group("neg")))
        return [f"«{w}»" + (" (лише після «не»)" if all(v) else "") for w, v in sorted(out.items())]
    found = []
    for i, d in enumerate(distractors, 1):
        w = find(d)
        if w: found.append(f"d{i}: " + ", ".join(w))
    wk = find(key)
    parts = found + ([("ключ: " + ", ".join(wk))] if wk else [])
    return "; ".join(parts) if parts else "кандидатів немає"

def source_reference(text: str) -> str:
    """Посилання на джерело в запитанні («згідно з навчальним текстом» тощо): повертає знайдену фразу або порожній рядок."""
    import re
    m = re.search(param("source_reference", "regex"), text or "", re.IGNORECASE)
    return m.group(0).strip() if m else ""

def format_candidates(key: str, distractors) -> str:
    """Ознаки запису (дужки, латиниця, цифри), що є лише в частини з чотирьох варіантів. Лише підказка для моделі."""
    import re
    names = ["key"] + [f"d{i}" for i in range(1, len(distractors) + 1)]
    texts = [key] + list(distractors)
    out = []
    for feat, rx in param("format_asymmetry", "features").items():
        has = [bool(re.search(rx, t)) for t in texts]
        if 0 < sum(has) < len(texts):
            out.append(f"{feat}: є в " + ", ".join(n for n, h in zip(names, has) if h) + "; немає в " + ", ".join(n for n, h in zip(names, has) if not h))
    return "; ".join(out) if out else "розбіжностей немає"

def latin_outlier(key: str, distractors) -> list:
    """Жорстке правило: запис латиницею є в усіх варіантах, крім одного (3 із 4) — це неоднорідність формату. Повертає id усіх варіантів для приведення до однакового формату або []."""
    import re
    texts = [key] + list(distractors)
    has = [bool(re.search(r"[A-Za-z]{2,}", t)) for t in texts]
    if len(texts) == 4 and sum(has) == param("format_asymmetry", "latin_outlier", "min_with_latin"):
        return ["key", "d1", "d2", "d3"]
    return []

def format_defect(key: str, distractors) -> str:
    """Жорсткі правила однорідності формату (без судження моделі): (а) латиниця є в 3 з 4 варіантів; (б) пояснення в дужках
    (дужка після пробілу: не виклик функції) є в 1–3 з 4 варіантів. Повертає опис порушення або порожній рядок."""
    import re
    texts = [key] + list(distractors)
    if len(texts) != 4: return ""
    lat = sum(bool(re.search(r"[A-Za-z]{2,}", t)) for t in texts)
    par = sum(bool(re.search(r"\s\([^)]+\)", t)) for t in texts)
    if lat == 3: return "запис латиницею є не в усіх варіантах (відсутній в одному з чотирьох)"
    if 0 < par < 4: return f"пояснення в дужках є лише в {par} варіантах із чотирьох"
    return ""
