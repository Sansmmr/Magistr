"""
Незалежний LLM-Арбітр (LLM-as-a-Judge) за регламентом ExplAI-2026, розділ 5.

Відмінності від Критика:
  * не є частиною пайплайна: оцінює ГОТОВІ завдання всіх конфігурацій M1-M4 однаково;
  * temperature = 0, сліпа оцінка (не знає, яка конфігурація згенерувала завдання);
  * нічого не виправляє, повертає лише бінарні бали;
  * рівень Блума визначає окремим сліпим викликом (без цільового рівня) -> метрика H1,
    у 4/4 Acceptance Rate НЕ входить.
Підсумки (total_score, is_accepted) обчислюються в коді, а не беруться з відповіді моделі.
"""
import json, os, sys
from typing import Dict, List
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from text_utils import truncate_source_text
import rules_loader as RL

ARBITER_MODEL = os.getenv("ARBITER_MODEL", Config.VERIFICATION_MODEL)
ARBITER_BASE_URL = os.getenv("ARBITER_BASE_URL", Config.VERIFICATION_BASE_URL)

# Дослівно за регламентом (розділ 5)
ARBITER_SYSTEM_PROMPT = RL.prompt("arbiter_system")



# --- Додаткові правила (відхилення від регламенту, узгоджено з викладачем): регламентний промпт
# сам по собі пропускав абсурдні дистрактори і підказки за довжиною (калібрування 2026-10-06). ---
ARBITER_ADDENDUM = RL.prompt("arbiter_addendum")
BLIND_SYSTEM = RL.prompt("arbiter_blind_system") + RL.prompt("arbiter_blind_addendum")
SOLVER_SYSTEM = RL.prompt("arbiter_solver")
# Режим контексту арбітра: "none" (за замовчуванням, без тексту лекції + незалежне розв'язання) або "block" (з текстом блоку)
ARBITER_CONTEXT = os.getenv("ARBITER_CONTEXT", "none").lower()

# Сліпа класифікація рівня Блума (метрика H1). Операціоналізація рівнів — з регламенту (п. 3.1).
BLOOM_SYSTEM_PROMPT = """Ти — експерт з таксономії Блума-Андерсона. Визнач, на якому рівні когнітивної складності
знаходиться тестове завдання (ти НЕ знаєш, який рівень задумувався автором).
Рівні:
- 1 (Запам'ятовування): достатньо впізнати або дослівно відтворити термін, назву, факт чи визначення з тексту.
- 2 (Розуміння): потрібно пояснити суть своїми словами, порівняти поняття, класифікувати або обрати причину; відповідь не є дослівною цитатою.
- 3 (Застосування): потрібно розрахувати результат або передбачити поведінку системи у НОВОМУ практичному сценарії / фрагменті коду, якого немає дослівно в тексті.
Якщо завдання формально звучить складніше, але відповідь можна знайти як готове формулювання в тексті — це рівень 1.
Поверни ВИКЛЮЧНО JSON: {"bloom_level": 1 | 2 | 3, "justification": "одне речення"}"""

_client = None
def _cli():
    global _client
    if _client is None:
        _client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=ARBITER_BASE_URL)
    return _client

def _item_block(context: str, question: str, key: str, distractors: List[str]) -> str:
    d = "\n".join(f"D{i+1}: {t}" for i, t in enumerate(distractors))
    if context is None:
        return f"Тестове завдання MCQ:\nЗапитання: {question}\nПравильна відповідь (key): {key}\nДистрактори:\n{d}\n"
    return (f'Навчальний текст (контекст C):\n"""\n{truncate_source_text(context, RL.limit("context_chars"))}\n"""\n\n'
            f"Тестове завдання MCQ:\nЗапитання: {question}\nПравильна відповідь (key): {key}\n"
            f"Дистрактори:\n{d}\n")

def _is_claude() -> bool:
    return ARBITER_MODEL.lower().startswith("claude")

def _json_from_text(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.index("{"):]
    return json.loads(text[text.index("{"): text.rindex("}") + 1])

BLOOM_MODEL = os.getenv("BLOOM_MODEL", "claude-haiku-4-5")      # Блум-класифікація (H1) на дешевшій моделі
PRICES = {"haiku": (1.0, 5.0), "sonnet-5": (2.0, 10.0), "sonnet-4": (3.0, 15.0)}
USAGE = {"usd": 0.0, "in": 0, "cache_write": 0, "cache_read": 0, "out": 0, "calls": 0}
import threading as _th
_ULOCK = _th.Lock()

def usage_cost() -> float:
    return USAGE["usd"]

def _chat_claude(system: str, user: str, prefix: str = None, model: str = None) -> Dict:
    model = model or ARBITER_MODEL
    """Арбітр на Claude (Anthropic Messages API, ANTHROPIC_API_KEY у .env); без залежності від пакета anthropic."""
    import time, urllib.request, urllib.error
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY не задано в .env")
    import re as _re
    # prompt caching: статичний system та (за наявності) спільний префікс користувача (текст лекції) кешуються на 5 хв
    sysb = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    ucontent = [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": user}] if prefix else user
    pl = {"model": model, "max_tokens": 2000, "system": sysb, "messages": [{"role": "user", "content": ucontent}]}
    if not _re.match(r"claude-(sonnet|opus|haiku)-5", model): pl["temperature"] = 0.0     # у моделей 5.x параметр temperature застарів (API повертає 400)
    body = json.dumps(pl).encode()
    for attempt in range(8):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=120))
            u = d.get("usage", {})
            with _ULOCK:
                USAGE["in"] += u.get("input_tokens", 0); USAGE["out"] += u.get("output_tokens", 0); USAGE["calls"] += 1
                pi, po = PRICES["haiku"] if "haiku" in model else PRICES["sonnet-5"] if "-5" in model else PRICES["sonnet-4"]
                USAGE["usd"] += (u.get("input_tokens", 0) * pi + u.get("cache_creation_input_tokens", 0) * pi * 1.25 + u.get("cache_read_input_tokens", 0) * pi * 0.1 + u.get("output_tokens", 0) * po) / 1e6
                USAGE["cache_write"] += u.get("cache_creation_input_tokens", 0); USAGE["cache_read"] += u.get("cache_read_input_tokens", 0)
            return _json_from_text("".join(b.get("text", "") for b in d["content"]))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(6 + 4 * attempt); continue
            raise RuntimeError(f"Arbiter (Claude): HTTP {e.code}: {e.read()[:200]!r}")
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(4 + 3 * attempt)
    raise RuntimeError("Arbiter (Claude): retries exhausted")

def _chat(system: str, user: str, prefix: str = None, model: str = None) -> Dict:
    if _is_claude():
        return _chat_claude(system, user, prefix, model)
    if prefix: user = prefix + user
    import time
    from openai import RateLimitError
    for attempt in range(8):
        try:
            r = _cli().chat.completions.create(
                model=ARBITER_MODEL, temperature=0.0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"})
            return json.loads(r.choices[0].message.content)
        except RateLimitError:
            time.sleep(6 + 4 * attempt)  # ліміт TPM акаунта: чекаємо й повторюємо
    raise RuntimeError("Arbiter: rate limit retries exhausted")

def _bin(v) -> int:
    try:
        return 1 if int(v) == 1 else 0
    except (TypeError, ValueError):
        return 0

import re, statistics
SHORTCUT_PHRASES = re.compile(RL.param("shortcut_phrases", "regex"), re.IGNORECASE)

def word_count(text: str) -> int:
    """Кількість слів: токени без літер і цифр (тире «—», «-», «/» тощо) за слова не рахуються."""
    return sum(1 for t in text.split() if re.search(r"\w", t))

def code_shortcut_checks(key: str, distractors: List[str]) -> Dict:
    """Детермінована частина No_Shortcuts (регламент: паритет довжини ±15%, без «все вищезазначене»)."""
    lk = word_count(key)
    ld = [word_count(d) for d in distractors]
    mean_d = statistics.mean(ld) if ld else 0
    all_len = [lk] + ld
    cv = (statistics.pstdev(all_len) / statistics.mean(all_len)) if statistics.mean(all_len) else 0.0
    # ключ є «виділеним» за довжиною, якщо перевищує середню довжину дистракторів більш ніж на 15%
    # і більш ніж на 2 слова (різницю до 2 слів вважаємо допустимою)
    words_flag = lk > RL.param("length_parity", "ratio") * mean_d and (lk - mean_d) > RL.param("length_parity", "min_extra_words")
    # Довжину людина сприймає візуально: правило за словами спрацьовує лише разом із правилом за символами
    kc, dc = len(key.strip()), [len(d.strip()) for d in distractors]
    mean_c = statistics.mean(dc) if dc else 0
    chars_flag = bool(dc) and kc > max(dc) and kc > RL.param("length_parity", "ratio_chars") * mean_c \
        and (kc - mean_c) > RL.param("length_parity", "min_extra_chars")
    longer = words_flag and chars_flag
    phrase = any(SHORTCUT_PHRASES.search(t) for t in [key] + list(distractors))
    return {"key_words": lk, "distractor_mean_words": round(mean_d, 2), "cv_length": round(cv, 3),
            "key_too_long": bool(longer), "words_flag": bool(words_flag), "chars_flag": bool(chars_flag), "key_chars": kc, "distractor_mean_chars": round(mean_c, 1), "shortcut_phrase": bool(phrase),
            "code_no_shortcuts": int(not longer and not phrase)}

def solve_blind(question: str, key: str, distractors: List[str]) -> Dict:
    """Незалежне розв'язання без ключа й без тексту: варіанти перемішуються детермінованим чином."""
    import hashlib, random
    opts = [("key", key)] + [(f"D{i+1}", t) for i, t in enumerate(distractors)]
    random.Random(int(hashlib.md5(question.encode()).hexdigest()[:8], 16)).shuffle(opts)
    letters = "ABCD"
    body = f"Запитання: {question}\n" + "\n".join(f"{letters[i]}. {t}" for i, (_, t) in enumerate(opts))
    raw = _chat(SOLVER_SYSTEM, body)
    ch = str(raw.get("choice", "")).strip().upper()[:1]
    picked = opts[letters.index(ch)][0] if ch in letters else None
    return {"picked": picked, "agrees_with_key": picked == "key", "multiple_correct": bool(raw.get("multiple_correct")),
            "none_correct": bool(raw.get("none_correct")), "needs_source": bool(raw.get("needs_source")), "reason": str(raw.get("reason", ""))}

def judge_item(context: str, question: str, key: str, distractors: List[str]) -> Dict:
    """Чотири бінарні бали. LLM + детерміновані перевірки в коді; підсумок рахується в коді."""
    code = code_shortcut_checks(key, distractors)
    metrics = (f"\nМетрики довжини (обчислені автоматично): довжина ключа = {code['key_words']} слів; середня довжина дистракторів = "
               f"{code['distractor_mean_words']} слів; ключ помітно довший за дистрактори = {'так' if code['key_too_long'] else 'ні'}; "
               f"фраза-підказка = {'так' if code['shortcut_phrase'] else 'ні'}. "
               f"Кандидати у категоричні слова (знайдені автоматично за списком, лише підказка; вердикт виносиш ти): {RL.categorical_candidates(key, distractors)}. "
               f"Кандидати на неоднорідність формату варіантів (ознаки запису лише в частини варіантів; лише підказка, вердикт виносиш ти): {RL.format_candidates(key, distractors)}.\n")
    blind = ARBITER_CONTEXT == "none"
    solver = None
    if blind:
        solver = solve_blind(question, key, distractors)
        sv = (f"\nНезалежне розв'язання (експерт не бачив ключа): обрав {'ключ' if solver['agrees_with_key'] else 'дистрактор ' + str(solver['picked'])}; "
              f"кілька правильних = {'так' if solver['multiple_correct'] else 'ні'}; жодного правильного = {'так' if solver['none_correct'] else 'ні'}; "
              f"потрібне джерело = {'так' if solver['needs_source'] else 'ні'}; обґрунтування: {solver['reason']}\n")
        raw = _chat(BLIND_SYSTEM, _item_block(None, question, key, distractors) + sv + metrics + "\nОціни завдання та поверни JSON.")
    else:
        raw = _chat(ARBITER_SYSTEM_PROMPT + ARBITER_ADDENDUM,
                    _item_block(context, question, key, distractors) + metrics + "\nОціни завдання та поверни JSON.")
    f, a, p, s = (_bin(raw.get(k)) for k in
                  ("factuality_score", "ambiguity_score", "plausibility_score", "no_shortcuts_score"))
    checks = raw.get("distractor_checks") or []
    llm_p = p
    if len(checks) < len(distractors) or any(c.get("same_domain") is not True for c in checks):
        p = 0
    if raw.get("terminology_ok") is False:
        f = 0
    wi = [w for w in (raw.get("wording_issues") or []) if isinstance(w, dict) and str(w.get("element", "")).lower() in ("question", "key", "d1", "d2", "d3", "options")]
    wi = [w for w in wi if not (str(w.get("element", "")).lower() == "options" and not str(w.get("common_part") or "").strip())]   # «options» без названої спільної частини — не дефект
    key_vague = raw.get("key_precise") is False
    if wi or key_vague:
        a = 0
    cat_ids = [str(x).upper() for x in (raw.get("categorical_distractors") or []) if str(x).upper() in ("D1", "D2", "D3")]
    if len(cat_ids) < RL.param("categorical_words", "min_distractors"): cat_ids = []   # правило: сигнал лише від ≥N дистракторів
    if cat_ids: s = 0
    src_phrase = (str(raw.get("source_phrase") or "").strip() if raw.get("source_reference") is True else "") or RL.source_reference(question)
    if src_phrase: a = 0
    fmt_ids = [str(x).upper() for x in (raw.get("format_mismatch") or []) if str(x).upper() in ("KEY", "D1", "D2", "D3")] \
        if RL.format_candidates(key, distractors) != "розбіжностей немає" else []     # вердикт моделі лише за наявності кандидата від коду
    if RL.format_defect(key, distractors): fmt_ids = ["KEY", "D1", "D2", "D3"]     # латиниця в 3 з 4 варіантів — дефект без судження моделі
    if fmt_ids: s = 0
    key_rep = raw.get("key_repeats_question") is True
    if key_rep: s = 0
    s = min(s, code["code_no_shortcuts"])
    total = f + a + p + s
    # Підсумковий коментар узгоджується з остаточними балами (коментар LLM пишеться до корекції кодом)
    reasons = []
    if f == 0: reasons.append("Factuality=0" + (" (термінологічна невідповідність)" if raw.get("terminology_ok") is False else ""))
    if a == 0:
        why_a = []
        if wi: why_a.append("мовні помилки: " + "; ".join(f'{w.get("element")}: {w.get("problem", "")}' for w in wi))
        if key_vague: why_a.append("ключ розпливчастий")
        if src_phrase: why_a.append("посилання на джерело: «" + src_phrase + "»")
        reasons.append("No_Ambiguity=0" + (" (" + "; ".join(why_a) + ")" if why_a else ""))
    if p == 0:
        bad = [str(c.get("id", "")) for c in checks if c.get("same_domain") is not True]
        reasons.append("Plausibility=0 (дистрактори не з тієї ж області: " + (", ".join(bad) or "неповна перевірка") + ")")
    if s == 0:
        why = []
        if code["key_too_long"]: why.append(f"ключ {code['key_words']} слів проти {code['distractor_mean_words']} у дистракторів")
        if cat_ids: why.append("категоричні слова в дистракторах " + ", ".join(cat_ids) + " (оцінка моделі)")
        if code["shortcut_phrase"]: why.append("фраза-підказка")
        if fmt_ids: why.append("неоднорідний формат варіантів: " + ", ".join(fmt_ids))
        if key_rep: why.append("ключ повторює запитання: " + str(raw.get("repeated_phrase", "")))
        reasons.append("No_Shortcuts=0" + (" (" + "; ".join(why) + ")" if why else ""))
    comment = str(raw.get("expert_comment", ""))
    if reasons: comment = "Знижено: " + "; ".join(reasons) + ". Коментар моделі: " + comment
    return {"factuality": f, "no_ambiguity": a, "plausibility": p, "no_shortcuts": s,
            "total_score": total, "is_accepted": total == 4,
            "llm_plausibility_raw": llm_p, "distractor_checks": checks,
            "terminology_ok": raw.get("terminology_ok"), "wording_issues": wi, "key_precise": raw.get("key_precise"), "categorical_distractors": cat_ids, "key_repeats_question": key_rep, "source_reference": src_phrase, "format_mismatch": fmt_ids, "context_mode": ARBITER_CONTEXT, "blind_solver": solver, "unverifiable_without_source": raw.get("unverifiable_without_source"),
            **code, "expert_comment": comment}

def _ctx_prefix(context: str) -> str:
    return f'Навчальний текст (контекст C):\n"""\n{truncate_source_text(context, RL.limit("context_chars"))}\n"""\n\n'

def classify_bloom(context: str, question: str, key: str, distractors: List[str], share_ctx: bool = False) -> Dict:
    """share_ctx=True: текст лекції спільний для всіх питань (M1/M2) і кешується; текст промпта не змінюється."""
    if share_ctx and context is not None:
        full = _item_block(context, question, key, distractors); pre = _ctx_prefix(context)
        assert full.startswith(pre)
        raw = _chat(BLOOM_SYSTEM_PROMPT, full[len(pre):] + "\nВизнач рівень Блума цього завдання.", prefix=pre, model=BLOOM_MODEL)
    else:
        raw = _chat(BLOOM_SYSTEM_PROMPT, _item_block(context, question, key, distractors)
                    + "\nВизнач рівень Блума цього завдання.", model=BLOOM_MODEL)
    try:
        lvl = int(raw.get("bloom_level"))
    except (TypeError, ValueError):
        lvl = 0
    return {"bloom_model": BLOOM_MODEL, "bloom_by_judge": lvl, "bloom_justification": str(raw.get("justification", ""))}

def evaluate(context, question, key, distractors, target_level=None) -> Dict:
    out = judge_item(context, question, key, distractors)
    out.update(classify_bloom(context, question, key, distractors))
    if target_level is not None:
        out["bloom_match"] = int(out["bloom_by_judge"] == int(target_level))
    return out
