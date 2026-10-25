"""
Агент-Критик та цільова локальна регенерація за регламентом ExplAI-2026 (розділи 3.3 і 3.4).

Критик — частина системи (M2, M4): двоетапна верифікація + XAI-звіт з target_element.
Незалежний арбітр (arbiter.py) — окремо, інша модель; Критик працює на базовій моделі агентів.
Формат завдання (MCQ): {"question": str, "key": str, "distractors": [{"id","text","misconception_modeled"}] x3}
"""
import json, os, sys
from typing import Dict, List, Optional
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from text_utils import truncate_source_text
import rules_loader as RL
from arbiter import code_shortcut_checks, word_count   # детермінована частина No-Shortcuts (паритет довжини)

CRITIC_SYSTEM_PROMPT = """Ти — суворий Агент-Критик дидактичних тестових завдань вищої школи.
Твоє завдання — перевірити згенероване тестове завдання MCQ за двоетапною рубрикою відносно наданого навчального тексту (контексту C):
ЕТАП 1 (Hard Filtering — Фактологічна валідність):
- Factuality: чи є правильна відповідь (key) абсолютно істинною згідно з текстом? (Так/Ні)
- Ambiguity: чи є всі три дистрактори абсолютно хибними? Чи немає серед них другого правильного варіанта або неоднозначності? (Так/Ні)
* Якщо є хоч одна фактологічна помилка — статус REJECT_FOR_REFINEMENT.
ЕТАП 2 (Soft Filtering — Дидактична якість та евристики вгадування):
- Plausibility: для КОЖНОГО дистрактора окремо визнач same_domain щодо НАВЧАЛЬНОГО ТЕКСТУ: true, якщо дистрактор є поняттям, пов'язаним із предметом, про який ідеться в тексті, і його могли б розглядати поруч із ключем у тому самому розділі; студент, який знає тему лише частково, міг би сплутати його з ключем (хибне, спрощене або менш оптимальне твердження про те саме поняття теж є same_domain=true). false — якщо дистрактор є поняттям з іншої галузі, не пов'язаної з темою тексту, або очевидною нісенітницею. Не вимагай «ідеальної конкурентності». Якщо є дистрактор з same_domain=false — plausibility_valid=false, target_element — цей дистрактор.
- No-Shortcuts (Паритет варіантів): чи немає поверхневих підказок? Чи правильна відповідь не виділяється більшою довжиною, складнішим синтаксисом або специфічними уточненнями?
Якщо завдання ідеальне — статус APPROVED.
Якщо є дефект — статус REJECT_FOR_REFINEMENT, вказати конкретний цільовий елемент (наприклад, "d2" або "key") та надати чітку інструкцію, як саме його виправити.
Поверни результат ВИКЛЮЧНО у валідному форматі JSON:
{
"verdict": "APPROVED" | "REJECT_FOR_REFINEMENT",
"factuality_valid": true | false,
"no_ambiguity": true | false,
"plausibility_valid": true | false,
"no_shortcuts": true | false,
"target_element": "d1" | "d2" | "d3" | "key" | "question" | "none",
"defect_explanation": "Чітке пояснення причини браку (якщо є)",
"actionable_refinement_instruction": "Точна вказівка Генератору, яку помилку змоделювати або як змінити довжину/формулювання"
}"""

# Додатково до регламенту (узгоджено з викладачем): Критик отримує обчислені в коді метрики паритету
# і уточнення про термінологію та дистрактори — у калібруванні промпт без них пропускав очевидні дефекти.
CRITIC_ADDENDUM = """
Додаткові правила перевірки:
- Plausibility: для КОЖНОГО дистрактора окремо визнач same_domain щодо НАВЧАЛЬНОГО ТЕКСТУ: true, якщо дистрактор є поняттям, пов'язаним із предметом, про який ідеться в тексті, і його могли б розглядати поруч із ключем у тому самому розділі; студент, який знає тему лише частково, міг би сплутати його з ключем (хибне, спрощене або менш оптимальне твердження про те саме поняття теж є same_domain=true). false — якщо дистрактор є поняттям з іншої галузі, не пов'язаної з темою тексту, або очевидною нісенітницею. Не вимагай «ідеальної конкурентності». Якщо є дистрактор з same_domain=false — plausibility_valid=false, target_element — цей дистрактор.
- Терміни: якщо запитання називає ключ «функцією/оператором/командою», а ключ — об'єкт іншого типу, factuality_valid=false, target_element="question".
- Ambiguity: перевір, що кожен дистрактор хибний саме в контексті ЦЬОГО запитання. Якщо дистрактор також правильний (повністю або частково) — no_ambiguity=false, target_element — цей дистрактор.
- Factuality означає ЛИШЕ одне: ключ не суперечить тексту й підтверджується ним. Неповнота ключа, відсутність деталей, інше формулювання чи те, що в тексті є ще правильні твердження, — це НЕ порушення Factuality. Для цього дай відповіді на два вузьких запитання: key_supported_by_text (чи є в тексті твердження, яке підтверджує ключ, хоча б частково або в іншому формулюванні) і key_contradicts_text (чи є в тексті твердження, яке суперечить ключу). Якщо у запитанні є слова «єдина», «обов'язкове», «усі» і текст містить кілька правильних тверджень — це дефект запитання (no_ambiguity=false, target_element="question"), а не ключа.
- Вказівка щодо виправлення НЕ може пропонувати поточний варіант без змін: якщо ти не можеш назвати конкретну хибність, став відповідний прапорець true.
- Якщо в запитанні, ключі чи дистракторі явно хибне твердження — вкажи його як дефект.
- Пріоритет target_element: спершу дефекти Етапу 1; вказуй ОДИН найсерйозніший елемент.
Додай у JSON поля:
"evidence_quote": "цитата з тексту, що підтверджує ключ (або порожньо, якщо тексту немає)",
"key_supported_by_text": true | false,
"key_contradicts_text": true | false,
"distractor_checks": [{"id": "d1", "same_domain": true|false}, ... для d1, d2, d3]
"""

GENERATOR_SYSTEM_PROMPT = RL.prompt("generator_system")

_PARITY_ADDENDUM_FULL = RL.prompt("generator_parity_addendum")

GENERATOR_PARITY_ADDENDUM = _PARITY_ADDENDUM_FULL if os.getenv("GENERATOR_PARITY_ADDENDUM", "1") == "1" else ""   # за замовчуванням увімкнено для Генераторів M2-M4 (GENERATOR_PARITY_ADDENDUM=0 вимикає)

REFINE_USER_PROMPT = RL.prompt("refine_user")
GENERATOR_REVISION_ADDENDUM = RL.prompt("generator_revision")

def _client(base: Optional[str] = None) -> OpenAI:
    base = base or Config.GENERATOR_BASE_URL
    key = Config.DEEPSEEK_API_KEY if "deepseek" in base else Config.OPENAI_API_KEY
    return OpenAI(api_key=key, base_url=base)

# Етап 1 Критика (Factuality + Ambiguity) виконує модель з міркуванням; решта агентів — базова модель (узгоджено з викладачем).
STAGE1_MODEL = os.getenv("CRITIC_STAGE1_MODEL", "deepseek-reasoner")
STAGE1_BASE_URL = os.getenv("CRITIC_STAGE1_BASE_URL", "https://api.deepseek.com")

def _chat_claude(system: str, msgs: list, temperature: float, model: str) -> Dict:
    """Виклик Claude (Anthropic Messages API, ANTHROPIC_API_KEY з .env) для ролей, налаштованих на модель «claude-…»."""
    import time, urllib.request, urllib.error
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY не задано в .env")
    import re as _re
    pl = {"model": model, "max_tokens": 8000, "system": system, "messages": msgs}
    if not _re.match(r"claude-(sonnet|opus|haiku)-5", model): pl["temperature"] = temperature     # у моделей 5.x параметр temperature застарів (API повертає 400)
    body = json.dumps(pl).encode()
    for attempt in range(8):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            text = "".join(b.get("text", "") for b in d["content"]).strip()
            return json.loads(text[text.index("{"): text.rindex("}") + 1])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(6 + 4 * attempt); continue
            raise RuntimeError(f"Claude: HTTP {e.code}: {e.read()[:200]!r}")
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(4 + 3 * attempt)
    raise RuntimeError("Claude: retries exhausted")

def analyst_chat(system: str, user, temperature: float = 0.0) -> Dict:
    """Виклик Агента-Аналітика: модель і адреса з Config.ANALYST_MODEL / ANALYST_BASE_URL (claude-… → Anthropic)."""
    return _chat(system, user, temperature, model=Config.ANALYST_MODEL, base=Config.ANALYST_BASE_URL)

def _chat(system: str, user: str, temperature: float = 0.3, model: Optional[str] = None, base: Optional[str] = None) -> Dict:
    model = model or Config.GENERATOR_MODEL
    msgs = user if isinstance(user, list) else [{"role": "user", "content": user}]     # user: рядок або історія діалогу
    if model.lower().startswith("claude"):
        return _chat_claude(system, msgs, temperature, model)
    cli = _client(base)
    kw = {"model": model, "messages": [{"role": "system", "content": system}] + msgs}
    reasoner = "reasoner" in model
    if not reasoner:                      # моделі з міркуванням не підтримують temperature / json_object
        kw["temperature"] = temperature
        kw["response_format"] = {"type": "json_object"}
    text = cli.chat.completions.create(**kw).choices[0].message.content
    if reasoner:
        text = text[text.index("{"): text.rindex("}") + 1]
    return json.loads(text)

def mcq_from_options(question: str, options: Dict[str, str], correct: str = "A") -> Dict:
    ds = [v for k, v in options.items() if k != correct]
    return {"question": question, "key": options[correct],
            "distractors": [{"id": f"d{i+1}", "text": t, "misconception_modeled": ""} for i, t in enumerate(ds[:3])]}

def _texts(mcq): return mcq["key"], [d["text"] for d in mcq["distractors"]]

STAGE1_SYSTEM = RL.prompt("critic_stage1")

STAGE2_SYSTEM = RL.prompt("critic_stage2")


def redundancy_instruction(common, in_question):
    """Шаблон вказівки (rules.json); спільну частину і те, чи є вона в запитанні, визначає модель-Критик (Етап 1)."""
    c = (common or "«спільна частина»").strip()
    t = "redundancy_in_question" if in_question else "redundancy_not_in_question"
    return RL.render(t, "explanation", common_part=c), RL.render(t, "instruction", common_part=c)

def length_instruction(m):
    lim = RL.length_limit(m["distractor_mean_words"])
    kw = dict(key_words=m["key_words"], distractor_mean_words=m["distractor_mean_words"], limit=lim)
    return RL.render("length", "explanation", **kw), RL.render("length", "instruction", **kw)

def absolute_instruction(targets):
    names = ", ".join(targets)
    return RL.template("categorical")["explanation"], RL.render_forms("categorical", "instruction", len(targets), names=names)

def plausibility_instruction(targets):
    names = ", ".join(targets)
    return RL.render_forms("plausibility", "explanation", len(targets), names=names), RL.render_forms("plausibility", "instruction", len(targets), names=names)

def _multi(report, target, targets):
    """Нормалізує ціль: одна → рядок; кілька дистракторів → 'd1, d2, d3' + список target_elements."""
    if targets and len(targets) > 1:
        return ", ".join(targets), list(targets)
    return (targets[0] if targets else target), None

def _report(iteration, flags, target, expl, instr, m, extra=None):
    approved = all(flags.values())
    failed = None if approved else ("Stage_1_Hard_Filtering" if not (flags["factuality_valid"] and flags["no_ambiguity"]) else "Stage_2_Soft_Filtering")
    r = {"verdict": "APPROVED" if approved else "REJECT_FOR_REFINEMENT", "iteration": iteration, "failed_stage": failed, **flags,
         "target_element": "none" if approved else target, "defect_explanation": "" if approved else expl,
         "actionable_refinement_instruction": "" if approved else instr,
         "metrics": {"key_length_words": m["key_words"], "distractor_mean_words": m["distractor_mean_words"], "cv_length": m["cv_length"]}}
    r.update(extra or {}); return r

def critique(context: Optional[str], mcq: Dict, target_level=None, iteration: int = 1, passport: Optional[str] = None) -> Dict:
    """Двоетапна верифікація. Етап 1 — Factuality + Ambiguity (модель з міркуванням); якщо провалено,
    завдання бракується одразу і до Етапу 2 не допускається (регламент п. 3.3)."""
    key, ds = _texts(mcq)
    m = code_shortcut_checks(key, ds)
    ctx = truncate_source_text(context, RL.limit("context_chars")) if context else None
    ctx_block = f'Навчальний текст (контекст C):\n"""\n{ctx}\n"""' if ctx else "Навчальний текст не надано; оцінюй за загальновідомими фактами."
    if passport: ctx_block += f"\n\nДовідка Аналітика (підказка; авторитетом є навчальний текст):\n{passport}"
    mcq_json = json.dumps(mcq, ensure_ascii=False, indent=1)
    # ---------- Етап 1 ----------
    s1 = _chat(STAGE1_SYSTEM, f"{ctx_block}\n\nТестове завдання MCQ:\n{mcq_json}\nПровеcти Етап 1 та повернути JSON.",
               model=STAGE1_MODEL, base=STAGE1_BASE_URL)
    supported = s1.get("key_supported_by_text") is True
    contradicts = s1.get("key_contradicts_text") is True
    fact = (supported and not contradicts) if ctx else (not contradicts)
    term = s1.get("terminology_mismatch") is True
    bad_d = [str(c.get("id", "")).lower() for c in (s1.get("distractor_checks") or []) if c.get("is_false") is not True]
    several = s1.get("question_allows_several_correct") is True
    wi = [w for w in (s1.get("wording_issues") or []) if isinstance(w, dict) and str(w.get("element", "")).lower() in ("question", "key", "d1", "d2", "d3", "options")]
    wi = [w for w in wi if not (str(w.get("element", "")).lower() == "options" and not any(str(w.get(k) or "").strip() for k in ("common_start", "common_end", "common_part")))]   # «options» без названої спільної частини — не дефект
    key_vague = s1.get("key_precise") is False
    key_in_q = s1.get("key_repeats_question") is True
    src_phrase = (str(s1.get("source_phrase") or "").strip() if s1.get("source_reference") is True else "") or RL.source_reference(mcq.get("question", ""))
    src_ref = bool(src_phrase)
    rep_phrase = str(s1.get("repeated_phrase") or "").strip()
    flags = {"factuality_valid": bool(fact and not term), "no_ambiguity": bool(not bad_d and not several and not wi and not key_vague and not key_in_q and not src_ref),
             "plausibility_valid": True, "no_shortcuts": True}
    target = str(s1.get("target_element", "none")).lower()
    expl, instr = str(s1.get("defect_explanation", "")), str(s1.get("actionable_refinement_instruction", ""))
    extra = {"stage1": {"statement": s1.get("statement", ""), "quote": s1.get("quote", ""), "supported": supported,
                        "contradicts": contradicts, "terminology_mismatch": term, "bad_distractors": bad_d, "several_correct": several, "key_repeats_question": key_in_q, "repeated_phrase": rep_phrase, "source_reference": src_ref, "source_phrase": src_phrase}}
    if not (flags["factuality_valid"] and flags["no_ambiguity"]):
        if term:     # тип запитання не збігається з типом ключа: узгоджене виправлення запитання і ключа (фіксований шаблон)
            tl = ["question", "key"]; target = ", ".join(tl); extra["target_elements"] = tl
            expl = RL.render("terminology_mismatch", "explanation", defect=(expl or "тип об'єкта в запитанні не збігається з типом ключа").strip())
            instr = RL.template("terminology_mismatch")["instruction"]
        elif several: target = "question"
        elif not flags["factuality_valid"]: target = "key"
        elif bad_d: target = bad_d[0] if bad_d[0] in ("d1", "d2", "d3") else target
        ok_ids = [d for d in bad_d if d in ("d1", "d2", "d3")]
        if len(ok_ids) > 1 and not (term or several) and flags["factuality_valid"]:
            target, tl = _multi(None, target, ok_ids); extra["target_elements"] = tl
        if several and not term:     # запитання надто широке: потрібно змінювати запитання і варіанти разом
            tl = ["question", "key", "d1", "d2", "d3"]
            target = ", ".join(tl); extra["target_elements"] = tl
            expl, instr = RL.template("several_correct")["explanation"], RL.template("several_correct")["instruction"]
        # захист від самосуперечності: «виправити» ключ на той самий текст
        if target == "key" and key.strip()[:40].lower() in instr.lower() and not (term or several or bad_d):
            flags["factuality_valid"] = True; extra["self_consistency_override"] = True
        if not flags["factuality_valid"] and not (term or several):     # ключ суперечить тексту/не підтверджений: запитання і ключ узгоджено правимо разом
            tl = ["question", "key"]; target = ", ".join(tl); extra["target_elements"] = tl
            t_ = RL.template("factuality_key")
            expl = RL.render("factuality_key", "explanation", defect=(expl or "ключ не підтверджується навчальним текстом").strip())
            instr = t_["instruction"]
        if not (term or several or bad_d) and flags["factuality_valid"]:   # лише мовні помилки / розпливчастий ключ
            if any(str(w["element"]).lower() == "options" for w in wi):
                wo = next(w for w in wi if str(w["element"]).lower() == "options")
                in_q = wo.get("in_question") is not False
                tl_ = ([] if in_q else ["question"]) + ["key", "d1", "d2", "d3"]
                target, tl = _multi(None, "key", tl_); extra["target_elements"] = tl
                parts = [str(wo.get(k) or "").strip() for k in ("common_start", "common_end")]
                parts = [x for x in parts if x] or [str(wo.get("common_part") or "").strip()]
                expl, instr = redundancy_instruction(" … ".join(f"«{x}»" for x in parts if x) if any(parts) else "", in_q)
            elif wi:
                els = []
                for w in wi:
                    e_ = str(w["element"]).lower()
                    if e_ not in els: els.append(e_)
                probs = "; ".join(f'{str(w["element"]).lower()}: {w.get("problem", "")}' for w in wi)
                if len(els) > 1:     # мовні помилки виправляються одразу в усіх названих елементах
                    target, tl = _multi(None, target, els); extra["target_elements"] = tl
                else:
                    target = els[0]
                expl = RL.render("wording", "explanation", problems=probs)
                instr = RL.render("wording", "instruction", target=target)
            elif src_ref:     # запитання посилається на джерело: правимо лише запитання
                target = "question"
                expl = RL.render("source_reference", "explanation", phrase=src_phrase)
                instr = RL.render("source_reference", "instruction", phrase=src_phrase)
            elif key_in_q:     # ключ повторюється в запитанні: правимо лише запитання
                target = "question"
                expl = RL.render("key_repeats_question", "explanation", phrase=rep_phrase or "слова, що відтворюють ключ")
                instr = RL.render("key_repeats_question", "instruction", phrase=rep_phrase or "слова, що відтворюють ключ")
            else:
                target = "key"; lim_ = RL.length_limit(m["distractor_mean_words"])
                expl = RL.template("key_vague")["explanation"]
                instr = RL.render("key_vague", "instruction", limit=lim_)
        if not (flags["factuality_valid"] and flags["no_ambiguity"]):
            return _report(iteration, flags, target, expl, instr, m, extra)
    # ---------- Етап 2 ----------
    yn = lambda v: "так" if v else "ні"
    metrics = RL.render("stage2_metrics", "line", key_words=m["key_words"], distractor_mean_words=m["distractor_mean_words"], cv_length=m["cv_length"],
                        key_too_long=yn(m["key_too_long"]), shortcut_phrase=yn(m["shortcut_phrase"]),
                        categorical_candidates=RL.categorical_candidates(key, ds), format_candidates=RL.format_candidates(key, ds))
    if m["key_too_long"]:
        metrics += RL.render("stage2_metrics", "target_suffix", limit=RL.length_limit(m["distractor_mean_words"]))
    s2 = _chat(STAGE2_SYSTEM, f"{ctx_block}\n\nТестове завдання MCQ:\n{mcq_json}\n{metrics}\nПровеcти Етап 2 та повернути JSON.", 0.0)
    pl = s2.get("plausibility_valid") is True
    bad2 = [c for c in (s2.get("distractor_checks") or []) if c.get("same_domain") is not True]
    chk = s2.get("distractor_checks") or []
    flags["plausibility_valid"] = bool(not bad2) if len(chk) >= 3 else bool(pl)
    cat_n = len([x for x in (s2.get("categorical_distractors") or []) if str(x).lower() in ("d1", "d2", "d3")])
    fmt_cand = RL.format_candidates(key, ds) != "розбіжностей немає"
    fmt_ids = [str(x).lower() for x in (s2.get("format_mismatch") or []) if str(x).lower() in ("key", "d1", "d2", "d3")] if fmt_cand else []   # вердикт моделі лише за наявності кандидата від коду
    fmt_rule = RL.format_defect(key, ds)     # жорсткі правила (латиниця 3 з 4; пояснення в дужках у 1–3 з 4) — без судження моделі
    fmt_ids = [x for x in ("key", "d1", "d2", "d3") if x in fmt_ids or fmt_rule]
    s2_false_named = bool(str(s2.get("shortcut_found") or "").strip()) or cat_n >= RL.param("categorical_words", "min_distractors") or bool(fmt_ids)
    # вердикт моделі «є підказка» враховується лише з названою причиною (фраза або категоричні слова); довжину і фрази-шаблони рахує код
    flags["no_shortcuts"] = bool((s2.get("no_shortcuts") is True or not s2_false_named) and m["code_no_shortcuts"])
    target = str(s2.get("target_element", "none")).lower()
    expl, instr = str(s2.get("defect_explanation", "")), str(s2.get("actionable_refinement_instruction", ""))
    targets = None
    if not flags["plausibility_valid"]:
        bad_ids = [str(c.get("id", "")).lower() for c in bad2 if str(c.get("id", "")).lower() in ("d1", "d2", "d3")]
        if bad_ids:
            targets = bad_ids
            expl, instr = plausibility_instruction(bad_ids)
        elif target not in ("d1", "d2", "d3"):
            target = "d1"; expl, instr = plausibility_instruction(["d1"])
    cat_ids = [str(x).lower() for x in (s2.get("categorical_distractors") or []) if str(x).lower() in ("d1", "d2", "d3")]
    if len(cat_ids) < RL.param("categorical_words", "min_distractors"): cat_ids = []   # правило: сигнал лише від ≥N дистракторів
    extra = dict(extra); extra["categorical_distractors"] = cat_ids
    if cat_ids and not m["key_too_long"]:     # категоричні слова (оцінка моделі): правимо всі такі дистрактори разом
        targets = cat_ids
        expl, instr = absolute_instruction(targets); flags["no_shortcuts"] = False
    if fmt_ids and not cat_ids and not m["key_too_long"]:     # неоднорідність формату (оцінка моделі за кандидатами коду): правимо названі варіанти разом
        targets = fmt_ids; flags["no_shortcuts"] = False
        prob = str(s2.get("format_problem") or (fmt_rule or "ознаки запису є лише в частини варіантів")).strip()
        expl = RL.render("format_symmetry", "explanation", problem=prob); instr = RL.render("format_symmetry", "instruction", names=", ".join(fmt_ids))
        extra["format_mismatch"] = fmt_ids
    if m["key_too_long"]:     # об'єктивний дефект (довжина) має пріоритет над суб'єктивним судженням LLM про дистрактори
        targets = None; target = "key"; expl, instr = length_instruction(m); flags["no_shortcuts"] = False
    if targets:
        target, tl = _multi(None, target, targets)
        if tl: extra = dict(extra); extra["target_elements"] = tl
    if not flags["no_shortcuts"] and (target in ("none", "") or flags["plausibility_valid"]):
        if not flags["plausibility_valid"]: pass
        else:
            target = "key" if target in ("none", "") else target
            instr = instr or RL.template("length_fallback")["instruction"]
            expl = expl or RL.template("length_fallback")["explanation"]
    return _report(iteration, flags, target, expl, instr, m, extra)

def local_regenerate(context: Optional[str], mcq: Dict, report: Dict, passport: Optional[str] = None) -> Dict:
    """Переписує ЛИШЕ дефектний елемент; решта береться з попередньої версії без змін (гарантується кодом)."""
    target = report["target_element"]
    ctx = truncate_source_text(context, RL.limit("context_chars")) if context else "(вихідний текст не надано)"
    # Той самий Генератор у режимі виправлення: діалог «завдання → моя попередня відповідь → зауваження Критика»
    first = f'Навчальний текст:\n"""\n{ctx}\n"""\n' + (f"Довідка Аналітика (для орієнтування; єдиним джерелом фактів є навчальний текст):\n{passport}\n" if passport else "") + "Синтезуй тестове завдання MCQ у форматі JSON."
    fb = REFINE_USER_PROMPT.format(critic_target_element=report.get("target_element", target),
        critic_defect_explanation=report["defect_explanation"],
        critic_actionable_refinement_instruction=report["actionable_refinement_instruction"])
    msgs = [{"role": "user", "content": first},
            {"role": "assistant", "content": json.dumps(mcq, ensure_ascii=False)},
            {"role": "user", "content": fb}]
    new = _chat(GENERATOR_SYSTEM_PROMPT + GENERATOR_REVISION_ADDENDUM, msgs)
    if isinstance(new, dict) and new.get("cannot_fix"):     # генератор повідомив, що виправлення неможливе без вигадування фактів:
        out = json.loads(json.dumps(mcq))                    # доопрацювання завершується незалежно від цільового елемента (формула 2.8)
        out["_fix_failed"] = str(new["cannot_fix"])
        return out
    out = json.loads(json.dumps(mcq))
    try:
        if target == "key": out["key"] = new["key"]
        elif target == "question": out["question"] = new["question"]
        elif target in ("d1", "d2", "d3") or report.get("target_elements"):
            for tid in (report.get("target_elements") or [target]):
                if tid == "key": out["key"] = new["key"]; continue
                if tid == "question": out["question"] = new["question"]; continue
                idx = int(tid[1]) - 1
                cand = next((d for d in new["distractors"] if d.get("id") == tid), new["distractors"][idx])
                out["distractors"][idx] = {"id": tid, "text": cand["text"],
                                           "misconception_modeled": cand.get("misconception_modeled", "")}
        else:
            return new   # target не визначено: допускаємо повний переписаний варіант
    except (KeyError, IndexError, TypeError):
        return mcq
    return out

def refine_loop(context: Optional[str], mcq: Dict, target_level=None, max_iterations: int = None) -> Dict:
    """Refinement Loop M4: до max_iterations спроб. Статус: passed_at_iter_1 | fixed_at_iter_2 | hard_reject."""
    max_iterations = max_iterations or RL.max_iterations()
    history = []
    cur = mcq
    for it in range(1, max_iterations + 1):
        rep = critique(context, cur, target_level, it)
        history.append({"mcq": cur, "report": rep})
        if rep["verdict"] == "APPROVED":
            return {"final_mcq": cur, "status": "passed_at_iter_1" if it == 1 else f"fixed_at_iter_{it}", "iterations": it, "history": history}
        if it < max_iterations:
            cur = local_regenerate(context, cur, rep)
            if "_fix_failed" in cur:     # виправлення неможливе без вигадування фактів (формула 2.8)
                history[-1]["fix_failed"] = cur.pop("_fix_failed")
                return {"final_mcq": cur, "status": "hard_reject", "iterations": it, "history": history}
    return {"final_mcq": cur, "status": "hard_reject", "iterations": max_iterations, "history": history}


class LectureTooLarge(ValueError):
    """Лекція перевищує допустимий розмір вхідного тексту."""

def check_lecture_size(text: str) -> int:
    """Вхідний параметр: розмір лекції обмежений (rules.json → limits.lecture_max_chars).
    Більшу лекцію користувач ділить на частини самостійно; система її не обрізає і не ділить."""
    n = len((text or "").strip()); lim = RL.limit("lecture_max_chars")
    if n > lim:
        raise LectureTooLarge(f"Лекція завелика: {n} символів, допустимо не більше {lim}. "
                              f"Розділіть її на частини за темами (по ≤ {lim} символів) і завантажте кожну окремо.")
    return n
