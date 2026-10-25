"""
Агент-генератор (режим генерації) для всіх конфігурацій.

  plan_m1 — монолітний zero-shot запит на всю лекцію (M1, базова лінія, без правил паритету);
  plan_m2 — генерація завдань із правилами методу (M2 — уся лекція; M3, M4 — текст блока з довідкою аналітика);
  _norm_mcq — приведення відповіді моделі до формату завдання {"question", "key", "distractors"}.
Режим виправлення (local_regenerate) і критик — у модулі critic.py; конвеєр, що викликає генератор, — lecture_graph.py.
Функції-вузли нижче (analyst_node … ) використовує лише старий маршрут /api/generate-test (langgraph_graph.py, швидкий режим).
"""
import json, os, sys
from typing import Dict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import critic
from critic import GENERATOR_SYSTEM_PROMPT, _chat
from text_utils import truncate_source_text
from langgraph_state import GeneratedQuestion, VerificationResult

LEVEL_NAMES = {1: "Remembering", 2: "Understanding", 3: "Applying"}
BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply"}

# ---- 3.1 Агент-Аналітик (дослівно за регламентом) ----
ANALYST_SYSTEM_PROMPT = critic.RL.prompt("analyst_system")
GENERATOR_SPEC_ADDENDUM = critic.RL.prompt("generator_spec_addendum")
ANALYST_USER = 'Цільовий рівень когнітивної складності: Рівень {target_level} ({level_name})\nВхідний навчальний текст:\n"""\n{context_text}\n"""\nСформуй специфікацію завдання (Task Spec) у форматі JSON.'

# ---- 3.2 Генератор ----
GENERATOR_USER = 'Навчальний текст:\n"""\n{context_text}\n"""\nСпецифікація від Агента-Аналітика:\n{task_spec_json}\nСинтезуй тестове завдання MCQ у форматі JSON.'
# M2: без Аналітика. Системний промпт той самий, але без згадки Task Spec (замість неї - цільовий рівень).
GENERATOR_SYSTEM_NO_SPEC = GENERATOR_SYSTEM_PROMPT.replace(
    "Ти отримуєш навчальний текст (контекст C) та педагогічну специфікацію (Task Spec) від Агента-Аналітика.",
    "Ти отримуєш навчальний текст (контекст C) та цільовий рівень за таксономією Блума.").replace(
    "відповідати когнітивному завданню зі специфікації (особливо для Рівня 3 — це має бути конкретна задача / кейс)",
    "відповідати цільовому рівню Блума (для Рівня 3 — це має бути конкретна задача / кейс)")
GENERATOR_USER_NO_SPEC = 'Навчальний текст:\n"""\n{context_text}\n"""\nЦільовий рівень Блума: Рівень {target_level}\nСинтезуй тестове завдання MCQ у форматі JSON.'

# ---- 4.1 M1: монолітний Baseline (дослівно за регламентом) ----
M1_SYSTEM = ("Ти — викладач університету. Склади одне тестове завдання з чотирма варіантами відповідей (одна правильна, три хибні) "
             "за наданим текстом. Завдання має відповідати Рівню {target_level} за таксономією Блума.\n"
             'Поверни JSON з полями: "question", "key", "distractors" (масив з 3 рядків).')
M1_USER = 'Навчальний текст:\n"""\n{context_text}\n"""'

def _norm_mcq(raw: Dict) -> Dict:
    ds = raw.get("distractors") or []
    ds = [d if isinstance(d, dict) else {"id": f"d{i+1}", "text": str(d), "misconception_modeled": ""} for i, d in enumerate(ds)]
    ds = (ds + [{"text": ""}] * 3)[:3]
    return {"question": str(raw.get("question", "")), "key": str(raw.get("key", "")),
            "distractors": [{"id": f"d{i+1}", "text": str(d.get("text", "")), "misconception_modeled": str(d.get("misconception_modeled", ""))}
                            for i, d in enumerate(ds)]}

def _sync_legacy(state, mcq):
    """Для сумісності з рештою проєкту (Excel-експорт, інтерфейс): варіанти A-D, ключ = A."""
    level = int(state.get("target_bloom_level") or 1)
    opts = {"A": mcq["key"], "B": mcq["distractors"][0]["text"], "C": mcq["distractors"][1]["text"], "D": mcq["distractors"][2]["text"]}
    state["generated_question"] = GeneratedQuestion(question_stem=mcq["question"], options=opts, correct_answer="A",
                                                    explanation="", bloom_level=BLOOM_NAMES.get(level, "Remember"))

def _ctx(state): return truncate_source_text(state["source_text"], critic.RL.limit("context_chars"))

def m1_node(state):
    level = int(state.get("target_bloom_level") or 1)
    if state.get("mcq"):                       # питання вже створене пакетним запитом (batch_plan)
        raw = state["mcq"]
    else:
        raw = _chat(M1_SYSTEM.format(target_level=level), M1_USER.format(context_text=_ctx(state)), 0.3)
    state["mcq"] = _norm_mcq(raw); _sync_legacy(state, state["mcq"])
    state["iteration_count"] = 1; state["current_agent"] = "Generator"; state["h3_status"] = "n/a"
    state["status"] = "M1 zero-shot generated"; return state

def analyst_node(state):
    level = int(state.get("target_bloom_level") or 1)
    if state.get("task_spec"):                 # Task Spec уже складено пакетним планом Аналітика
        state["current_agent"] = "Analyst"; state["status"] = "Task Spec from batch plan"; return state
    spec = critic.analyst_chat(ANALYST_SYSTEM_PROMPT, ANALYST_USER.format(target_level=level, level_name=LEVEL_NAMES[level], context_text=_ctx(state)), 0.0)
    state["task_spec"] = spec; state["current_agent"] = "Analyst"; state["status"] = "Task Spec created"; return state

def generator_node(state):
    level = int(state.get("target_bloom_level") or 1)
    if state.get("mcq") and not state.get("task_spec") and not int(state.get("iteration_count") or 0):
        state["mcq"] = _norm_mcq(state["mcq"]); _sync_legacy(state, state["mcq"])   # M2: питання з пакетного запиту
        state["iteration_count"] = 1; state["current_agent"] = "Generator"; state["status"] = "MCQ from batch"; return state
    if state.get("task_spec"):
        raw = _chat(GENERATOR_SYSTEM_PROMPT + GENERATOR_SPEC_ADDENDUM + critic.GENERATOR_PARITY_ADDENDUM, GENERATOR_USER.format(context_text=_ctx(state),
                    task_spec_json=json.dumps(state["task_spec"], ensure_ascii=False, indent=1)), 0.3)
    else:
        raw = _chat(GENERATOR_SYSTEM_NO_SPEC, GENERATOR_USER_NO_SPEC.format(context_text=_ctx(state), target_level=level), 0.3)
    state["mcq"] = _norm_mcq(raw); _sync_legacy(state, state["mcq"])
    state["iteration_count"] = 1; state["current_agent"] = "Generator"; state["status"] = "MCQ generated"; return state

def critic_node(state):
    it = int(state.get("iteration_count") or 1)
    rep = critic.critique(state["source_text"], state["mcq"], state.get("target_bloom_level"), it)
    state["critic_report"] = rep
    state.setdefault("critic_history", [])
    state["critic_history"] = list(state["critic_history"]) + [{"iteration": it, "mcq": state["mcq"], "report": rep}]
    approved = rep["verdict"] == "APPROVED"
    state["verification"] = VerificationResult(approved=approved, feedback=rep["actionable_refinement_instruction"],
                                               rejection_reason=rep["defect_explanation"],
                                               unambiguity=int(rep["factuality_valid"] and rep["no_ambiguity"]),
                                               distractor_adequacy=int(rep["plausibility_valid"]),
                                               grammatical_consistency=int(rep["no_shortcuts"]), bloom_level=1)
    cfg = state.get("configuration")
    if cfg == "M4":
        state["h3_status"] = "passed_at_iter_1" if (approved and it == 1) else (f"fixed_at_iter_{it}" if approved else
                             ("hard_reject" if it >= int(state.get("max_iterations") or 2) else "rejected_pending"))
    else:
        state["h3_status"] = "approved" if approved else "rejected"
    state["current_agent"] = "Critic"; state["status"] = rep["verdict"]; return state

def refine_node(state):
    mcq = critic.local_regenerate(state["source_text"], state["mcq"], state["critic_report"])
    state["mcq"] = _norm_mcq(mcq); _sync_legacy(state, state["mcq"])
    state["iteration_count"] = int(state.get("iteration_count") or 1) + 1
    state["current_agent"] = "Generator"; state["status"] = f"Local refinement of {state['critic_report']['target_element']}"; return state


# ================= Пакетне планування: усі питання однієї частини лекції створюються одним викликом =================
def split_lecture(text, max_chars=12000):
    """Ділить лекцію на послідовні різні частини (без перетину) приблизно рівної довжини по межах абзаців."""
    text = text.strip(); n = max(1, -(-len(text) // max_chars)); size = len(text) / n; parts, start = [], 0
    for i in range(1, n + 1):
        end = len(text) if i == n else text.rfind("\n", start, int(size * i) + 400)
        if end <= start: end = int(size * i)
        parts.append(text[start:end].strip()); start = end
    return [p for p in parts if p]

_BATCH_NOTE = ("\n\nВАЖЛИВО: тобі потрібно створити {n} РІЗНИХ завдань за цим текстом. Кожне присвячене іншому поняттю чи факту "
               "(повтори тем та схожі запитання заборонені). Цільові рівні Блума — суворо за порядком: {levels}.")

def build_m1_batch(ctx, levels):
    n = len(levels)
    sys_ = ("Ти — викладач університету. Склади " + str(n) + " різних тестових завдань, кожне з чотирма варіантами відповідей (одна правильна, три хибні) "
            "за наданим текстом. Кожне завдання присвячене іншому поняттю з тексту (без повторів). Рівні за таксономією Блума — суворо за порядком: "
            + str(levels) + ".\n" 'Поверни JSON: {"items": [{"target_level": int, "question": "...", "key": "...", "distractors": [3 рядки]}, ...]}.')
    return sys_, M1_USER.format(context_text=ctx)

def build_m2_batch(ctx, levels, parity=True, passport=None):
    sys_ = GENERATOR_SYSTEM_NO_SPEC.split("Ти ПОВИНЕН повернути")[0] + _BATCH_NOTE.format(n=len(levels), levels=levels)
    if parity: sys_ += critic.GENERATOR_PARITY_ADDENDUM
    sys_ += ('\nПоверни ВИКЛЮЧНО валідний JSON: {"items": [{"target_level": int, "question": "...", "key": "...", "distractors": '
             '[{"id": "d1", "text": "...", "misconception_modeled": "..."}, {"id": "d2", ...}, {"id": "d3", ...}]'
             + '}, ...]}')
    return sys_, M1_USER.format(context_text=ctx) + (f"\nДовідка Аналітика (для орієнтування; єдиним джерелом фактів є навчальний текст):\n{passport}" if passport else "")

def build_specs_batch(ctx, levels):
    sys_ = ANALYST_SYSTEM_PROMPT.split("Ти ПОВИНЕН повернути")[0] + _BATCH_NOTE.format(n=len(levels), levels=levels) + (
        ' selected_concept у кожної специфікації має бути іншим.\nТи ПОВИНЕН повернути ВИКЛЮЧНО валідний JSON: {"specs": [<Task Spec зі структурою '
        'target_bloom_level, level_name, selected_concept, underlying_rule, cognitive_action_verb, context_scenario_requirement, answer_spec, terms_normalized, forbidden_patterns, '
        'instruction_for_generator>, ...]}')
    return sys_, M1_USER.format(context_text=ctx)

def plan_m1(ctx, levels): return _chat(*build_m1_batch(ctx, levels), 0.3)["items"]
def plan_m2(ctx, levels, passport=None): return _chat(*build_m2_batch(ctx, levels, passport=passport), 0.3)["items"]
def plan_specs(ctx, levels): return _chat(*build_specs_batch(ctx, levels), 0.3)["specs"]
