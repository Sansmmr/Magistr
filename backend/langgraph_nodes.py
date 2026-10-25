"""
LangGraph Nodes for Multi-Agent Test Generation System
Defines the three agent nodes: Analyst, Generator, Critic
"""

import json
from typing import Dict, Any, Callable
from openai import OpenAI
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from langgraph_state import AgentState, AnalysisResult, GeneratedQuestion, VerificationResult

# Ukrainian language prompts
UKRAINIAN_MODE = Config.LANGUAGE == "ukrainian"


# --- M1 (Baseline): монолітний zero-shot запит. Текст промпту дослівно за регламентом (п. 4.1). ---
M1_SYSTEM_PROMPT = (
    "Ти — викладач університету. Склади одне тестове завдання з чотирма варіантами відповідей "
    "(одна правильна, три хибні) за наданим текстом. Завдання має відповідати Рівню {target_level} "
    "за таксономією Блума.\n"
    "Поверни JSON з полями: \"question\", \"key\", \"distractors\" (масив з 3 рядків)."
)
M1_USER_PROMPT = 'Навчальний текст:\n"""\n{context_text}\n"""'
BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply"}

ANALYST_PROMPT_UA = """Проаналізуйте наступний навчальний текст та виділіть ключову інформацію для створення тестових питань.

Вихідний текст:
{source_text}

Надайте відповідь у наступному JSON форматі:
{{
    "key_facts": ["важливий факт 1", "важливий факт 2", ...],
    "key_terms": ["термін 1", "термін 2", ...],
    "rules": ["правило або принцип 1", "правило або принцип 2", ...],
    "concepts": ["концепція 1", "концепція 2", ...],
    "relationships": ["звязок 1", "звязок 2", ...],
    "summary": "Коротке концептуальне резюме тексту"
}}

Фокусуйтеся на фактах та концепціях, які можуть бути основою для тестових питань."""

ANALYST_SYSTEM_UA = "Ви є експерт-аналітик тексту, який виділяє ключові факти та концепції з навчальних текстів українською мовою."

GENERATOR_PROMPT_UA = """На основі наступного вихідного тексту та аналізу згенеруйте повністю нове тестове питання з вибором однієї відповіді українською мовою.

Вихідний текст:
{source_text}

Аналіз:
Ключові факти: {analysis.key_facts}
Ключові терміни: {analysis.key_terms}
Правила: {analysis.rules}
Концепції: {analysis.concepts}
Звязки: {analysis.relationships}
Резюме: {analysis.summary}
{feedback_section}

Згенеруйте НОВЕ тестове питання з вибором однієї відповіді:
|- Чітким формулюванням питання українською мовою
|- 4 варіантами відповідей (A, B, C, D)
|- Точно 1 правильною відповіддю (однозначно)
|- Правдоподібними дистракторами, які не є очевидно неправильними
|- Варіанти відповідей граматично узгоджені з формулюванням питання
|- Питання має бути відповідним з вихідного тексту
|- Визначте рівень таксономії Блума (Remember, Understand, Apply, Analyze, Evaluate, Create)

Надайте відповідь у цьому JSON форматі:
{{
    "question_stem": "Текст питання українською мовою",
    "options": {{
        "A": "Текст варіанту A українською мовою",
        "B": "Текст варіанту B українською мовою",
        "C": "Текст варіанту C українською мовою",
        "D": "Текст варіанту D українською мовою"
    }},
    "correct_answer": "A",
    "explanation": "Коротке пояснення українською мовою, чому це правильно",
    "bloom_level": "Remember/Understand/Apply/Analyze/Evaluate/Create"
}}"""

GENERATOR_SYSTEM_UA = "Ви є експерт з генерації тестових питань для освітнього контенту українською мовою."

CRITIC_PROMPT_UA = """Ви є Агент-Критик (LLM-as-a-Judge) для освітніх тестових питань українською мовою.
БУДЬТЕ ВЕЛИКЕ КРИТИЧНИМ! ВІДХИЛЯЙТЕ навіть дрібні помилки.

Оцінити згенероване питання, використовуючи ТОЧНО ці чотири критерії, оцінюючи кожен від 0 до 1:

Критерій 1 - Однозначність (Unambiguity):
  Чи має питання ТОЧКО ОДНУ чітко правильну відповідь?
  ВІДХИЛИТЬ, якщо:
  - Кілька відповідей можуть бути правильними
  - Ключова відповідь неточна або помилкова
  - Питання допускає подвійні трактування
  Оцінка 1 = однозначно (одна правильна відповідь); Оцінка 0 = неоднозначно або кілька правильних відповідей

Критерій 2 - Адекватність дистракторів (Distractor Adequacy):
  Чи є неправильні варіанти правдоподібними і не очевидно безглуздими?
  ВІДХИЛИТЬ, якщо:
  - Дистрактори абсурдні (наприклад, Microsoft Word для питання про SQL)
  - Дистрактори тривіально неправильні
  - Дистрактори не повязані з темою
  - Дистрактори очевидно безглуздні
  Оцінка 1 = всі дистрактори реалістичні; Оцінка 0 = дистрактори абсурдні або тривіально неправильні

Критерій 3 - Граматична узгодженість (Grammatical Consistency):
  Чи граматично узгоджені варіанти відповідей з формулюванням питання?
  ВІДХИЛИТЬ, якщо:
  - Термінологічна невідповідність (наприклад, запитує про функції, але ключ - це клауза)
  - Варіанти не підходять граматично
  Оцінка 1 = ідеальна узгодженість; Оцінка 0 = варіанти не підходять граматично

Критерій 4 - Рівень таксономії Блума (Bloom Level):
  Чи відповідає рівень запитання/відповіді заявленому рівню Блума?
  ВІДХИЛИТЬ, якщо:
  - Відповідь - просте впізнавання (Remember), але заявлено Understand/Apply
  - Відповідь не пояснює або не порівнює
  Оцінка 1 = відповідає рівню; Оцінка 0 = не відповідає рівню або рівень невизначено

Вихідний текст:
{source_text}

Згенероване питання:
Питання: {question_stem}
Варіанти:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
Правильна відповідь: {correct_answer}
Рівень Блума: {bloom_level}

{original_text}

Якщо ХОЧ Б ОДИН критерій отримав оцінку 0, надайте конкретні дієві зауваження для Генератора.

Надайте відповідь у цьому JSON форматі:
{{
    "unambiguity": 0 або 1,
    "distractor_adequacy": 0 або 1,
    "grammatical_consistency": 0 або 1,
    "bloom_level": 0 або 1,
    "feedback": "Конкретна критика для Генератора, якщо хоча б одна оцінка = 0, інакше порожній рядок",
    "issues": ["список конкретних виявлених проблем"]
}}"""

CRITIC_PROMPT_EN = """You are a Critic Agent (LLM-as-a-Judge) for educational test questions.
BE VERY CRITICAL! REJECT even minor errors.

Evaluate the generated question using EXACTLY these four criteria, scoring each from 0 to 1:

Criterion 1 - Unambiguity:
  Does the question have exactly ONE clearly correct answer?
  REJECT if:
  - Multiple answers could be correct
  - Key answer is inaccurate or wrong
  - Question allows ambiguous interpretations
  Score 1 = unambiguous (one correct answer); Score 0 = ambiguous or multiple correct answers

Criterion 2 - Distractor Adequacy:
  Are the incorrect options plausible and not obviously nonsensical?
  REJECT if:
  - Distractors are absurd (e.g., Microsoft Word for SQL question)
  - Distractors are trivially wrong
  - Distractors are unrelated to the topic
  - Distractors are obviously nonsensical
  Score 1 = all distractors are realistic; Score 0 = distractors are absurd or trivially wrong

Criterion 3 - Grammatical Consistency:
  Are the answer options grammatically aligned with the question stem?
  REJECT if:
  - Terminological mismatch (e.g., asks about functions but key is a clause)
  - Options dont fit grammatically
  Score 1 = perfect alignment; Score 0 = options don't fit grammatically

Criterion 4 - Bloom Level:
  Does the question/answer level match the claimed Bloom level?
  REJECT if:
  - Answer is simple recognition (Remember) but claimed as Understand/Apply
  - Answer doesn't explain or compare
  Score 1 = matches level; Score 0 = doesnt match level or level is undefined

Source Text:
{source_text}

Generated Question:
Question: {question_stem}
Options:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
Correct Answer: {correct_answer}
Bloom Level: {bloom_level}

{original_text}

If ANY criterion receives a score of 0, provide specific actionable feedback for the Generator.

Return your response in this JSON format:
{{
    "unambiguity": 0 or 1,
    "distractor_adequacy": 0 or 1,
    "grammatical_consistency": 0 or 1,
    "bloom_level": 0 or 1,
    "feedback": "Specific critique for the Generator if any score = 0, otherwise empty string",
    "issues": ["list of specific issues found"]
}}"""

CRITIC_SYSTEM_UA = "Ви є експерт з оцінювання освітніх тестових завдань українською мовою."

# Use Ukrainian prompts if configured
ANALYST_PROMPT = ANALYST_PROMPT_UA if UKRAINIAN_MODE else """Analyze the following text and extract key information suitable for creating test questions.

Source Text:
{source_text}

Provide your response in the following JSON format:
{{
    "key_facts": ["fact 1", "fact 2", ...],
    "concepts": ["concept 1", "concept 2", ...],
    "relationships": ["relationship 1", "relationship 2", ...],
    "summary": "Brief conceptual summary of the text"
}}

Focus on facts and concepts that could form the basis of multiple-choice questions."""

ANALYST_SYSTEM = ANALYST_SYSTEM_UA if UKRAINIAN_MODE else "You are an expert text analyst who extracts key facts and concepts from educational texts."

GENERATOR_PROMPT = GENERATOR_PROMPT_UA if UKRAINIAN_MODE else """Based on the following source text and analysis, generate a completely new multiple-choice question.

Source Text:
{source_text}

Analysis:
Key Facts: {analysis.key_facts}
Concepts: {analysis.concepts}
Relationships: {analysis.relationships}
Summary: {analysis.summary}

Generate a NEW multiple-choice question with:
|- A clear question stem
|- 4 options (A, B, C, D)
|- Exactly 1 correct answer
|- The question must be answerable from the source text

Return your response in this JSON format:
{{
    "question_stem": "The question text",
    "options": {{
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
    }},
    "correct_answer": "A",
    "explanation": "Brief explanation of why this is correct",
    "bloom_level": "Remember"
}}"""

GENERATOR_SYSTEM = GENERATOR_SYSTEM_UA if UKRAINIAN_MODE else "You are an expert test question generator for educational content."

CRITIC_PROMPT = CRITIC_PROMPT_UA if UKRAINIAN_MODE else CRITIC_PROMPT_EN

CRITIC_SYSTEM = CRITIC_SYSTEM_UA if UKRAINIAN_MODE else "You are an expert educational assessment evaluator."

from verification_utils import build_verification_result, CRITERIA_THRESHOLD
from text_utils import truncate_source_text
from difficulty_utils import (
    generator_difficulty_instruction,
    critic_difficulty_instruction,
    normalize_difficulty,
)


def _api_key_for_base_url(base_url: str) -> str:
    """Pick the correct API key for the configured provider."""
    if "deepseek" in (base_url or "").lower():
        return Config.DEEPSEEK_API_KEY
    return Config.OPENAI_API_KEY


class LangGraphAgents:
    """Container for LangGraph agent nodes with WebSocket support"""
    
    def __init__(self, log_callback: Callable = None, test_mode: bool = False):
        self.log_callback = log_callback
        self.test_mode = test_mode
        
        if not test_mode:
            self.analyst_client = OpenAI(
                api_key=_api_key_for_base_url(Config.ANALYST_BASE_URL),
                base_url=Config.ANALYST_BASE_URL
            )
            self.generator_client = OpenAI(
                api_key=_api_key_for_base_url(Config.GENERATOR_BASE_URL),
                base_url=Config.GENERATOR_BASE_URL
            )
            self.critic_client = OpenAI(
                api_key=_api_key_for_base_url(Config.VERIFICATION_BASE_URL),
                base_url=Config.VERIFICATION_BASE_URL
            )
        else:
            # Mock clients for testing
            self.analyst_client = None
            self.generator_client = None
            self.critic_client = None
    
    async def log(self, agent: str, message: str):
        """Send log message via callback"""
        if self.log_callback:
            await self.log_callback({
                "type": "log",
                "agent": agent,
                "message": message,
                "timestamp": "now"
            })
    
    def analyst_node(self, state: AgentState) -> AgentState:
        """
        Analyst Node: Extracts key facts and concepts from source text
        """
        try:
            source_text = truncate_source_text(state["source_text"])

            # Check if source text is too short
            if len(source_text) < 50:
                state["error_message"] = "Source text is too short for analysis (minimum 50 characters required)"
                state["status"] = "Analysis failed - text too short"
                state["analysis"] = AnalysisResult()
                return state
            
            # Test mode - return mock analysis
            if self.test_mode:
                words = source_text.split()[:10]
                analysis = AnalysisResult(
                    key_facts=[f"Fact about {word}" for word in words[:3]],
                    key_terms=[word for word in words[:5]],
                    rules=[f"Rule: {words[0]} applies to {words[1]}" if len(words) > 1 else "General rule from text"],
                    concepts=[f"Concept related to {word}" for word in words[3:6]],
                    relationships=[f"Relationship between {words[6]} and {words[7]}" if len(words) > 7 else "No relationships"],
                    summary="Test mode: Simulated analysis based on source text"
                )
                state["analysis"] = analysis
                state["current_agent"] = "Analyst"
                state["status"] = f"Analysis completed (TEST MODE). Found {len(analysis.key_facts)} key facts."
                return state
            
            prompt = ANALYST_PROMPT.format(source_text=source_text)

            response = self.analyst_client.chat.completions.create(
                model=Config.ANALYST_MODEL,
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            analysis = AnalysisResult(**result)
            
            state["analysis"] = analysis
            state["current_agent"] = "Analyst"
            state["status"] = f"Analysis completed. Found {len(analysis.key_facts)} key facts."
            
            return state
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                state["error_message"] = "API Error: Authentication failed. Check DEEPSEEK_API_KEY in .env file."
            elif "402" in error_msg or "Payment" in error_msg or "quota" in error_msg.lower():
                state["error_message"] = "API Error: Insufficient credits or quota exceeded. Check your API account balance."
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                state["error_message"] = "API Error: Connection timeout. Check your internet connection."
            else:
                state["error_message"] = f"Analyst error: {error_msg}"
            
            state["status"] = "Analysis failed"
            state["analysis"] = AnalysisResult()
            return state
    
    def _generate_m1_zero_shot(self, state: AgentState) -> AgentState:
        """M1: один zero-shot запит до базової LLM. Без Аналітика, без рубрики якості, без заборон,
        без підказок щодо дистракторів. Тільки промпт із регламенту (п. 4.1), temperature=0.3."""
        level = int(state.get("target_bloom_level") or 1)
        source_text = truncate_source_text(state["source_text"])
        response = self.generator_client.chat.completions.create(
            model=Config.GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": M1_SYSTEM_PROMPT.format(target_level=level)},
                {"role": "user", "content": M1_USER_PROMPT.format(context_text=source_text)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
        d = (raw.get("distractors") or []) + ["", "", ""]
        d = [x["text"] if isinstance(x, dict) else str(x) for x in d[:3]]
        gq = GeneratedQuestion(
            question_stem=raw.get("question", ""),
            options={"A": str(raw.get("key", "")), "B": d[0], "C": d[1], "D": d[2]},
            correct_answer="A",
            explanation="",
            bloom_level=BLOOM_NAMES.get(level, "Remember"),
        )
        state["generated_question"] = gq
        state["current_agent"] = "Generator"
        state["status"] = f"M1 zero-shot question generated (level {level}): {gq.question_stem[:60]}..."
        return state

    def generator_node(self, state: AgentState) -> AgentState:
        """
        Generator Node: Creates MCQ question based on analysis
        """
        try:
            state["iteration_count"] = state.get("iteration_count", 0) + 1

            if state.get("configuration") == "M1" and not self.test_mode:
                return self._generate_m1_zero_shot(state)

            source_text = truncate_source_text(state["source_text"])
            configuration_name = state.get("configuration", "M4")
            use_analyst = configuration_name in ("M3", "M4")
            analysis = state.get("analysis")
            if not use_analyst:
                # M1/M2: без Аналітика — генератор отримує лише вихідний текст
                analysis = AnalysisResult()

            if use_analyst and (not analysis or (not analysis.key_facts and not analysis.key_terms)):
                state["error_message"] = "Cannot generate question: No analysis available"
                state["status"] = "Generation failed"
                return state

            verification = state.get("verification")
            iteration_count = state.get("iteration_count", 0)
            feedback_section = ""
            if verification and not verification.approved and iteration_count > 0:
                if UKRAINIAN_MODE:
                    feedback_section = f"""
ВАЖЛИВО - Це спроба виправлення (ітерація {iteration_count + 1}).
Попереднє питання було ВІДХИЛЕНО Критиком. Виправте наступні проблеми:

Feedback від Критика: {verification.feedback or verification.rejection_reason}
Проблеми: {verification.issues}
Оцінки (1-5, поріг {CRITERIA_THRESHOLD}):
- Однозначність: {verification.unambiguity}
- Адекватність дистракторів: {verification.distractor_adequacy}
- Граматична узгодженість: {verification.grammatical_consistency}

Згенеруйте ПОКРАЩЕНЕ питання, яке враховує весь feedback вище.
"""
                else:
                    feedback_section = f"""
IMPORTANT - This is a revision attempt (iteration {iteration_count + 1}).
The previous question was REJECTED by the Critic. Fix the following issues:

Critic Feedback: {verification.feedback or verification.rejection_reason}
Issues: {verification.issues}
Scores (1-5, threshold {CRITERIA_THRESHOLD}):
- Unambiguity: {verification.unambiguity}
- Distractor adequacy: {verification.distractor_adequacy}
- Grammatical consistency: {verification.grammatical_consistency}

Generate an IMPROVED question that addresses all feedback above.
"""

            # Test mode - return mock question
            if self.test_mode:
                words = source_text.split()[:5]
                suffix = " (revised)" if feedback_section else ""
                generated_question = GeneratedQuestion(
                    question_stem=f"What is the main concept related to {words[0] if words else 'the topic'}?{suffix}",
                    options={
                        "A": f"Option related to {words[1] if len(words) > 1 else 'concept A'}",
                        "B": f"Option related to {words[2] if len(words) > 2 else 'concept B'}",
                        "C": f"Option related to {words[3] if len(words) > 3 else 'concept C'}",
                        "D": f"Option related to {words[4] if len(words) > 4 else 'concept D'}"
                    },
                    correct_answer="A",
                    explanation="Test mode: Simulated question generation",
                    bloom_level="Understand"
                )
                state["generated_question"] = generated_question
                state["current_agent"] = "Generator"
                state["status"] = f"Question generated (TEST MODE): {generated_question.question_stem[:60]}..."
                return state

            if UKRAINIAN_MODE:
                # НЕ використовуємо історію попередніх питань - чиста сесія
                # previous_questions = state.get("generated_questions_history", [])
                # previous_context = ""
                # if previous_questions:
                #     previous_context = "\nПопередні питання (НЕ повторюйте ці теми):\n"
                #     for i, prev_q in enumerate(previous_questions[-3:], 1):  # Останні 3 питання
                #         previous_context += f"{i}. {prev_q.get(\"question_stem\", \"\")[:100]}...\n"

                previous_context = ""  # Чистий контекст без історії

                # Отримуємо конфігурацію для різних інструкцій
                configuration = state.get("configuration", "M4")

                # Різні інструкції для кожної конфігурації
                config_instructions = {
                    "M1": "",
                    "M2": "",
                    "M3": "",
                    "M4": ""
                }

                config_instruction = config_instructions.get(configuration, "")

                if use_analyst:
                    analysis_block_ua = f"""Аналіз:
Ключові факти: {analysis.key_facts}
Ключові терміни: {analysis.key_terms}
Правила: {analysis.rules}
Концепції: {analysis.concepts}
Звязки: {analysis.relationships}
Резюме: {analysis.summary}"""
                else:
                    analysis_block_ua = "Аналіз: відсутній (Агент-Аналітик не використовується в цій конфігурації)."

                prompt = f"""На основі наступного вихідного тексту та аналізу згенеруйте повністю нове тестове питання з вибором однієї відповіді українською мовою.

Вихідний текст:
{source_text}

{analysis_block_ua}
{previous_context}
{generator_difficulty_instruction(normalize_difficulty(state.get("difficulty", "medium")))}
{feedback_section}
ВАЖЛИВО: Згенеруйте УНІКАЛЬНЕ тестове питання.
- НЕ використовуйте історію попередніх питань
- Питання має бути новим і оригінальним

ЗАБОРОНА (СТРОГО ЗАБОРОНЕНО):
- НЕ генеруйте питання про синтаксис коду (як написати команду)
- НЕ генеруйте питання про назви конкретних функцій/методів
- НЕ генеруйте питання про SQL команди (SELECT, INSERT, DROP, CREATE тощо)
- НЕ генеруйте питання про технічні деталі коду
- НЕ генеруйте питання про налаштування конфігурації
- НЕ генеруйте питання про змінні, параметри, аргументи
- НЕ генеруйте питання про структуру баз даних (таблиці, колонки, індекси)
- НЕ генеруйте питання про API ендпоінти, HTTP методи, статус коди

ЯКЩО ПИТАННЯ ВИМАЩЄ КОД, СИНТАКСИС, ТЕХНІЧНІ ДЕТАЛІ - НЕГРАЙТЬ ЙОГО!

Фокусуйтеся ТІЛЬКИ на:
- Архітектурних принципах (як система працює в цілому)
- Концептуальних поняттях (теоретичні основи)
- Взаємодії компонентів (які модулі взаємодіють)
- Метриках та KPI (які показники важливі)
- Бізнес-процесах (які бізнес-задачі вирішуються)
- Виборі технологій (чому обрати одну технологію над іншою)
- Паттернах проектування (які паттерни використовуються)

Згенеруйте НОВЕ тестове питання з вибором однієї відповіді:
- Чітким формулюванням питання українською мовою
- 4 варіантами відповідей (A, B, C, D)
- Точно 1 правильною відповіддю (однозначно)
- Правдоподібними дистракторами, які не є очевидно неправильними
- Варіанти відповідей граматично узгоджені з формулюванням питання
- Питання має бути відповідним з вихідного тексту

Надайте відповідь у цьому JSON форматі:
{{
    "question_stem": "Текст питання українською мовою",
    "options": {{
        "A": "Текст варіанту A українською мовою",
        "B": "Текст варіанту B українською мовою",
        "C": "Текст варіанту C українською мовою",
        "D": "Текст варіанту D українською мовою"
    }},
    "correct_answer": "A",
    "explanation": "Коротке пояснення українською мовою, чому це правильно",
    "bloom_level": "Remember"
}}

Рівні таксономії Блума:
- Remember: запамятовування фактів
- Understand: розуміння понять
- Apply: застосування знань
- Analyze: аналіз інформації
- Evaluate: оцінка інформації
- Create: створення нового
"""
            else:
                # НЕ використовуємо історію попередніх питань - чиста сесія
                # previous_questions = state.get("generated_questions_history", [])
                # previous_context = ""
                # if previous_questions:
                #     previous_context = "\nPrevious questions (DO NOT repeat these topics):\n"
                #     for i, prev_q in enumerate(previous_questions[-3:], 1):  # Last 3 questions
                #         previous_context += f"{i}. {prev_q.get(\"question_stem\", \"\")[:100]}...\n"

                previous_context = ""  # Чистий контекст без історії

                # Get configuration for different instructions
                configuration = state.get("configuration", "M4")

                # Different instructions for each configuration
                config_instructions = {
                    "M1": "",
                    "M2": "",
                    "M3": "",
                    "M4": ""
                }

                config_instruction = config_instructions.get(configuration, "")

                prompt = f"""Based on the following source text and analysis, generate a completely new multiple-choice question.

Configuration: {configuration}
Configuration instruction: {config_instruction}

Source Text:
{source_text}

Analysis:
Key Facts: {analysis.key_facts}
Key Terms: {analysis.key_terms}
Rules: {analysis.rules}
Concepts: {analysis.concepts}
Relationships: {analysis.relationships}
Summary: {analysis.summary}
{previous_context}
{generator_difficulty_instruction(normalize_difficulty(state.get("difficulty", "medium")))}
{feedback_section}
IMPORTANT: Generate a UNIQUE question.
- DO NOT use history of previous questions
- The question must be new and original

PROHIBITED (STRICTLY PROHIBITED):
- DO NOT generate questions about code syntax (how to write a command)
- DO NOT generate questions about specific function/method names
- DO NOT generate questions about SQL commands (SELECT, INSERT, DROP, CREATE, etc.)
- DO NOT generate questions about technical code details
- DO NOT generate questions about configuration settings
- DO NOT generate questions about variables, parameters, arguments
- DO NOT generate questions about database structure (tables, columns, indexes)
- DO NOT generate questions about API endpoints, HTTP methods, status codes

IF THE QUESTION CONTAINS CODE, SYNTAX, TECHNICAL DETAILS - DO NOT GENERATE IT!

Focus ONLY on:
- Architectural principles (how the system works as a whole)
- Conceptual understanding (theoretical foundations)
- Component interactions (which modules interact)
- Metrics and KPI (which metrics are important)
- Business processes (which business problems are solved)
- Technology choices (why choose one technology over another)
- Design patterns (which patterns are used)

Generate a NEW multiple-choice question with:
- A clear question stem
- 4 options (A, B, C, D)
- Exactly 1 correct answer (unambiguous)
- Plausible distractors that are not obviously wrong
- Options grammatically consistent with the question stem
- The question must be answerable from the source text

Return your response in this JSON format:
{{
    "question_stem": "The question text",
    "options": {{
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
    }},
    "correct_answer": "A",
    "explanation": "Brief explanation of why this is correct",
    "bloom_level": "Remember"
}}

Blooms Taxonomy levels:
- Remember: remembering facts
- Understand: understanding concepts
- Apply: applying knowledge
- Analyze: analyzing information
- Evaluate: evaluating information
- Create: creating something new
"""

            response = self.generator_client.chat.completions.create(
                model=Config.GENERATOR_MODEL,
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            generated_question = GeneratedQuestion(**result)

            # НЕ оновлюємо історію - кожен агент має чисту сесію
            # history = state.get("generated_questions_history", [])
            # history.append({
            #     "question_stem": generated_question.question_stem,
            #     "correct_answer": generated_question.correct_answer,
            #     "bloom_level": generated_question.bloom_level
            # })
            # state["generated_questions_history"] = history

            state["generated_question"] = generated_question
            state["current_agent"] = "Generator"
            state["status"] = f"Question generated: {generated_question.question_stem[:60]}..."

            return state
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                state["error_message"] = "API Error: Authentication failed. Check DEEPSEEK_API_KEY."
            elif "402" in error_msg or "Payment" in error_msg or "quota" in error_msg.lower():
                state["error_message"] = "API Error: Insufficient credits or quota exceeded."
            else:
                state["error_message"] = f"Generator error: {error_msg}"
            
            state["status"] = "Generation failed"
            state["generated_question"] = GeneratedQuestion()
            return state
    
    def critic_node(self, state: AgentState) -> AgentState:
        """
        Critic Node: Verifies question quality and approves/rejects
        """
        try:
            source_text = truncate_source_text(state["source_text"])
            generated_question = state["generated_question"]
            original_question = state.get("original_question")
            original_options = state.get("original_options")
            
            if not generated_question or not generated_question.question_stem:
                state["error_message"] = "Cannot verify: No question generated"
                state["status"] = "Verification failed"
                return state
            
            # Test mode - simulate iteration: reject first attempt, approve on retry
            if self.test_mode:
                iteration_count = state.get("iteration_count", 0)
                if iteration_count < 2:
                    verification = VerificationResult(
                        unambiguity=3,
                        distractor_adequacy=3,
                        grammatical_consistency=4,
                        feedback="Option B is too obvious; replace it with a more scientific term from the text. "
                                 "Ensure only one answer is clearly correct.",
                        issues=["Distractor B is obviously wrong", "Two options could be considered correct"],
                        approved=False,
                        rejection_reason="Scores below threshold (4): Unambiguity=3, Distractor adequacy=3"
                    )
                    state["status"] = (
                        f"Question REJECTED (TEST MODE). Scores: "
                        f"Unambiguity={verification.unambiguity}/5, "
                        f"Distractors={verification.distractor_adequacy}/5, "
                        f"Grammar={verification.grammatical_consistency}/5. "
                        f"Sending feedback to Generator..."
                    )
                else:
                    verification = VerificationResult(
                        unambiguity=5,
                        distractor_adequacy=5,
                        grammatical_consistency=5,
                        feedback="",
                        issues=[],
                        approved=True,
                        rejection_reason=""
                    )
                    state["status"] = (
                        f"Question APPROVED (TEST MODE). Scores: "
                        f"Unambiguity={verification.unambiguity}/5, "
                        f"Distractors={verification.distractor_adequacy}/5, "
                        f"Grammar={verification.grammatical_consistency}/5"
                    )
                state["verification"] = verification
                state["current_agent"] = "Critic"
                return state
            
            if original_question:
                original_text = f"Original Question: {original_question}\nOptions: {original_options}"
            else:
                original_text = "(No original question provided for comparison)"

            if UKRAINIAN_MODE:
                # Використовуємо критичний промпт з 4 критеріями і бінарними оцінками 0/1
                prompt = CRITIC_PROMPT_UA.format(
                    source_text=source_text,
                    question_stem=generated_question.question_stem,
                    option_a=generated_question.options.get("A", ""),
                    option_b=generated_question.options.get("B", ""),
                    option_c=generated_question.options.get("C", ""),
                    option_d=generated_question.options.get("D", ""),
                    correct_answer=generated_question.correct_answer,
                    bloom_level=generated_question.bloom_level,
                    original_text=original_text
                )
            else:
                # Використовуємо критичний промпт з 4 критеріями і бінарними оцінками 0/1
                prompt = CRITIC_PROMPT_EN.format(
                    source_text=source_text,
                    question_stem=generated_question.question_stem,
                    option_a=generated_question.options.get("A", ""),
                    option_b=generated_question.options.get("B", ""),
                    option_c=generated_question.options.get("C", ""),
                    option_d=generated_question.options.get("D", ""),
                    correct_answer=generated_question.correct_answer,
                    bloom_level=generated_question.bloom_level,
                    original_text=original_text
                )

            response = self.critic_client.chat.completions.create(
                model=Config.VERIFICATION_MODEL,
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Зменшена температура для більшої точності
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            verification = build_verification_result(result)

            state["verification"] = verification
            state["current_agent"] = "Critic"

            if verification.approved:
                state["status"] = (
                    f"Question APPROVED. Scores: "
                    f"Unambiguity={verification.unambiguity}/5, "
                    f"Distractors={verification.distractor_adequacy}/5, "
                    f"Grammar={verification.grammatical_consistency}/5"
                )
            else:
                state["status"] = (
                    f"Question REJECTED. Scores: "
                    f"Unambiguity={verification.unambiguity}/5, "
                    f"Distractors={verification.distractor_adequacy}/5, "
                    f"Grammar={verification.grammatical_consistency}/5. "
                    f"{verification.rejection_reason} Sending feedback to Generator..."
                )
            
            return state
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                state["error_message"] = "API Error: Authentication failed. Check OPENAI_API_KEY."
            elif "402" in error_msg or "Payment" in error_msg or "quota" in error_msg.lower():
                state["error_message"] = "API Error: Insufficient credits or quota exceeded."
            else:
                state["error_message"] = f"Critic error: {error_msg}"
            
            state["status"] = "Verification failed"
            state["verification"] = VerificationResult()
            return state