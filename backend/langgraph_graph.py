"""
LangGraph Pipeline for Multi-Agent Test Generation System
Builds the graph with conditional edges and iteration control
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph_state import AgentState
from langgraph_nodes import LangGraphAgents
from text_utils import pick_source_for_question
from difficulty_utils import normalize_difficulty, max_iterations_for_difficulty


def _model_to_dict(model):
    """Serialize Pydantic v1/v2 models consistently."""
    if model is None:
        return None
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def should_continue(state: AgentState) -> Literal["refine", "end"]:
    """Після Критика (M4): якщо відхилено і ще є спроби — локальна регенерація, інакше кінець."""
    rep = state.get("critic_report") or {}
    if rep.get("verdict") == "APPROVED":
        return "end"
    if int(state.get("iteration_count") or 1) >= int(state.get("max_iterations") or 5):
        return "end"
    return "refine"


def build_test_generation_graph(log_callback=None, test_mode=False, configuration="M4"):
    """
    Конфігурації за регламентом ExplAI-2026 (див. backend/generator.py):
    - M1: монолітний zero-shot запит (без Аналітика і Критика)
    - M2: Генератор (без Task Spec) -> Критик, одна перевірка
    - M3: Аналітик -> Генератор (без Критика)
    - M4: Аналітик -> Генератор -> Критик -> локальна регенерація (макс. 5 ітерацій)
    Якість оцінює незалежний арбітр (backend/arbiter.py) поза пайплайном.
    """
    import generator as P
    wf = StateGraph(AgentState)
    if configuration == "M1":
        wf.add_node("m1", P.m1_node); wf.set_entry_point("m1"); wf.add_edge("m1", END)
    elif configuration == "M2":
        wf.add_node("generator", P.generator_node); wf.add_node("critic", P.critic_node)
        wf.set_entry_point("generator"); wf.add_edge("generator", "critic"); wf.add_edge("critic", END)
    elif configuration == "M3":
        wf.add_node("analyst", P.analyst_node); wf.add_node("generator", P.generator_node)
        wf.set_entry_point("analyst"); wf.add_edge("analyst", "generator"); wf.add_edge("generator", END)
    else:  # M4
        wf.add_node("analyst", P.analyst_node); wf.add_node("generator", P.generator_node)
        wf.add_node("critic", P.critic_node); wf.add_node("refine", P.refine_node)
        wf.set_entry_point("analyst"); wf.add_edge("analyst", "generator"); wf.add_edge("generator", "critic")
        wf.add_conditional_edges("critic", should_continue, {"refine": "refine", "end": END})
        wf.add_edge("refine", "critic")
    return wf.compile()


async def run_test_generation_graph(
    source_text: str,
    source_chunks: list = None,
    original_question: str = None,
    original_options: list = None,
    max_iterations: int = None,
    question_count: int = 1,
    difficulty: str = "medium",
    log_callback=None,
    test_mode=False,
    configuration: str = "M4",
    target_level: int = 1,
    preset_mcq: dict = None,
    preset_task_spec: dict = None
) -> dict:
    """
    Runs the LangGraph pipeline for test generation

    Args:
        source_text: The text to generate questions from
        original_question: Optional original question for benchmark comparison
        original_options: Optional original options for benchmark comparison
        max_iterations: Maximum number of generation attempts
        question_count: Number of questions to generate
        log_callback: Optional callback for WebSocket logging
        test_mode: If True, use mock responses instead of real API calls
        configuration: M1 (single pass), M2 (standard), M3 (limited), M4 (full)

    Returns:
        Final state with generated questions and verification results
    """
    # Build the graph
    app = build_test_generation_graph(log_callback, test_mode, configuration)
    
    difficulty = normalize_difficulty(difficulty)
    resolved_max_iterations = max_iterations or max_iterations_for_difficulty(difficulty)

    # Send initial log
    if log_callback:
        await log_callback({
            "type": "log",
            "agent": "System",
            "message": (
                f"Starting test generation pipeline for {question_count} question(s) "
                f"({difficulty} difficulty, max {resolved_max_iterations} iterations per question)..."
            ),
            "timestamp": "now"
        })
    
    all_results = []
    # НЕ використовуємо спільну історію - кожен агент має чисту сесію
    # generated_questions_history = []
    
    for question_num in range(1, question_count + 1):
        if log_callback:
            await log_callback({
                "type": "log",
                "agent": "System",
                "message": f"Generating question {question_num} of {question_count}...",
                "timestamp": "now"
            })
        
        question_source = pick_source_for_question(
            source_text=source_text,
            question_index=question_num - 1,
            source_chunks=source_chunks,
        )

        if log_callback and source_chunks:
            await log_callback({
                "type": "log",
                "agent": "System",
                "message": (
                    f"Using dataset row {(question_num - 1) % len(source_chunks) + 1} "
                    f"({len(question_source)} chars) for question {question_num}."
                ),
                "timestamp": "now",
                "question_number": question_num,
            })

        # Initialize state for this question
        initial_state: AgentState = {
            "source_text": question_source,
            "original_question": original_question,
            "original_options": original_options,
            "analysis": None,
            "generated_question": None,
            "verification": None,
            "iteration_count": 0,
            "max_iterations": resolved_max_iterations,
            "difficulty": difficulty,
            "current_agent": "System",
            "status": f"Initializing question {question_num}...",
            "error_message": None,
            "final_result": None,
            "generated_questions_history": [],  # Чиста сесія - порожня історія
            "configuration": configuration,
            "target_bloom_level": target_level,
            "mcq": preset_mcq,
            "task_spec": preset_task_spec
        }
        
        # Run the graph for this question
        final_state = None
        async for event in app.astream(initial_state):
            for node_name, node_output in event.items():
                final_state = node_output

                # Send status update via WebSocket
                if log_callback:
                    await log_callback({
                        "type": "log",
                        "agent": final_state.get("current_agent", node_name),
                        "message": final_state.get("status", ""),
                        "timestamp": "now",
                        "iteration": final_state.get("iteration_count", 0),
                        "question_number": question_num
                    })
        
        generated = final_state.get("generated_question") if final_state else None
        has_question = bool(generated and generated.question_stem)

        if final_state and has_question:
            question_result = {
                "question_number": question_num,
                "source_text": final_state["source_text"],
                "original_question": final_state.get("original_question"),
                "original_options": final_state.get("original_options"),
                "analysis": _model_to_dict(final_state.get("analysis")),
                "generated_question": _model_to_dict(final_state.get("generated_question")),
                "verification": _model_to_dict(final_state.get("verification")),
                "iteration_count": final_state.get("iteration_count", 0),
                "configuration": configuration,
                "mcq": final_state.get("mcq"),
                "task_spec": final_state.get("task_spec"),
                "critic_history": final_state.get("critic_history"),
                "h3_status": final_state.get("h3_status"),
                "target_bloom_level": target_level,
                "difficulty": difficulty,
                "approved": final_state.get("verification").approved if final_state.get("verification") else False,
                "error": final_state.get("error_message")
            }
            all_results.append(question_result)

            # НЕ оновлюємо спільну історію - кожен агент має чисту сесію
            # generated_questions_history.append({
            #     "question_stem": generated.question_stem,
            #     "correct_answer": generated.correct_answer,
            #     "bloom_level": generated.bloom_level
            # })
    
    # Prepare final result with all questions
    final_result = {
        "source_text": source_text,
        "chunk_count": len(source_chunks) if source_chunks else 0,
        "question_count": question_count,
        "difficulty": difficulty,
        "max_iterations": resolved_max_iterations,
        "questions": all_results,
        "total_iterations": sum(q["iteration_count"] for q in all_results),
        "approved_count": sum(1 for q in all_results if q["approved"]),
        "generated_count": len(all_results),
    }
    
    if log_callback:
        await log_callback({
            "type": "log",
            "agent": "System",
            "message": f"Generation completed! {final_result['approved_count']}/{question_count} questions approved.",
            "timestamp": "now"
        })
    
    return {"final_result": final_result}