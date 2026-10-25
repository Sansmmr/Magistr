"""Live end-to-end test with real API keys (requires .env)."""

import asyncio
import json
import os
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_dir = os.path.dirname(backend_dir)
sys.path.insert(0, backend_dir)
sys.path.insert(0, project_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_dir, ".env"), override=True)

# Force OpenAI for live E2E when DeepSeek balance is unavailable.
for key, value in {
    "ANALYST_MODEL": "gpt-4o-mini",
    "GENERATOR_MODEL": "gpt-4o-mini",
    "VERIFICATION_MODEL": "gpt-4o-mini",
    "ANALYST_BASE_URL": "https://api.openai.com/v1",
    "GENERATOR_BASE_URL": "https://api.openai.com/v1",
    "VERIFICATION_BASE_URL": "https://api.openai.com/v1",
}.items():
    os.environ[key] = value

import importlib
import config as config_module
importlib.reload(config_module)
from config import Config

import langgraph_nodes
import langgraph_graph
importlib.reload(langgraph_nodes)
importlib.reload(langgraph_graph)
from langgraph_graph import run_test_generation_graph

SAMPLE_TEXT = """
Mitochondria are organelles found in most eukaryotic cells. They are often called
the powerhouse of the cell because they produce ATP through cellular respiration.
The inner membrane of mitochondria contains folds called cristae, which increase
surface area for energy production. Mitochondria have their own circular DNA,
supporting the endosymbiotic theory that they evolved from ancient bacteria.
"""


async def run_live_test():
    Config.validate()

    result = await run_test_generation_graph(
        source_text=SAMPLE_TEXT,
        max_iterations=3,
        question_count=1,
        test_mode=False,
    )

    final = result["final_result"]
    assert final["questions"], "No questions generated"

    question = final["questions"][0]
    verification = question["verification"]
    generated = question["generated_question"]

    assert generated["question_stem"], "Empty question stem"
    assert len(generated["options"]) == 4, "Expected 4 options"
    assert generated["correct_answer"] in {"A", "B", "C", "D"}

    for key in ("unambiguity", "distractor_adequacy", "grammatical_consistency"):
        score = verification[key]
        assert 1 <= score <= 5, f"Invalid score for {key}: {score}"

    print("LIVE E2E TEST PASSED")
    print(json.dumps({
        "approved": verification["approved"],
        "iteration_count": question["iteration_count"],
        "scores": {
            "unambiguity": verification["unambiguity"],
            "distractor_adequacy": verification["distractor_adequacy"],
            "grammatical_consistency": verification["grammatical_consistency"],
        },
        "question_stem": generated["question_stem"],
        "correct_answer": generated["correct_answer"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run_live_test())
