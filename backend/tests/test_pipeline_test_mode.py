"""Integration test for LangGraph pipeline in test mode."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_graph import run_test_generation_graph


SAMPLE_TEXT = """
Photosynthesis is the process by which green plants convert light energy into chemical energy.
Chlorophyll in chloroplasts absorbs sunlight. The process produces glucose and releases oxygen.
Plants use carbon dioxide and water as raw materials during photosynthesis.
"""


async def run_pipeline():
    result = await run_test_generation_graph(
        source_text=SAMPLE_TEXT,
        max_iterations=3,
        question_count=1,
        test_mode=True,
    )
    final = result["final_result"]
    assert final["question_count"] == 1
    assert len(final["questions"]) == 1

    question = final["questions"][0]
    verification = question["verification"]

    assert question["generated_question"]["question_stem"]
    assert verification["approved"] is True
    assert verification["unambiguity"] == 5
    assert verification["distractor_adequacy"] == 5
    assert verification["grammatical_consistency"] == 5
    assert question["iteration_count"] == 2


def test_pipeline_test_mode():
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    test_pipeline_test_mode()
    print("Pipeline test_mode test passed.")
