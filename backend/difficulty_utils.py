"""Difficulty settings — prompt-only (no extra API calls)."""

DIFFICULTY_LEVELS = ("easy", "medium", "hard")


def normalize_difficulty(value: str) -> str:
    difficulty = (value or "medium").lower().strip()
    return difficulty if difficulty in DIFFICULTY_LEVELS else "medium"


def max_iterations_for_difficulty(difficulty: str) -> int:
    """Keep retries low to save time; easy gets fewer attempts."""
    # Регламент v2: refinement loop — максимум 5 ітерацій для всіх рівнів складності.
    # Для аналізу чутливості можна змінити через змінну середовища MAX_REFINE_ITERATIONS.
    import os
    try:
        return max(1, int(os.getenv("MAX_REFINE_ITERATIONS", "5")))
    except ValueError:
        return 5


def generator_difficulty_instruction(difficulty: str) -> str:
    difficulty = normalize_difficulty(difficulty)
    instructions = {
        "easy": (
            "Difficulty: EASY. Ask a straightforward factual question with one clear correct answer. "
            "Distractors may be clearly wrong but must stay grammatically consistent."
        ),
        "medium": (
            "Difficulty: MEDIUM. Ask a question that requires understanding, not just keyword matching. "
            "All distractors must be plausible to a student who read the text casually."
        ),
        "hard": (
            "Difficulty: HARD. Target subtle distinctions from the text. "
            "Use plausible near-miss distractors from the same topic (similar terms, related concepts). "
            "Avoid trivia; require careful reading."
        ),
    }
    return instructions[difficulty]


def critic_difficulty_instruction(difficulty: str) -> str:
    difficulty = normalize_difficulty(difficulty)
    instructions = {
        "easy": (
            "Difficulty target: EASY. Accept clear, direct questions. "
            "Distractors need not be highly subtle if the correct answer is unambiguous."
        ),
        "medium": (
            "Difficulty target: MEDIUM. Distractors should be plausible; reject only if obviously absurd."
        ),
        "hard": (
            "Difficulty target: HARD. Expect challenging but fair distractors. "
            "Reject if distractors are trivially wrong for a hard item, but do not reject solely for being difficult."
        ),
    }
    return instructions[difficulty]
