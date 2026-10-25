"""
LangGraph State and TypedDict for Multi-Agent Test Generation System
Defines the shared state structure that flows between agents
"""

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Result from Analyst agent"""
    key_facts: List[str] = Field(default_factory=list)
    key_terms: List[str] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    summary: str = ""


class GeneratedQuestion(BaseModel):
    """Generated MCQ question from Generator agent"""
    question_stem: str = ""
    options: Dict[str, str] = Field(default_factory=lambda: {"A": "", "B": "", "C": "", "D": ""})
    correct_answer: str = ""
    explanation: str = ""
    bloom_level: str = ""  # Added Bloom level


class VerificationResult(BaseModel):
    """Verification result from Critic agent (LLM-as-a-Judge)"""
    unambiguity: int = 1
    distractor_adequacy: int = 1
    grammatical_consistency: int = 1
    bloom_level: int = 1  # Added Bloom level score
    feedback: str = ""
    issues: List[str] = Field(default_factory=list)
    approved: bool = False
    rejection_reason: str = ""


class AgentState(TypedDict):
    """
    Shared state that flows between LangGraph agents
    """
    # Input data
    source_text: str
    original_question: Optional[str]
    original_options: Optional[List[str]]

    # Analysis result (from Analyst)
    analysis: Optional[AnalysisResult]

    # Generated question (from Generator)
    generated_question: Optional[GeneratedQuestion]

    # Verification result (from Critic)
    verification: Optional[VerificationResult]

    # Control flow
    iteration_count: int
    max_iterations: int
    difficulty: str
    current_agent: str

    # Status for WebSocket broadcasting
    status: str
    error_message: Optional[str]

    # Final result
    final_result: Optional[Dict[str, Any]]

    # History of generated questions (to avoid duplicates)
    generated_questions_history: List[Dict[str, Any]]

    # Configuration (M1, M2, M3, M4) - for different instructions
    configuration: str
    task_spec: dict
    mcq: dict
    critic_report: dict
    critic_history: list
    h3_status: str
    target_bloom_level: int  # 1=Remember, 2=Understand, 3=Apply (цільовий рівень, задається викладачем)
