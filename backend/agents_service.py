"""
Agent service for test generation with WebSocket support
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Callable
from openai import OpenAI
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class AgentService:
    """Service for running multi-agent test generation with real-time logging"""
    
    def __init__(self, log_callback: Callable = None):
        self.log_callback = log_callback
        self.analyst_client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.ANALYST_BASE_URL
        )
        self.generator_client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.GENERATOR_BASE_URL
        )
        self.critic_client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.VERIFICATION_BASE_URL
        )
    
    async def log(self, agent: str, message: str):
        """Send log message via callback"""
        if self.log_callback:
            await self.log_callback({
                "type": "log",
                "agent": agent,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
    
    def convert_to_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable types"""
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            return list(obj)
        return obj
    
    async def analyze_text(self, source_text: str) -> Dict[str, Any]:
        """Extract key facts, concepts, and relationships from source text"""
        await self.log("Analyst", "Analyzing source text...")
        
        prompt = f"""Analyze the following text and extract key information suitable for creating test questions.

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

        try:
            response = self.analyst_client.chat.completions.create(
                model=Config.ANALYST_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert text analyst who extracts key facts and concepts from educational texts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            await self.log("Analyst", f"Analysis completed. Found {len(result.get('key_facts', []))} key facts.")
            return result
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "402" in error_msg:
                await self.log("Analyst", f"API Error: Authentication or billing issue. Check API keys.")
            else:
                await self.log("Analyst", f"Error: {error_msg}")
            return {
                "key_facts": [],
                "concepts": [],
                "relationships": [],
                "summary": "Analysis failed"
            }
    
    async def generate_question(self, source_text: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new multiple-choice question based on source text and analysis"""
        await self.log("Generator", "Generating new question...")
        
        prompt = f"""Based on the following source text and analysis, generate a completely new multiple-choice question.

Source Text:
{source_text}

Analysis:
Key Facts: {analysis.get('key_facts', [])}
Concepts: {analysis.get('concepts', [])}
Relationships: {analysis.get('relationships', [])}
Summary: {analysis.get('summary', '')}

Generate a NEW multiple-choice question with:
- A clear question stem
- 4 options (A, B, C, D)
- Exactly 1 correct answer
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
    "explanation": "Brief explanation of why this is correct"
}}"""

        try:
            response = self.generator_client.chat.completions.create(
                model=Config.GENERATOR_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert test question generator for educational content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            await self.log("Generator", f"Question generated: {result.get('question_stem', '')[:60]}...")
            return result
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "402" in error_msg:
                await self.log("Generator", f"API Error: Authentication or billing issue. Check API keys.")
            else:
                await self.log("Generator", f"Error: {error_msg}")
            return {
                "question_stem": "",
                "options": {"A": "", "B": "", "C": "", "D": ""},
                "correct_answer": "",
                "explanation": "Generation failed"
            }
    
    async def verify_question(self, source_text: str, generated_question: Dict[str, Any], 
                            original_question: str = None, original_options: list = None) -> Dict[str, Any]:
        """Perform self-verification and benchmark verification"""
        await self.log("Critic", "Verifying question...")
        
        original_text = f"""
Original Question: {original_question}
Options: {original_options}
""" if original_question else "(No original question provided for comparison)"
        
        prompt = f"""You are a verification critic. Evaluate the following generated question.

Source Text:
{source_text}

Generated Question:
Question: {generated_question.get('question_stem', '')}
Options:
A: {generated_question.get('options', {}).get('A', '')}
B: {generated_question.get('options', {}).get('B', '')}
C: {generated_question.get('options', {}).get('C', '')}
D: {generated_question.get('options', {}).get('D', '')}
Correct Answer: {generated_question.get('correct_answer', '')}

{original_text}

Perform TWO evaluations:

1. Self-Verification of Generated Question:
   - Is the question logically consistent? (yes/no)
   - Is the answer clearly derivable from the source text? (yes/no)
   - Is the formatting correct? (yes/no)
   - Are the options plausible and distinct? (yes/no)
   - Overall quality score (1-10)

2. Benchmark Comparison (if original provided):
   - Difficulty comparison (easier/similar/harder)
   - Quality comparison (better/similar/worse)
   - Style comparison (more formal/similar/more casual)
   - Key differences in approach

Return your response in this JSON format:
{{
    "self_verification": {{
        "logically_consistent": "yes/no",
        "answerable_from_text": "yes/no",
        "formatting_correct": "yes/no",
        "options_distinct": "yes/no",
        "quality_score": 1-10,
        "issues": ["any issues found"]
    }},
    "benchmark_comparison": {{
        "difficulty": "easier/similar/harder",
        "quality": "better/similar/worse",
        "style": "more formal/similar/more casual",
        "key_differences": "description of key differences"
    }}
}}"""

        try:
            response = self.critic_client.chat.completions.create(
                model=Config.VERIFICATION_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert educational assessment evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            quality_score = result['self_verification'].get('quality_score', 0)
            await self.log("Critic", f"Verification completed. Quality score: {quality_score}/10")
            return result
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "402" in error_msg:
                await self.log("Critic", f"API Error: Authentication or billing issue. Check API keys.")
            else:
                await self.log("Critic", f"Error: {error_msg}")
            return {
                "self_verification": {
                    "logically_consistent": "no",
                    "answerable_from_text": "no",
                    "formatting_correct": "no",
                    "options_distinct": "no",
                    "quality_score": 1,
                    "issues": ["Verification failed"]
                },
                "benchmark_comparison": {
                    "difficulty": "unknown",
                    "quality": "unknown",
                    "style": "unknown",
                    "key_differences": "Verification failed"
                }
            }
    
    async def generate_test(self, source_text: str, original_question: str = None, 
                           original_options: list = None) -> Dict[str, Any]:
        """Run full pipeline: Analyst -> Generator -> Critic"""
        await self.log("System", "Starting test generation pipeline...")
        
        # Step 1: Analyze
        analysis = await self.analyze_text(source_text)
        
        # Step 2: Generate
        generated_question = await self.generate_question(source_text, analysis)
        
        # Step 3: Verify
        verification = await self.verify_question(
            source_text, 
            generated_question, 
            original_question, 
            original_options
        )
        
        result = {
            "source_text": source_text,
            "original_question": original_question,
            "original_options": original_options,
            "analysis": analysis,
            "generated_question": generated_question,
            "verification": verification,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.log("System", "Pipeline completed successfully")
        
        return result
