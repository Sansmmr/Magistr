"""
Configuration module for Multi-Agent Test Generation System
Supports modular switching between OpenAI and DeepSeek models
"""

import os
from dotenv import load_dotenv

# Force reload of environment variables
load_dotenv(override=True)

class Config:
    """Configuration class for API keys and model selection"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    
    # Model Configuration
    # Generator model: Use DeepSeek for cost-effective high-volume generation
    GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "deepseek-chat")
    GENERATOR_BASE_URL = os.getenv("GENERATOR_BASE_URL", "https://api.deepseek.com")
    
    # Analyst model: Use DeepSeek for cost-effective analysis
    ANALYST_MODEL = os.getenv("ANALYST_MODEL", "deepseek-chat")
    ANALYST_BASE_URL = os.getenv("ANALYST_BASE_URL", "https://api.deepseek.com")
    
    # Verification model: Use GPT-4 for complex verification tasks
    VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", "gpt-4o-mini")
    VERIFICATION_BASE_URL = os.getenv("VERIFICATION_BASE_URL", "https://api.openai.com/v1")
    
    # Dataset Configuration
    RACE_DATASET_PATH = os.getenv("RACE_DATASET_PATH", r"d:\magistr\race\all\train-00000-of-00001.parquet")
    SCIQ_DATASET_PATH = os.getenv("SCIQ_DATASET_PATH", r"d:\magistr\sciq\data\train-00000-of-00001.parquet")
    
    # Sample size for testing
    SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "5"))
    
    # Output files
    OUTPUT_JSON = os.getenv("OUTPUT_JSON", "final_verified_benchmark.json")
    OUTPUT_REPORT = os.getenv("OUTPUT_REPORT", "verification_report.txt")
    
    # Language Configuration
    LANGUAGE = os.getenv("LANGUAGE", "ukrainian")  # Options: "ukrainian", "english"
    LANGUAGE_CODE = "uk" if LANGUAGE == "ukrainian" else "en"
    
    @classmethod
    def validate(cls):
        """Validate that required API keys are present"""
        if not cls.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set in environment")
        if not cls.DEEPSEEK_API_KEY:
            print("Warning: DEEPSEEK_API_KEY not set in environment")
        
        # At least one API key should be present
        if not cls.OPENAI_API_KEY and not cls.DEEPSEEK_API_KEY:
            raise ValueError("At least one API key (OPENAI_API_KEY or DEEPSEEK_API_KEY) must be set")
