"""
Simple SQLite database client (alternative to Prisma)
Direct SQLite implementation without Prisma dependency
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class SimpleDatabase:
    """Simple SQLite database wrapper"""
    
    def __init__(self, db_path: str = "test_generation.db"):
        self.db_path = db_path
        self._init_db()
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursor"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database tables"""
        with self._get_cursor() as cursor:
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Generated tests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generated_tests (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    original_question TEXT,
                    original_options TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            
            # Questions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    test_id TEXT NOT NULL,
                    question_stem TEXT NOT NULL,
                    options TEXT NOT NULL,
                    correct_answer TEXT NOT NULL,
                    explanation TEXT,
                    analysis TEXT,
                    verification TEXT,
                    quality_score INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES generated_tests(id)
                )
            """)
    
    def create_document(self, source_text: str) -> str:
        """Create a new document"""
        doc_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with self._get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO documents (id, source_text, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (doc_id, source_text, now, now)
            )
        
        return doc_id
    
    def create_generated_test(self, document_id: str, original_question: str = None, 
                            original_options: List[str] = None) -> str:
        """Create a new generated test"""
        test_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        options_json = json.dumps(original_options) if original_options else None
        
        with self._get_cursor() as cursor:
            cursor.execute(
                """INSERT INTO generated_tests 
                   (id, document_id, original_question, original_options, created_at, updated_at) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (test_id, document_id, original_question, options_json, now, now)
            )
        
        return test_id
    
    def create_question(self, test_id: str, question_stem: str, options: Dict[str, str],
                      correct_answer: str, explanation: str = None, analysis: Dict = None,
                      verification: Dict = None, quality_score: int = None) -> str:
        """Create a new question"""
        question_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        options_json = json.dumps(options)
        analysis_json = json.dumps(analysis) if analysis else None
        verification_json = json.dumps(verification) if verification else None
        
        with self._get_cursor() as cursor:
            cursor.execute(
                """INSERT INTO questions 
                   (id, test_id, question_stem, options, correct_answer, explanation, 
                    analysis, verification, quality_score, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (question_id, test_id, question_stem, options_json, correct_answer, 
                 explanation, analysis_json, verification_json, quality_score, now)
            )
        
        return question_id
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID"""
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
    
    def get_generated_test(self, test_id: str) -> Optional[Dict]:
        """Get generated test by ID"""
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM generated_tests WHERE id = ?", (test_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result['original_options']:
                    result['original_options'] = json.loads(result['original_options'])
                return result
        return None
    
    def get_questions_by_test(self, test_id: str) -> List[Dict]:
        """Get all questions for a test"""
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM questions WHERE test_id = ?", (test_id,))
            rows = cursor.fetchall()
            questions = []
            for row in rows:
                result = dict(row)
                # Parse JSON fields
                if result['options']:
                    result['options'] = json.loads(result['options'])
                if result['analysis']:
                    result['analysis'] = json.loads(result['analysis'])
                if result['verification']:
                    result['verification'] = json.loads(result['verification'])
                questions.append(result)
            return questions
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


# Global database instance
db = SimpleDatabase()