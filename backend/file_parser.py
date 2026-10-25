"""
Universal File Parser for Multi-Agent Test Generation System
Supports multiple file formats: PDF, TXT, DOCX, CSV, Parquet, JSON
"""

from pypdf import PdfReader
import pandas as pd
import json
import io
from typing import Optional, Dict, Any, List
from docx import Document

from text_utils import CONTEXT_COLUMN_CANDIDATES, truncate_source_text


class FileParser:
    """Universal parser for multiple file formats"""
    
    @staticmethod
    def parse_pdf(file_content: bytes) -> str:
        """Parse PDF file"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    @staticmethod
    def parse_txt(file_content: bytes) -> str:
        """Parse plain text file"""
        try:
            return file_content.decode('utf-8').strip()
        except Exception as e:
            raise ValueError(f"Failed to parse TXT: {str(e)}")
    
    @staticmethod
    def parse_docx(file_content: bytes) -> str:
        """Parse Word document (.docx)"""
        try:
            doc_file = io.BytesIO(file_content)
            doc = Document(doc_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX: {str(e)}")
    
    @staticmethod
    def _extract_row_chunks(df: pd.DataFrame) -> List[str]:
        """Extract one source passage per dataset row (SciQ/RACE-style)."""
        lower_columns = {col.lower(): col for col in df.columns}
        context_col = next(
            (lower_columns[name] for name in CONTEXT_COLUMN_CANDIDATES if name in lower_columns),
            None,
        )

        chunks: List[str] = []
        if context_col:
            for value in df[context_col].astype(str):
                cleaned = value.strip()
                if cleaned and cleaned.lower() not in {"nan", "none", ""}:
                    chunks.append(cleaned)
            if chunks:
                return chunks

        for _, row in df.iterrows():
            parts = [
                str(value).strip()
                for value in row.values
                if str(value).strip().lower() not in {"nan", "none", ""}
            ]
            if parts:
                chunks.append(" | ".join(parts))

        return chunks

    @staticmethod
    def parse_tabular(file_content: bytes, reader) -> Dict[str, Any]:
        """Parse CSV/Parquet into per-row chunks plus preview text."""
        df = reader(file_content)
        original_rows = len(df)

        if len(df) > 1000:
            df = df.head(1000)

        chunks = FileParser._extract_row_chunks(df)
        preview_text = truncate_source_text(chunks[0] if chunks else "", max_chars=500)

        return {
            "text": preview_text,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "rows": len(df),
            "original_rows": original_rows,
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "preview": df.head(5).to_string(),
            "truncated": original_rows > 1000,
        }

    @staticmethod
    def parse_csv(file_content: bytes) -> Dict[str, Any]:
        """Parse CSV file - returns extracted text and metadata"""
        try:
            def _read(content: bytes) -> pd.DataFrame:
                return pd.read_csv(io.BytesIO(content))

            return FileParser.parse_tabular(file_content, _read)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {str(e)}")

    @staticmethod
    def parse_parquet(file_content: bytes) -> Dict[str, Any]:
        """Parse Parquet file - returns extracted text and metadata"""
        try:
            def _read(content: bytes) -> pd.DataFrame:
                return pd.read_parquet(io.BytesIO(content))

            return FileParser.parse_tabular(file_content, _read)
        except Exception as e:
            raise ValueError(f"Failed to parse Parquet: {str(e)}")
    
    @staticmethod
    def parse_json(file_content: bytes) -> str:
        """Parse JSON file - extract text values"""
        try:
            data = json.loads(file_content.decode('utf-8'))
            
            def extract_text(obj, result=""):
                if isinstance(obj, dict):
                    for value in obj.values():
                        result = extract_text(value, result)
                elif isinstance(obj, list):
                    for item in obj:
                        result = extract_text(item, result)
                elif isinstance(obj, str):
                    result += obj + " "
                return result
            
            return extract_text(data).strip()
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")
    
    @staticmethod
    def parse_file(file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Universal file parser that detects format and extracts text
        
        Args:
            file_content: Raw bytes of the file
            filename: Name of the file with extension
        
        Returns:
            Dictionary with extracted text and metadata
        """
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        parsers = {
            'pdf': FileParser.parse_pdf,
            'txt': FileParser.parse_txt,
            'docx': FileParser.parse_docx,
            'csv': FileParser.parse_csv,
            'parquet': FileParser.parse_parquet,
            'json': FileParser.parse_json
        }
        
        if file_ext not in parsers:
            raise ValueError(f"Unsupported file format: .{file_ext}. Supported formats: {', '.join(parsers.keys())}")
        
        result = parsers[file_ext](file_content)
        
        # Normalize result format
        if isinstance(result, str):
            return {
                "text": truncate_source_text(result),
                "chunks": [truncate_source_text(result)] if result.strip() else [],
                "chunk_count": 1 if result.strip() else 0,
                "format": file_ext,
                "rows": None,
                "columns": None,
                "column_names": None,
                "preview": None,
                "truncated": len(result) > 8000,
            }

        result["format"] = file_ext
        result.setdefault("chunks", [])
        result.setdefault("chunk_count", len(result["chunks"]))
        return result