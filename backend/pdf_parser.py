"""
PDF Parser for Multi-Agent Test Generation System
Extracts text from PDF files using pypdf
"""

from pypdf import PdfReader
from typing import Optional
import io


class PDFParser:
    """Parser for extracting text from PDF files"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """
        Extract text from PDF file content
        
        Args:
            file_content: Raw bytes of the PDF file
        
        Returns:
            Extracted text as string
        """
        try:
            # Create a PDF reader from the file content
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_pdf_file(file_path: str) -> str:
        """
        Extract text from PDF file path
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            Extracted text as string
        """
        try:
            pdf_reader = PdfReader(file_path)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    @staticmethod
    def get_pdf_page_count(file_content: bytes) -> int:
        """
        Get the number of pages in a PDF
        
        Args:
            file_content: Raw bytes of the PDF file
        
        Returns:
            Number of pages
        """
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            return len(pdf_reader.pages)
        except Exception:
            return 0