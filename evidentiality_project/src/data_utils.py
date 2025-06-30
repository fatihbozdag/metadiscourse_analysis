"""Utility functions for data preparation and corpus handling."""

import os
import re
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import csv

def preprocess_corpus_file(file_path: str, output_path: str, corpus_type: str, text_column: str = None) -> str:
    """Preprocess a corpus file and convert it to a standardized CSV format.
    
    Args:
        file_path: Path to the corpus file
        output_path: Path to save the processed CSV
        corpus_type: Either 'TICLE' or 'LOCNESS'
        text_column: Name of the column containing text (if already in CSV format)
        
    Returns:
        Path to the processed CSV file
    """
    # Determine file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if ext == '.csv':
        # If already CSV, just standardize the format
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        # If text column is specified, ensure it exists
        if text_column and text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV. Available columns: {', '.join(df.columns)}")
        
        # If text column is not specified, try to find it
        if not text_column:
            # Look for common text column names
            common_names = ['text', 'content', 'essay', 'essay_text', 'body', 'document']
            for name in common_names:
                if name in df.columns:
                    text_column = name
                    break
            
            if not text_column:
                # If still not found, use the column with the longest average string length
                text_column = df.select_dtypes(include=['object']).columns[0]
                for col in df.select_dtypes(include=['object']).columns:
                    if df[col].str.len().mean() > df[text_column].str.len().mean():
                        text_column = col
        
        # Add corpus source column if not exists
        if 'corpus_source' not in df.columns:
            df['corpus_source'] = corpus_type
        
        # Add essay_id if not exists
        if 'essay_id' not in df.columns:
            df['essay_id'] = [f"{corpus_type}_{i}" for i in range(len(df))]
        
        # Rename text column to 'text' if different
        if text_column != 'text':
            df = df.rename(columns={text_column: 'text'})
        
        # Save standardized CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        
    elif ext in ['.txt', '.text']:
        # For plain text files, we need to parse them
        
        # Determine if it's a single essay or multiple essays
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Check if it contains multiple essays (look for common separators)
        essay_separators = [
            r'\n\s*={3,}\s*\n',  # ===== separator
            r'\n\s*-{3,}\s*\n',  # ----- separator
            r'\n\s*\*{3,}\s*\n', # ***** separator
            r'\n\s*#{3,}\s*\n',  # ##### separator
            r'\n\s*Essay \d+\s*\n', # Essay 1, Essay 2, etc.
            r'\n\s*Text \d+\s*\n'   # Text 1, Text 2, etc.
        ]
        
        essays = []
        
        # Try each separator pattern
        for separator in essay_separators:
            split_content = re.split(separator, content)
            if len(split_content) > 1:
                essays = [essay.strip() for essay in split_content if essay.strip()]
                break
        
        # If no separator worked, treat as a single essay
        if not essays:
            essays = [content]
        
        # Create a DataFrame
        df = pd.DataFrame({
            'essay_id': [f"{corpus_type}_{i}" for i in range(len(essays))],
            'text': essays,
            'corpus_source': corpus_type
        })
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please provide a .csv, .txt, or .text file.")
    
    return output_path

def estimate_word_counts(df: pd.DataFrame, text_column: str = 'text') -> Dict[str, int]:
    """Estimate word counts for each corpus in the DataFrame.
    
    Args:
        df: DataFrame with corpus data
        text_column: Name of the column containing text
        
    Returns:
        Dictionary with corpus names as keys and word counts as values
    """
    word_counts = {}
    
    # Group by corpus source
    grouped = df.groupby('corpus_source')
    
    for corpus, group in grouped:
        # Calculate total word count
        total_words = 0
        for text in group[text_column]:
            if isinstance(text, str):
                # Split by whitespace and count
                words = text.split()
                total_words += len(words)
        
        word_counts[corpus] = total_words
    
    return word_counts

def detect_essay_structure(text: str) -> Dict[str, Tuple[int, int]]:
    """Attempt to detect essay structure (introduction, body, conclusion).
    
    Args:
        text: Essay text
        
    Returns:
        Dictionary with section names as keys and (start_index, end_index) as values
    """
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Remove empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return {
            'introduction': (0, 0),
            'body': (0, 0),
            'conclusion': (0, 0)
        }
    
    # Simple heuristic: first paragraph is intro, last is conclusion, rest is body
    intro_end = len(paragraphs[0])
    
    # Calculate body start and end
    body_start = intro_end
    body_end = len(text) - len(paragraphs[-1])
    
    # Calculate conclusion start
    conclusion_start = body_end
    
    return {
        'introduction': (0, intro_end),
        'body': (body_start, body_end),
        'conclusion': (conclusion_start, len(text))
    }

def merge_extraction_results(results_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple extraction result DataFrames.
    
    Args:
        results_list: List of DataFrames with extraction results
        
    Returns:
        Merged DataFrame
    """
    if not results_list:
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    merged = pd.concat(results_list, ignore_index=True)
    
    # Remove duplicates (same marker in same position in same essay)
    merged = merged.drop_duplicates(subset=['essay_id', 'marker', 'position_in_text'])
    
    return merged
