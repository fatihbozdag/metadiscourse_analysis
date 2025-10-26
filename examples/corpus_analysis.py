"""
Corpus Analysis Example for Metalinguistics Library

This script demonstrates batch processing and corpus-level analysis
using the Metalinguistics library.
"""

import pandas as pd
from pathlib import Path
from metalinguistics.analyzers import EnhancedMetadiscourseAnalyzer


def analyze_corpus(texts_df: pd.DataFrame, text_column: str = 'text',
                   id_column: str = 'doc_id') -> pd.DataFrame:
    """
    Analyze a corpus of texts for metadiscourse markers.

    Parameters
    ----------
    texts_df : pd.DataFrame
        DataFrame containing texts to analyze
    text_column : str
        Name of column containing text data
    id_column : str
        Name of column containing document IDs

    Returns
    -------
    pd.DataFrame
        Results with one row per detected marker
    """
    analyzer = EnhancedMetadiscourseAnalyzer()
    results = []

    print(f"Analyzing {len(texts_df)} documents...")

    for idx, row in texts_df.iterrows():
        doc_id = row[id_column]
        text = row[text_column]

        # Analyze text
        analysis = analyzer.analyze_text(text)

        # Extract markers
        for marker in analysis['markers']:
            results.append({
                'doc_id': doc_id,
                'marker_text': marker.text,
                'category': marker.category,
                'confidence': marker.confidence,
                'start_pos': marker.start_char,
                'end_pos': marker.end_char
            })

        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(texts_df)} documents...")

    print("Analysis complete!\n")
    return pd.DataFrame(results)


def calculate_corpus_statistics(results_df: pd.DataFrame,
                                texts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate corpus-level statistics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Marker detection results
    texts_df : pd.DataFrame
        Original texts

    Returns
    -------
    pd.DataFrame
        Category-level statistics
    """
    # Calculate category frequencies
    category_stats = results_df.groupby('category').agg({
        'marker_text': 'count',
        'confidence': 'mean'
    }).rename(columns={
        'marker_text': 'frequency',
        'confidence': 'avg_confidence'
    })

    # Calculate rate per 1000 words
    total_words = sum(len(text.split()) for text in texts_df['text'])
    category_stats['rate_per_1k'] = (category_stats['frequency'] / total_words) * 1000

    # Sort by frequency
    category_stats = category_stats.sort_values('frequency', ascending=False)

    return category_stats


def main():
    """Run corpus analysis example."""

    print("=" * 70)
    print("CORPUS ANALYSIS EXAMPLE")
    print("=" * 70 + "\n")

    # Create sample corpus
    sample_corpus = pd.DataFrame({
        'doc_id': ['DOC001', 'DOC002', 'DOC003', 'DOC004', 'DOC005'],
        'text': [
            """This study examines the effectiveness of online learning. According to
            recent research, student engagement is crucial. However, our findings
            reveal new patterns. Clearly, the evidence supports our hypothesis.""",

            """First, we review the theoretical framework. The literature demonstrates
            several key points. Moreover, our analysis shows significant correlations.
            In conclusion, these results have important implications.""",

            """Perhaps the most notable finding relates to student motivation. We argue
            that intrinsic factors matter more. It seems that external rewards may
            actually decrease engagement. Our data clearly indicate this trend.""",

            """According to Smith (2020), teaching methods vary widely. For example,
            some approaches emphasize collaboration. However, we found that individual
            work also benefits students. Note that context influences these outcomes.""",

            """This research investigates learning strategies. Obviously, metacognitive
            skills play a role. We suggest that explicit instruction helps. Moreover,
            the evidence shows sustained improvement over time. Clearly, this matters."""
        ]
    })

    print("Sample Corpus:")
    print(f"  Total documents: {len(sample_corpus)}")
    print(f"  Total words: {sum(len(text.split()) for text in sample_corpus['text'])}\n")

    # Analyze corpus
    results_df = analyze_corpus(sample_corpus)

    print(f"Detection Results:")
    print(f"  Total markers detected: {len(results_df)}")
    print(f"  Unique categories: {results_df['category'].nunique()}\n")

    # Calculate statistics
    stats = calculate_corpus_statistics(results_df, sample_corpus)

    print("Category Statistics:")
    print(stats.to_string())
    print()

    # Document-level summary
    doc_summary = results_df.groupby('doc_id').agg({
        'marker_text': 'count',
        'confidence': 'mean'
    }).rename(columns={
        'marker_text': 'total_markers',
        'confidence': 'avg_confidence'
    })

    print("\nDocument-Level Summary:")
    print(doc_summary.to_string())
    print()

    # Most common markers
    print("\nMost Common Markers:")
    marker_counts = results_df['marker_text'].value_counts().head(10)
    for marker, count in marker_counts.items():
        print(f"  '{marker}': {count} occurrences")

    # Save results (optional)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(output_dir / "corpus_markers.csv", index=False)
    stats.to_csv(output_dir / "corpus_statistics.csv")
    doc_summary.to_csv(output_dir / "document_summary.csv")

    print(f"\n✓ Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
