"""Main script for evidentiality and metadiscourse marker extraction from TICLE and LOCNESS corpora."""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from evidentiality_processor import EvidentialityProcessor

def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract evidentiality and metadiscourse markers from TICLE and LOCNESS corpora.')
    parser.add_argument('--ticle_path', type=str, required=True,
                      help='Path to the TICLE corpus file (CSV format)')
    parser.add_argument('--locness_path', type=str, required=False,
                      help='Path to the LOCNESS corpus file (CSV format)')
    parser.add_argument('--output_dir', type=str, default='../results',
                      help='Directory for saving results')
    parser.add_argument('--model', type=str, default='en_core_web_trf',
                      help='spaCy model to use')
    parser.add_argument('--text_field', type=str, default='text',
                      help='Name of the column containing text to analyze')
    args = parser.parse_args()
    
    # Initialize processor
    print("Initializing processor...")
    processor = EvidentialityProcessor(model_name=args.model)
    
    # Process corpora
    print(f"Processing TICLE and LOCNESS corpora...")
    results_df = processor.process_corpus(
        ticle_path=args.ticle_path,
        locness_path=args.locness_path if args.locness_path else None,
        text_field=args.text_field
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save full extraction results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extraction_path = os.path.join(args.output_dir, f'marker_extraction_{timestamp}.csv')
    results_df.to_csv(extraction_path, index=False)
    print(f"Full extraction results saved to: {extraction_path}")
    
    # Calculate statistics
    print("Calculating statistics...")
    stats = processor.calculate_statistics(results_df)
    
    # Save statistics
    for stat_name, stat_df in stats.items():
        stat_path = os.path.join(args.output_dir, f'{stat_name}_{timestamp}.csv')
        stat_df.to_csv(stat_path)
        print(f"{stat_name} statistics saved to: {stat_path}")
    
    # Generate basic visualizations
    print("Generating visualizations...")
    generate_visualizations(results_df, stats, args.output_dir, timestamp)
    
    print("Analysis complete!")

def generate_visualizations(results_df, stats, output_dir, timestamp):
    """Generate basic visualizations from the results."""
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Distribution of marker categories by corpus
    plt.figure(figsize=(12, 8))
    category_counts = results_df.groupby(['corpus_source', 'marker_category']).size().reset_index(name='count')
    sns.barplot(x='marker_category', y='count', hue='corpus_source', data=category_counts)
    plt.title('Distribution of Marker Categories by Corpus')
    plt.xlabel('Marker Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'category_distribution_{timestamp}.png'))
    
    # 2. Distribution of markers across essay positions
    plt.figure(figsize=(12, 8))
    position_dist = stats['position_distribution']
    sns.barplot(x='essay_position', y='count', hue='corpus_source', data=position_dist)
    plt.title('Distribution of Markers by Essay Position')
    plt.xlabel('Essay Position')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'position_distribution_{timestamp}.png'))
    
    # 3. Comparison of normalized frequencies between corpora
    if 'pivot_norm' in stats:
        pivot_norm = stats['pivot_norm']
        if 'TICLE' in pivot_norm.columns and 'LOCNESS' in pivot_norm.columns:
            # Filter for top categories for readability
            top_categories = pivot_norm.sum(axis=1).nlargest(15).index
            top_pivot = pivot_norm.loc[top_categories]
            
            plt.figure(figsize=(14, 10))
            top_pivot[['TICLE', 'LOCNESS']].plot(kind='bar', figsize=(14, 10))
            plt.title('Normalized Frequencies (per 10,000 words) - TICLE vs LOCNESS')
            plt.xlabel('Marker Category')
            plt.ylabel('Frequency per 10,000 words')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'normalized_comparison_{timestamp}.png'))
    
    # 4. Heatmap of marker subcategories
    subcategory_counts = results_df.groupby(['corpus_source', 'marker_subcategory']).size().unstack(fill_value=0)
    plt.figure(figsize=(16, 12))
    sns.heatmap(subcategory_counts, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Heatmap of Marker Subcategories by Corpus')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'subcategory_heatmap_{timestamp}.png'))

if __name__ == "__main__":
    main()
