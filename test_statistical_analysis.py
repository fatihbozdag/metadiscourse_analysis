import pandas as pd
import numpy as np
import os
import sys

# Import functions from our main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyze_metalanguage_fixed import generate_latex_tables, perform_language_group_statistics

# Create a sample dataframe with language groups
def create_sample_data():
    # Create sample data with three language groups
    data = {
        'Native_Language': ['English', 'English', 'English', 'English', 'English',
                          'French', 'French', 'French', 'French', 'French',
                          'German', 'German', 'German', 'German', 'German'],
        'word_count': [2500, 3000, 2800, 2700, 2900, 2600, 3100, 2900, 2800, 3000, 2700, 3200, 2950, 3100, 2850],
        'interactive_total': [150, 170, 160, 155, 165, 180, 195, 185, 190, 188, 130, 140, 135, 138, 132],
        'interactional_total': [280, 300, 290, 285, 295, 210, 230, 220, 215, 225, 310, 330, 320, 325, 315],
    }
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Calculate densities (per 1000 words)
    df['interactive_density'] = df['interactive_total'] / df['word_count'] * 1000
    df['interactional_density'] = df['interactional_total'] / df['word_count'] * 1000
    
    # Add log and sqrt transformations
    df['interactive_density_log'] = np.log(df['interactive_density'])
    df['interactive_density_sqrt'] = np.sqrt(df['interactive_density'])
    df['interactional_density_log'] = np.log(df['interactional_density'])
    df['interactional_density_sqrt'] = np.sqrt(df['interactional_density'])
    
    # Add entropy measures (simulated)
    df['interactive_entropy'] = np.random.uniform(1.2, 1.8, len(df))
    df['interactional_entropy'] = np.random.uniform(1.5, 2.1, len(df))
    
    # Add some marker categories
    # Interactive markers
    df['interactive_transitions'] = np.random.uniform(15, 25, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 1.2, 'German': 0.8})
    df['interactive_frame_markers'] = np.random.uniform(8, 12, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 1.1, 'German': 0.9})
    df['interactive_endophoric_markers'] = np.random.uniform(5, 10, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 0.9, 'German': 1.1})
    df['interactive_evidentials'] = np.random.uniform(10, 15, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 1.3, 'German': 0.7})
    df['interactive_code_glosses'] = np.random.uniform(12, 18, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 1.1, 'German': 0.9})
    
    # Interactional markers
    df['interactional_hedges'] = np.random.uniform(20, 30, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 0.8, 'German': 1.2})
    df['interactional_boosters'] = np.random.uniform(15, 25, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 0.7, 'German': 1.3})
    df['interactional_attitude_markers'] = np.random.uniform(10, 20, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 0.9, 'German': 1.1})
    df['interactional_self_mentions'] = np.random.uniform(5, 15, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 1.2, 'German': 0.8})
    df['interactional_engagement_markers'] = np.random.uniform(15, 25, len(df)) * df['Native_Language'].map({'English': 1.0, 'French': 0.8, 'German': 1.2})
    
    return df

# Main function
if __name__ == "__main__":
    # Create output directory
    output_dir = "test_statistical_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data()
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_tables(df, output_dir)
    
    # Perform statistical tests
    print("Performing statistical tests...")
    perform_language_group_statistics(df, output_dir)
    
    print("\nTest complete! Check the output directory for results:")
    print(f"  {os.path.abspath(output_dir)}")
