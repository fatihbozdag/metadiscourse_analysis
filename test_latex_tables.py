import pandas as pd
import os
import numpy as np

# Create a sample dataframe with language groups
def create_sample_data():
    # Create sample data
    data = {
        'Native_Language': ['English', 'English', 'English', 'French', 'French', 'French', 'German', 'German', 'German'],
        'interactive_density': [35.2, 40.1, 38.5, 42.3, 45.8, 41.2, 30.5, 32.1, 31.8],
        'interactional_density': [55.3, 58.1, 56.7, 40.2, 42.5, 41.8, 60.3, 62.1, 61.5],
        'interactive_entropy': [1.45, 1.52, 1.48, 1.62, 1.58, 1.60, 1.30, 1.35, 1.32],
        'interactional_entropy': [1.75, 1.80, 1.78, 1.65, 1.70, 1.68, 1.85, 1.90, 1.88],
        'word_count': [2500, 3000, 2800, 2700, 3200, 2900, 2600, 3100, 2850]
    }
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Add some marker categories
    df['interactive_transitions'] = [15.2, 16.5, 15.8, 18.3, 19.5, 18.9, 12.5, 13.2, 12.8]
    df['interactive_frame_markers'] = [8.3, 9.1, 8.7, 10.5, 11.2, 10.8, 7.2, 7.8, 7.5]
    df['interactional_hedges'] = [20.5, 21.3, 20.9, 15.2, 16.5, 15.8, 25.3, 26.1, 25.7]
    df['interactional_boosters'] = [18.2, 19.5, 18.8, 12.5, 13.8, 13.1, 20.5, 21.3, 20.9]
    
    return df

# Function to generate LaTeX tables for academic publication
def generate_latex_tables(df, output_dir='test_results'):
    """
    Generate LaTeX tables for academic publication from the analysis results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing analysis results
    output_dir : str, optional
        Directory to save the LaTeX tables
        
    Returns:
    --------
    None
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # LaTeX preamble for standalone tables
        latex_preamble = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"  # For professional tables with proper spacing
            "\\usepackage{siunitx}\n"   # For proper alignment of numbers
            "\\usepackage{caption}\n"   # For better captions
            "\\usepackage{threeparttable}\n"  # For table notes
            "\\usepackage[table]{xcolor}\n"  # For table colors
            "\\usepackage{multirow}\n"  # For merged rows
            "\\begin{document}\n"
        )
        
        # LaTeX closing
        latex_closing = "\\end{document}"
        
        # Table 1: Summary statistics by native language
        if 'Native_Language' in df.columns and len(df['Native_Language'].unique()) > 1:
            print("Generating language summary table...")
            language_summary = df.groupby('Native_Language')[
                ['interactive_density', 'interactional_density', 'interactive_entropy', 'interactional_entropy']
            ].agg(['mean', 'std'])
            
            # Format the table with proper column names
            language_summary.columns = [
                f"{col[0].split('_')[0].capitalize()} {col[0].split('_')[1].capitalize()} ({col[1].capitalize()})"
                for col in language_summary.columns
            ]
            
            # Generate LaTeX table with booktabs style
            latex_table = language_summary.to_latex(float_format="%.2f", escape=False)
            
            # Enhance the LaTeX table with proper formatting
            enhanced_table = (
                "\\begin{table}[htbp]\n"
                "\\centering\n"
                "\\caption{Summary Statistics of Metadiscourse Markers by Native Language}\n"
                "\\label{tab:language_summary}\n"
                "\\begin{threeparttable}\n"
                + latex_table +
                "\\begin{tablenotes}\n"
                "\\small\n"
                "\\item Note: Values represent means and standard deviations of marker densities (per 1000 words) "
                "and entropy measures across different native language groups.\n"
                "\\end{tablenotes}\n"
                "\\end{threeparttable}\n"
                "\\end{table}\n"
            )
            
            # Save to file - both standalone and embedded versions
            with open(f"{output_dir}/language_summary_table.tex", "w") as f:
                f.write(latex_table)
                
            with open(f"{output_dir}/language_summary_table_full.tex", "w") as f:
                f.write(latex_preamble + enhanced_table + latex_closing)
            
            print(f"Language summary table saved to {output_dir}/language_summary_table_full.tex")
        
        # Create a combined LaTeX file with all tables
        print("Creating combined LaTeX file with all tables...")
        combined_latex = latex_preamble
        
        # Add table part
        full_file = f"{output_dir}/language_summary_table_full.tex"
        if os.path.exists(full_file):
            with open(full_file, 'r') as f:
                content = f.read()
                # Extract just the table part (between \begin{document} and \end{document})
                if '\\begin{document}\n' in content and '\\end{document}' in content:
                    table_part = content.split('\\begin{document}\n')[1].split('\\end{document}')[0]
                    combined_latex += table_part
        
        combined_latex += latex_closing
        
        # Save combined file
        with open(f"{output_dir}/all_tables.tex", "w") as f:
            f.write(combined_latex)
        
        print(f"Enhanced LaTeX tables have been saved to the '{output_dir}' directory")
    except Exception as e:
        print(f"Error generating LaTeX tables: {str(e)}")
        import traceback
        traceback.print_exc()

# Main function
if __name__ == "__main__":
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data()
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_tables(df)
    
    print("Done!")
