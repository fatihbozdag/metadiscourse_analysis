import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
df = pd.read_csv('results/metadiscourse_analysis.csv')

# List of marker columns (counts and frequencies)
marker_cols = [
    'interactive_transitions_count', 'interactive_frame_markers_count', 'interactive_endophoric_markers_count',
    'interactive_evidentials_count', 'interactive_code_glosses_count',
    'interactional_hedges_count', 'interactional_boosters_count', 'interactional_attitude_markers_count',
    'interactional_engagement_markers_count', 'interactional_self_mentions_count'
]
marker_freq_cols = [
    'interactive_transitions_freq', 'interactive_frame_markers_freq', 'interactive_endophoric_markers_freq',
    'interactive_evidentials_freq', 'interactive_code_glosses_freq',
    'interactional_hedges_freq', 'interactional_boosters_freq', 'interactional_attitude_markers_freq',
    'interactional_engagement_markers_freq', 'interactional_self_mentions_freq'
]

# Group by Native_Language (L1)
grouped = df.groupby('Native_Language')[marker_cols + marker_freq_cols].mean().round(2)

# Academic Table (Markdown)
def to_markdown_table(df):
    md = df.reset_index().to_markdown(index=False)
    return md

# Academic Table (LaTeX)
def to_latex_table(df):
    return df.reset_index().to_latex(index=False, float_format="%.2f")

# Save tables
with open('results/l1_marker_table.md', 'w') as f:
    f.write(to_markdown_table(grouped))

with open('results/l1_marker_table.tex', 'w') as f:
    f.write(to_latex_table(grouped))

# Optional: L1-wise bar plot (sum of all marker counts)
grouped_counts = grouped[marker_cols].sum(axis=1)
plt.figure(figsize=(10, 6))
grouped_counts.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Total Metadiscourse Marker Counts by L1')
plt.ylabel('Mean Marker Count per Document')
plt.xlabel('Native Language (L1)')
plt.tight_layout()
plt.savefig('results/l1_marker_counts_barplot.png')
plt.close()

# 1. Overall Distribution Table (sum of all marker counts)
overall_counts = df[marker_cols].sum().astype(int)
overall_table = overall_counts.reset_index()
overall_table.columns = ['Marker', 'Total Count']
overall_table_md = overall_table.to_markdown(index=False)
overall_table_tex = overall_table.to_latex(index=False)
with open('results/overall_marker_distribution.md', 'w') as f:
    f.write(overall_table_md)
with open('results/overall_marker_distribution.tex', 'w') as f:
    f.write(overall_table_tex)

# 2. Radar Chart for Relative Frequencies (mean by category)
category_freqs = {
    'Transitions': df['interactive_transitions_freq'].mean(),
    'Frame Markers': df['interactive_frame_markers_freq'].mean(),
    'Endophoric Markers': df['interactive_endophoric_markers_freq'].mean(),
    'Evidentials': df['interactive_evidentials_freq'].mean(),
    'Code Glosses': df['interactive_code_glosses_freq'].mean(),
    'Hedges': df['interactional_hedges_freq'].mean(),
    'Boosters': df['interactional_boosters_freq'].mean(),
    'Attitude Markers': df['interactional_attitude_markers_freq'].mean(),
    'Engagement Markers': df['interactional_engagement_markers_freq'].mean(),
    'Self Mentions': df['interactional_self_mentions_freq'].mean(),
}
categories = list(category_freqs.keys())
values = list(category_freqs.values())
values += values[:1]  # close the loop
angles = np.linspace(0, 2 * np.pi, len(categories) + 1, endpoint=True)
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, linewidth=2, linestyle='solid')
ax.fill(angles, values, 'skyblue', alpha=0.4)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
plt.title('Relative Frequencies of Metadiscourse Marker Categories', size=14)
plt.tight_layout()
plt.savefig('results/marker_categories_radar.png')
plt.close()

# 3. Descriptive Statistics for Interactive & Interactional Markers
def sum_category_counts(row, cols):
    return row[cols].sum()
interactive_cols = [c for c in marker_cols if c.startswith('interactive_')]
interactional_cols = [c for c in marker_cols if c.startswith('interactional_')]
df['Interactive_Total'] = df[interactive_cols].sum(axis=1)
df['Interactional_Total'] = df[interactional_cols].sum(axis=1)
desc_stats = df[['Interactive_Total', 'Interactional_Total']].describe().T.round(2)
desc_stats_md = desc_stats.to_markdown()
desc_stats_tex = desc_stats.to_latex()
with open('results/interactive_interactional_desc_stats.md', 'w') as f:
    f.write(desc_stats_md)
with open('results/interactive_interactional_desc_stats.tex', 'w') as f:
    f.write(desc_stats_tex)

print("Academic tables (Markdown & LaTeX), L1-wise bar plot, overall distribution table, radar chart, and descriptive statistics have been generated in the results/ directory.")
