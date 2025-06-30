import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from statsmodels.formula.api import ols

class StatisticalAnalyzer:
    """Perform statistical analysis on metadiscourse marker data."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame containing marker data."""
        self.df = df
        self.marker_columns = [col for col in df.columns 
                             if col.endswith('_count') or col.endswith('_freq')]
    
    def perform_anova(self, marker_col: str, group_col: str) -> Dict[str, Any]:
        """Perform one-way ANOVA test for a marker across groups."""
        # Prepare data
        groups = self.df[group_col].unique()
        group_data = [self.df[self.df[group_col] == group][marker_col] 
                     for group in groups]
        
        # Perform ANOVA
        f_statistic, p_value = stats.f_oneway(*group_data)
        
        # Calculate effect size (eta-squared)
        model = ols(f'{marker_col} ~ C({group_col})', data=self.df).fit()
        eta_squared = model.rsquared
        
        return {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'groups': groups,
            'group_means': {group: data.mean() for group, data in zip(groups, group_data)}
        }
    
    def perform_post_hoc(self, marker_col: str, group_col: str) -> pd.DataFrame:
        """Perform Tukey's HSD post-hoc test after ANOVA."""
        mc = MultiComparison(self.df[marker_col], self.df[group_col])
        tukey = mc.tukeyhsd()
        
        # Convert to DataFrame for easier handling
        results = pd.DataFrame({
            'group1': tukey.groups[:, 0],
            'group2': tukey.groups[:, 1],
            'mean_diff': tukey.meandiffs,
            'p_value': tukey.pvalues,
            'lower': tukey.confint[:, 0],
            'upper': tukey.confint[:, 1],
            'reject': tukey.reject
        })
        
        return results
    
    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlations between all marker frequencies."""
        freq_columns = [col for col in self.marker_columns if col.endswith('_freq')]
        return self.df[freq_columns].corr()
    
    def perform_regression(self, marker_col: str, predictor_cols: List[str]) -> Dict[str, Any]:
        """Perform multiple regression analysis."""
        X = self.df[predictor_cols]
        y = self.df[marker_col]
        
        # Add constant to predictors
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        return {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'p_value': model.f_pvalue,
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict()
        }
    
    def analyze_marker_distribution(self, marker_col: str) -> Dict[str, Any]:
        """Analyze the distribution of a marker across documents."""
        data = self.df[marker_col]
        
        # Basic statistics
        stats_dict = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'min': data.min(),
            'max': data.max(),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75)
        }
        
        # Test for normality
        _, p_value = stats.normaltest(data)
        stats_dict['is_normal'] = p_value > 0.05
        
        return stats_dict
    
    def analyze_marker_patterns(self) -> Dict[str, Any]:
        """Analyze patterns and relationships between markers."""
        patterns = {
            'correlations': self.calculate_correlations(),
            'marker_distributions': {
                col: self.analyze_marker_distribution(col)
                for col in self.marker_columns
            }
        }
        
        # Calculate marker co-occurrence
        freq_columns = [col for col in self.marker_columns if col.endswith('_freq')]
        cooccurrence = np.zeros((len(freq_columns), len(freq_columns)))
        
        for i, col1 in enumerate(freq_columns):
            for j, col2 in enumerate(freq_columns):
                if i != j:
                    cooccurrence[i, j] = np.sum(
                        (self.df[col1] > self.df[col1].mean()) & 
                        (self.df[col2] > self.df[col2].mean())
                    )
        
        patterns['cooccurrence'] = pd.DataFrame(
            cooccurrence,
            index=freq_columns,
            columns=freq_columns
        )
        
        return patterns
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report of all analyses."""
        report = {
            'basic_stats': {
                'total_documents': len(self.df),
                'marker_columns': self.marker_columns,
                'correlations': self.calculate_correlations()
            },
            'marker_analyses': {
                col: self.analyze_marker_distribution(col)
                for col in self.marker_columns
            },
            'patterns': self.analyze_marker_patterns()
        }
        
        return report 