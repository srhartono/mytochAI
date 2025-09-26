"""
Correlation Analysis Engine for SNP-Methylation associations.

"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    Main class for analyzing correlations between SNPs and DNA methylation.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.correlation_methods = config['analysis']['correlation']['methods']
        self.p_value_threshold = config['analysis']['correlation']['p_value_threshold']
        self.fdr_method = config['analysis']['correlation']['fdr_correction']
        self.window_size = config['analysis']['window_analysis']['window_size']
        self.step_size = config['analysis']['window_analysis']['step_size']
        
        # Initialize analyzers
        self.statistical_analyzer = StatisticalCorrelationAnalyzer(config)
        self.ml_analyzer = MLCorrelationAnalyzer(config)
        self.window_analyzer = WindowBasedAnalyzer(config)
    
    def run_comprehensive_analysis(self, snp_data: pd.DataFrame, 
                                 methylation_data: pd.DataFrame) -> Dict:
        """
        Run comprehensive correlation analysis using all available methods.
        
        Args:
            snp_data: SNP genotype data
            methylation_data: DNA methylation beta values
            
        Returns:
            Dictionary containing results from all analysis methods
        """
        results = {
            'statistical': {},
            'machine_learning': {},
            'window_based': {},
            'summary': {}
        }
        
        logger.info("Starting comprehensive correlation analysis...")
        
        # Statistical correlation analysis
        logger.info("Running statistical correlation analysis...")
        results['statistical'] = self.statistical_analyzer.analyze_correlations(
            snp_data, methylation_data
        )
        
        # Machine learning-based analysis
        logger.info("Running machine learning correlation analysis...")
        results['machine_learning'] = self.ml_analyzer.analyze_ml_associations(
            snp_data, methylation_data
        )
        
        # Window-based analysis
        logger.info("Running window-based analysis...")
        results['window_based'] = self.window_analyzer.analyze_genomic_windows(
            snp_data, methylation_data
        )
        
        # Generate summary
        logger.info("Generating analysis summary...")
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate a summary of all correlation analysis results."""
        summary = {
            'total_snps_analyzed': 0,
            'total_cpg_sites_analyzed': 0,
            'significant_associations': 0,
            'top_associations': [],
            'method_performance': {}
        }
        
        # Count significant associations from different methods
        if 'pairwise_correlations' in results['statistical']:
            pairwise = results['statistical']['pairwise_correlations']
            if isinstance(pairwise, pd.DataFrame) and 'padj' in pairwise.columns:
                significant = pairwise[pairwise['padj'] < self.p_value_threshold]
                summary['significant_associations'] += len(significant)
                
                # Get top associations
                if len(significant) > 0:
                    top_associations = significant.nlargest(10, 'abs_correlation')
                    summary['top_associations'] = top_associations.to_dict('records')
        
        # Add ML performance metrics
        if 'feature_importance' in results['machine_learning']:
            ml_results = results['machine_learning']['feature_importance']
            if isinstance(ml_results, dict) and 'r2_scores' in ml_results:
                summary['method_performance']['ml_r2_mean'] = np.mean(ml_results['r2_scores'])
        
        return summary


class StatisticalCorrelationAnalyzer:
    """
    Statistical methods for correlation analysis between SNPs and methylation.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.methods = config['analysis']['correlation']['methods']
        self.p_threshold = config['analysis']['correlation']['p_value_threshold']
        self.fdr_method = config['analysis']['correlation']['fdr_correction']
    
    def analyze_correlations(self, snp_data: pd.DataFrame, 
                           methylation_data: pd.DataFrame) -> Dict:
        """
        Perform statistical correlation analysis between SNPs and methylation sites.
        
        Args:
            snp_data: SNP genotype data
            methylation_data: DNA methylation data
            
        Returns:
            Dictionary with correlation analysis results
        """
        results = {}
        
        # Extract and align data
        snp_matrix, meth_matrix, sample_names = self._prepare_matrices(snp_data, methylation_data)
        
        if snp_matrix is None or meth_matrix is None:
            return {'error': 'Could not align SNP and methylation data'}
        
        logger.info(f"Analyzing correlations for {snp_matrix.shape[1]} SNPs and {meth_matrix.shape[1]} CpG sites")
        
        # Pairwise correlation analysis
        results['pairwise_correlations'] = self._pairwise_correlation_analysis(
            snp_matrix, meth_matrix, snp_data, methylation_data
        )
        
        # Global correlation patterns
        results['global_patterns'] = self._global_correlation_patterns(
            snp_matrix, meth_matrix
        )
        
        # Distance-based correlation analysis
        results['distance_correlations'] = self._distance_correlation_analysis(
            snp_matrix, meth_matrix, snp_data
        )
        
        return results
    
    def _prepare_matrices(self, snp_data: pd.DataFrame, 
                         methylation_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare aligned matrices for correlation analysis."""
        
        # Extract genotype columns
        genotype_cols = [col for col in snp_data.columns if col.startswith('genotype_')]
        if not genotype_cols:
            logger.error("No genotype columns found in SNP data")
            return None, None, []
        
        # Extract sample names
        snp_samples = [col.replace('genotype_', '') for col in genotype_cols]
        meth_samples = methylation_data.columns.tolist()
        
        # Find common samples
        common_samples = list(set(snp_samples) & set(meth_samples))
        if len(common_samples) < 3:
            logger.error(f"Insufficient common samples: {len(common_samples)}")
            return None, None, []
        
        # Create aligned matrices
        snp_matrix = snp_data[[f'genotype_{sample}' for sample in common_samples]].values.T
        meth_matrix = methylation_data[common_samples].values.T
        
        # Handle missing values
        snp_matrix = np.nan_to_num(snp_matrix, nan=np.nanmean(snp_matrix))
        meth_matrix = np.nan_to_num(meth_matrix, nan=np.nanmean(meth_matrix))
        
        return snp_matrix, meth_matrix, common_samples
    
    def _pairwise_correlation_analysis(self, snp_matrix: np.ndarray, meth_matrix: np.ndarray,
                                     snp_data: pd.DataFrame, methylation_data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform pairwise correlation analysis between all SNP-CpG pairs.
        """
        correlations = []
        n_snps = snp_matrix.shape[1]
        n_cpgs = meth_matrix.shape[1]
        
        # Limit analysis for computational efficiency
        max_pairs = 10000
        step_snp = max(1, n_snps // 100) if n_snps * n_cpgs > max_pairs else 1
        step_cpg = max(1, n_cpgs // 100) if n_snps * n_cpgs > max_pairs else 1
        
        logger.info(f"Computing correlations for {n_snps//step_snp} × {n_cpgs//step_cpg} pairs")
        
        snp_indices = range(0, n_snps, step_snp)
        cpg_indices = range(0, n_cpgs, step_cpg)
        
        for i, snp_idx in enumerate(snp_indices):
            if i % 50 == 0:
                logger.info(f"Processing SNP {i+1}/{len(snp_indices)}")
            
            snp_genotypes = snp_matrix[:, snp_idx]
            snp_id = snp_data.iloc[snp_idx]['snp_id']
            
            for cpg_idx in cpg_indices:
                cpg_methylation = meth_matrix[:, cpg_idx]
                cpg_id = methylation_data.index[cpg_idx]
                
                # Calculate correlations using different methods
                correlation_results = {}
                
                for method in self.methods:
                    try:
                        if method == 'pearson':
                            corr, p_val = stats.pearsonr(snp_genotypes, cpg_methylation)
                        elif method == 'spearman':
                            corr, p_val = stats.spearmanr(snp_genotypes, cpg_methylation)
                        elif method == 'kendall':
                            corr, p_val = stats.kendalltau(snp_genotypes, cpg_methylation)
                        else:
                            continue
                        
                        correlation_results[f'{method}_corr'] = corr
                        correlation_results[f'{method}_pval'] = p_val
                        
                    except Exception as e:
                        logger.warning(f"Correlation calculation failed for {method}: {e}")
                        correlation_results[f'{method}_corr'] = np.nan
                        correlation_results[f'{method}_pval'] = 1.0
                
                # Store result
                result = {
                    'snp_id': snp_id,
                    'cpg_id': cpg_id,
                    'snp_chr': snp_data.iloc[snp_idx]['chr'],
                    'snp_pos': snp_data.iloc[snp_idx]['pos'],
                    **correlation_results
                }
                
                correlations.append(result)
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(correlations)
        
        if len(corr_df) == 0:
            return corr_df
        
        # Add summary statistics
        primary_method = self.methods[0] if self.methods else 'pearson'
        corr_col = f'{primary_method}_corr'
        pval_col = f'{primary_method}_pval'
        
        if corr_col in corr_df.columns:
            corr_df['abs_correlation'] = np.abs(corr_df[corr_col])
            
            # Multiple testing correction
            valid_pvals = ~corr_df[pval_col].isna()
            if valid_pvals.sum() > 0:
                _, padj, _, _ = multipletests(
                    corr_df.loc[valid_pvals, pval_col],
                    method=self.fdr_method
                )
                
                corr_df.loc[valid_pvals, 'padj'] = padj
                corr_df.loc[~valid_pvals, 'padj'] = 1.0
            else:
                corr_df['padj'] = 1.0
        
        # Sort by significance
        if 'padj' in corr_df.columns:
            corr_df = corr_df.sort_values('padj')
        
        return corr_df
    
    def _global_correlation_patterns(self, snp_matrix: np.ndarray, 
                                   meth_matrix: np.ndarray) -> Dict:
        """
        Analyze global correlation patterns between SNP and methylation data.
        """
        results = {}
        
        # Overall correlation between SNP and methylation principal components
        from sklearn.decomposition import PCA
        
        # Reduce dimensionality
        n_components = min(10, snp_matrix.shape[0] - 1, snp_matrix.shape[1], meth_matrix.shape[1])
        
        if n_components > 1:
            pca_snp = PCA(n_components=n_components)
            pca_meth = PCA(n_components=n_components)
            
            snp_pcs = pca_snp.fit_transform(snp_matrix)
            meth_pcs = pca_meth.fit_transform(meth_matrix)
            
            # Cross-correlate principal components
            pc_correlations = np.corrcoef(snp_pcs.T, meth_pcs.T)[:n_components, n_components:]
            
            results['pc_correlations'] = pc_correlations
            results['snp_variance_explained'] = pca_snp.explained_variance_ratio_
            results['meth_variance_explained'] = pca_meth.explained_variance_ratio_
            
            # Find most correlated PC pairs
            max_corr_idx = np.unravel_index(np.argmax(np.abs(pc_correlations)), pc_correlations.shape)
            results['max_pc_correlation'] = {
                'correlation': pc_correlations[max_corr_idx],
                'snp_pc': max_corr_idx[0],
                'meth_pc': max_corr_idx[1]
            }
        
        # Canonical correlation analysis
        try:
            from sklearn.cross_decomposition import CCA
            
            n_components_cca = min(5, snp_matrix.shape[0] - 1, snp_matrix.shape[1], meth_matrix.shape[1])
            if n_components_cca > 1:
                cca = CCA(n_components=n_components_cca)
                snp_c, meth_c = cca.fit_transform(snp_matrix, meth_matrix)
                
                # Calculate canonical correlations
                canonical_corrs = [np.corrcoef(snp_c[:, i], meth_c[:, i])[0, 1] 
                                 for i in range(n_components_cca)]
                
                results['canonical_correlations'] = canonical_corrs
                results['max_canonical_correlation'] = np.max(np.abs(canonical_corrs))
                
        except Exception as e:
            logger.warning(f"Canonical correlation analysis failed: {e}")
        
        return results
    
    def _distance_correlation_analysis(self, snp_matrix: np.ndarray, 
                                     meth_matrix: np.ndarray,
                                     snp_data: pd.DataFrame) -> Dict:
        """
        Analyze correlation patterns based on genomic distance.
        """
        results = {}
        
        # Group correlations by genomic distance (for SNPs on same chromosome)
        if 'chr' not in snp_data.columns or 'pos' not in snp_data.columns:
            return {'error': 'Missing chromosome or position information'}
        
        distance_correlations = []
        
        # Analyze correlations within chromosomes
        for chromosome in snp_data['chr'].unique():
            chrom_snps = snp_data[snp_data['chr'] == chromosome]
            
            if len(chrom_snps) < 2:
                continue
            
            chrom_indices = chrom_snps.index.tolist()
            
            # Calculate pairwise distances and correlations for SNPs on this chromosome
            for i in range(len(chrom_indices)):
                for j in range(i + 1, min(i + 21, len(chrom_indices))):  # Limit to nearby SNPs
                    idx1, idx2 = chrom_indices[i], chrom_indices[j]
                    
                    # Genomic distance
                    pos1 = snp_data.loc[idx1, 'pos']
                    pos2 = snp_data.loc[idx2, 'pos']
                    distance = abs(pos2 - pos1)
                    
                    # SNP correlation
                    snp1_genotypes = snp_matrix[:, idx1]
                    snp2_genotypes = snp_matrix[:, idx2]
                    
                    try:
                        snp_corr, _ = stats.pearsonr(snp1_genotypes, snp2_genotypes)
                        
                        distance_correlations.append({
                            'chromosome': chromosome,
                            'distance': distance,
                            'correlation': snp_corr,
                            'snp1_id': snp_data.loc[idx1, 'snp_id'],
                            'snp2_id': snp_data.loc[idx2, 'snp_id']
                        })
                        
                    except Exception as e:
                        continue
        
        if distance_correlations:
            distance_df = pd.DataFrame(distance_correlations)
            
            # Bin distances and calculate mean correlations
            distance_bins = [0, 1000, 10000, 100000, 1000000, np.inf]
            distance_df['distance_bin'] = pd.cut(distance_df['distance'], distance_bins)
            
            binned_correlations = distance_df.groupby('distance_bin')['correlation'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            
            results['distance_correlation_decay'] = binned_correlations
            results['raw_distance_correlations'] = distance_df
        
        return results


class MLCorrelationAnalyzer:
    """
    Machine learning-based correlation analysis between SNPs and methylation.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_selection_method = config['analysis']['feature_selection']['method']
        self.k_best = config['analysis']['feature_selection']['k_best']
    
    def analyze_ml_associations(self, snp_data: pd.DataFrame, 
                              methylation_data: pd.DataFrame) -> Dict:
        """
        Use machine learning methods to identify SNP-methylation associations.
        
        Args:
            snp_data: SNP genotype data  
            methylation_data: DNA methylation data
            
        Returns:
            Dictionary with ML analysis results
        """
        results = {}
        
        # Prepare data
        X, y, feature_names = self._prepare_ml_data(snp_data, methylation_data)
        
        if X is None or y is None:
            return {'error': 'Could not prepare data for ML analysis'}
        
        # Feature importance analysis
        results['feature_importance'] = self._analyze_feature_importance(X, y, feature_names)
        
        # Mutual information analysis
        results['mutual_information'] = self._mutual_information_analysis(X, y, feature_names)
        
        # Non-linear association detection
        results['nonlinear_associations'] = self._detect_nonlinear_associations(X, y, feature_names)
        
        return results
    
    def _prepare_ml_data(self, snp_data: pd.DataFrame, 
                        methylation_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare data for machine learning analysis."""
        
        # Extract genotype matrix
        genotype_cols = [col for col in snp_data.columns if col.startswith('genotype_')]
        if not genotype_cols:
            return None, None, []
        
        # Get common samples
        snp_samples = [col.replace('genotype_', '') for col in genotype_cols]
        meth_samples = methylation_data.columns.tolist()
        common_samples = list(set(snp_samples) & set(meth_samples))
        
        if len(common_samples) < 5:
            return None, None, []
        
        # Create feature matrix (SNPs) and target matrix (methylation)
        X = snp_data[[f'genotype_{sample}' for sample in common_samples]].values.T
        y = methylation_data[common_samples].values.T
        
        # Handle missing values
        X = np.nan_to_num(X, nan=np.nanmean(X))
        y = np.nan_to_num(y, nan=np.nanmean(y))
        
        # Feature names
        feature_names = snp_data['snp_id'].tolist()
        
        return X, y, feature_names
    
    def _analyze_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                  feature_names: List[str]) -> Dict:
        """
        Analyze feature importance using Random Forest.
        """
        results = {}
        n_targets = min(y.shape[1], 20)  # Limit number of targets for efficiency
        
        importances_list = []
        r2_scores = []
        
        for i in range(n_targets):
            target = y[:, i]
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, target)
            
            # Predictions and R2
            y_pred = rf.predict(X)
            r2 = r2_score(target, y_pred)
            r2_scores.append(r2)
            
            # Feature importances
            importances = rf.feature_importances_
            importances_list.append(importances)
        
        # Average feature importances across targets
        mean_importances = np.mean(importances_list, axis=0)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_importances
        }).sort_values('importance', ascending=False)
        
        results.update({
            'feature_importance_df': importance_df,
            'r2_scores': r2_scores,
            'mean_r2': np.mean(r2_scores),
            'top_features': importance_df.head(20).to_dict('records')
        })
        
        return results
    
    def _mutual_information_analysis(self, X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str]) -> Dict:
        """
        Analyze mutual information between SNPs and methylation sites.
        """
        results = {}
        n_targets = min(y.shape[1], 10)  # Limit for efficiency
        
        mi_scores_list = []
        
        for i in range(n_targets):
            target = y[:, i]
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, target, random_state=42)
            mi_scores_list.append(mi_scores)
        
        # Average MI scores
        mean_mi_scores = np.mean(mi_scores_list, axis=0)
        
        # Create MI DataFrame
        mi_df = pd.DataFrame({
            'feature': feature_names,
            'mutual_information': mean_mi_scores
        }).sort_values('mutual_information', ascending=False)
        
        results.update({
            'mutual_information_df': mi_df,
            'top_mi_features': mi_df.head(20).to_dict('records')
        })
        
        return results
    
    def _detect_nonlinear_associations(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> Dict:
        """
        Detect non-linear associations using polynomial features and regularization.
        """
        results = {}
        
        # Use subset for computational efficiency
        n_features = min(X.shape[1], 100)
        n_targets = min(y.shape[1], 5)
        
        X_subset = X[:, :n_features]
        y_subset = y[:, :n_targets]
        feature_subset = feature_names[:n_features]
        
        # Create polynomial features (degree 2)
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X_subset)
        
        nonlinear_scores = []
        
        for i in range(n_targets):
            target = y_subset[:, i]
            
            # Elastic Net with cross-validation
            elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            
            # Standardize features
            scaler = StandardScaler()
            X_poly_scaled = scaler.fit_transform(X_poly)
            
            # Fit model
            elastic_net.fit(X_poly_scaled, target)
            
            # Get non-zero coefficients (selected features)
            nonzero_coefs = elastic_net.coef_[elastic_net.coef_ != 0]
            nonzero_indices = np.where(elastic_net.coef_ != 0)[0]
            
            # Score model
            y_pred = elastic_net.predict(X_poly_scaled)
            r2 = r2_score(target, y_pred)
            
            nonlinear_scores.append({
                'target_index': i,
                'r2_score': r2,
                'n_selected_features': len(nonzero_coefs),
                'selected_features': nonzero_indices.tolist()
            })
        
        results['nonlinear_models'] = nonlinear_scores
        results['polynomial_feature_names'] = poly.get_feature_names_out(feature_subset)
        
        return results


class WindowBasedAnalyzer:
    """
    Genomic window-based correlation analysis.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.window_size = config['analysis']['window_analysis']['window_size']
        self.step_size = config['analysis']['window_analysis']['step_size']
    
    def analyze_genomic_windows(self, snp_data: pd.DataFrame, 
                              methylation_data: pd.DataFrame) -> Dict:
        """
        Perform window-based correlation analysis across the genome.
        
        Args:
            snp_data: SNP data with genomic coordinates
            methylation_data: Methylation data
            
        Returns:
            Dictionary with window-based analysis results
        """
        results = {}
        
        if 'chr' not in snp_data.columns or 'pos' not in snp_data.columns:
            return {'error': 'Missing genomic coordinates in SNP data'}
        
        # Analyze each chromosome
        chromosome_results = {}
        
        for chromosome in snp_data['chr'].unique():
            logger.info(f"Analyzing chromosome {chromosome}")
            
            chrom_results = self._analyze_chromosome_windows(
                snp_data[snp_data['chr'] == chromosome], 
                methylation_data, 
                chromosome
            )
            
            if chrom_results:
                chromosome_results[chromosome] = chrom_results
        
        results['chromosome_analysis'] = chromosome_results
        
        # Global summary across all chromosomes
        results['global_summary'] = self._summarize_window_results(chromosome_results)
        
        return results
    
    def _analyze_chromosome_windows(self, chrom_snps: pd.DataFrame, 
                                  methylation_data: pd.DataFrame, 
                                  chromosome: str) -> Dict:
        """Analyze correlation patterns within windows on a single chromosome."""
        
        if len(chrom_snps) < 2:
            return {}
        
        # Sort SNPs by position
        chrom_snps = chrom_snps.sort_values('pos')
        
        # Define windows
        min_pos = chrom_snps['pos'].min()
        max_pos = chrom_snps['pos'].max()
        
        windows = []
        window_start = min_pos
        
        while window_start < max_pos:
            window_end = window_start + self.window_size
            
            # Get SNPs in this window
            window_snps = chrom_snps[
                (chrom_snps['pos'] >= window_start) & 
                (chrom_snps['pos'] < window_end)
            ]
            
            if len(window_snps) >= 2:  # Need at least 2 SNPs for analysis
                window_result = self._analyze_single_window(
                    window_snps, methylation_data, chromosome, 
                    window_start, window_end
                )
                
                if window_result:
                    windows.append(window_result)
            
            window_start += self.step_size
        
        return {'windows': windows, 'n_windows': len(windows)}
    
    def _analyze_single_window(self, window_snps: pd.DataFrame, 
                             methylation_data: pd.DataFrame,
                             chromosome: str, start_pos: int, end_pos: int) -> Dict:
        """Analyze a single genomic window."""
        
        # Extract genotype data for this window
        genotype_cols = [col for col in window_snps.columns if col.startswith('genotype_')]
        if not genotype_cols:
            return {}
        
        # Get common samples
        snp_samples = [col.replace('genotype_', '') for col in genotype_cols]
        meth_samples = methylation_data.columns.tolist()
        common_samples = list(set(snp_samples) & set(meth_samples))
        
        if len(common_samples) < 3:
            return {}
        
        # Create matrices
        window_genotypes = window_snps[[f'genotype_{sample}' for sample in common_samples]].values.T
        window_methylation = methylation_data[common_samples].values.T
        
        # Handle missing values
        window_genotypes = np.nan_to_num(window_genotypes, nan=np.nanmean(window_genotypes))
        window_methylation = np.nan_to_num(window_methylation, nan=np.nanmean(window_methylation))
        
        # Calculate summary statistics
        n_snps = window_genotypes.shape[1]
        n_cpgs = window_methylation.shape[1]
        
        # Mean correlation between SNPs and methylation in this window
        correlations = []
        
        for i in range(min(n_snps, 10)):  # Limit for efficiency
            for j in range(min(n_cpgs, 20)):
                try:
                    corr, p_val = stats.pearsonr(window_genotypes[:, i], window_methylation[:, j])
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue
        
        if not correlations:
            return {}
        
        # Window statistics
        window_stats = {
            'chromosome': chromosome,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'n_snps': n_snps,
            'n_cpgs': n_cpgs,
            'n_samples': len(common_samples),
            'mean_correlation': np.mean(correlations),
            'max_correlation': np.max(np.abs(correlations)),
            'correlation_std': np.std(correlations),
            'n_correlations': len(correlations)
        }
        
        # Additional analysis for high-correlation windows
        if window_stats['max_correlation'] > 0.5:
            window_stats['high_correlation'] = True
            
            # Identify top correlating SNP-CpG pairs
            top_pairs = []
            
            for i in range(min(n_snps, 5)):
                for j in range(min(n_cpgs, 5)):
                    try:
                        corr, p_val = stats.pearsonr(window_genotypes[:, i], window_methylation[:, j])
                        if not np.isnan(corr) and abs(corr) > 0.3:
                            top_pairs.append({
                                'snp_id': window_snps.iloc[i]['snp_id'],
                                'cpg_idx': j,
                                'correlation': corr,
                                'p_value': p_val
                            })
                    except:
                        continue
            
            # Sort by absolute correlation
            top_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            window_stats['top_pairs'] = top_pairs[:5]
        
        return window_stats
    
    def _summarize_window_results(self, chromosome_results: Dict) -> Dict:
        """Summarize results across all chromosomes and windows."""
        
        all_windows = []
        for chrom_data in chromosome_results.values():
            if 'windows' in chrom_data:
                all_windows.extend(chrom_data['windows'])
        
        if not all_windows:
            return {'n_windows': 0}
        
        # Global statistics
        correlations = [w['mean_correlation'] for w in all_windows]
        max_correlations = [w['max_correlation'] for w in all_windows]
        
        summary = {
            'n_windows_total': len(all_windows),
            'global_mean_correlation': np.mean(correlations),
            'global_max_correlation': np.max(max_correlations),
            'correlation_std': np.std(correlations),
            'high_correlation_windows': len([w for w in all_windows if w.get('high_correlation', False)])
        }
        
        # Top windows by correlation
        sorted_windows = sorted(all_windows, key=lambda x: x['max_correlation'], reverse=True)
        summary['top_correlation_windows'] = sorted_windows[:10]
        
        return summary