"""
Classical machine learning models for genomic analysis.

This module implements classsic e.g. LDA and HMM (Linear Discriminant Analysis) and HMM (Hidden Markov Model)
approaches for pattern recognition in SNP and methylation data.
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from hmmlearn.hmm import GaussianHMM, MultinomialHMM
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
import warnings

# Optional pomegranate import
try:
    from pomegranate import HiddenMarkovModel, NormalDistribution, State
    POMEGRANATE_AVAILABLE = True
except ImportError:
    POMEGRANATE_AVAILABLE = False
    warnings.warn("pomegranate not available. Some advanced HMM features will be limited.")
# TBA: in linux

logger = logging.getLogger(__name__)

class ClassicModels:
    """
    Container class for classical machine learning models.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.lda_model = LDAAnalyzer(config['models']['classic']['lda'])
        self.hmm_model = HMMAnalyzer(config['models']['classic']['hmm'])
    
    def fit_all_models(self, snp_data: pd.DataFrame, methylation_data: pd.DataFrame, 
                      sample_labels: Optional[pd.Series] = None) -> Dict:
        """
        Fit all classical models to the data.
        
        Args:
            snp_data: SNP genotype data
            methylation_data: DNA methylation data
            sample_labels: Optional labels for supervised learning
            
        Returns:
            Dictionary containing fitted models and results
        """
        results = {}
        
        logger.info("Fitting LDA model...")
        results['lda'] = self.lda_model.fit_and_analyze(snp_data, methylation_data, sample_labels)
        
        logger.info("Fitting HMM model...")
        results['hmm'] = self.hmm_model.fit_and_analyze(snp_data, methylation_data)
        
        return results


class LDAAnalyzer:
    """
    Linear Discriminant Analysis for genomic data.
    
    LDA can be used for:
    1. Dimensionality reduction of high-dimensional genomic data
    2. Classification when sample labels are available
    3. Feature selection based on discriminative power
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_components = config.get('n_components', 10)
        self.solver = config.get('solver', 'svd')
        self.shrinkage = config.get('shrinkage', 'auto')
        
        self.lda_snp = None
        self.lda_methylation = None
        self.scaler_snp = StandardScaler()
        self.scaler_methylation = StandardScaler()
    
    def fit_and_analyze(self, snp_data: pd.DataFrame, methylation_data: pd.DataFrame,
                       sample_labels: Optional[pd.Series] = None) -> Dict:
        """
        Fit LDA models and perform analysis.
        
        Args:
            snp_data: SNP genotype matrix
            methylation_data: Methylation beta values
            sample_labels: Sample labels for supervised LDA
            
        Returns:
            Dictionary with LDA results
        """
        results = {'snp': {}, 'methylation': {}, 'combined': {}}
        
        # Extract sample genotype matrices
        snp_genotypes = self._extract_genotype_matrix(snp_data)
        methylation_matrix = methylation_data.T  # Transpose to samples x features
        
        # Align samples
        common_samples = list(set(snp_genotypes.index) & set(methylation_matrix.index))
        snp_aligned = snp_genotypes.loc[common_samples]
        meth_aligned = methylation_matrix.loc[common_samples]
        
        if sample_labels is not None:
            sample_labels = sample_labels.loc[common_samples]
        
        # Perform LDA analysis on SNP data
        results['snp'] = self._fit_lda_snp(snp_aligned, sample_labels)
        
        # Perform LDA analysis on methylation data
        results['methylation'] = self._fit_lda_methylation(meth_aligned, sample_labels)
        
        # Combined analysis
        results['combined'] = self._fit_combined_lda(snp_aligned, meth_aligned, sample_labels)
        
        return results
    
    def _extract_genotype_matrix(self, snp_data: pd.DataFrame) -> pd.DataFrame:
        """Extract genotype matrix from SNP data."""
        genotype_cols = [col for col in snp_data.columns if col.startswith('genotype_')]
        
        if not genotype_cols:
            raise ValueError("No genotype columns found in SNP data")
        
        # Create sample x SNP matrix
        genotype_matrix = snp_data[genotype_cols].T
        genotype_matrix.index = [idx.replace('genotype_', '') for idx in genotype_matrix.index]
        genotype_matrix.columns = snp_data['snp_id']
        
        # Handle missing values
        genotype_matrix = genotype_matrix.fillna(genotype_matrix.mean())
        
        return genotype_matrix
    
    def _fit_lda_snp(self, snp_matrix: pd.DataFrame, sample_labels: Optional[pd.Series]) -> Dict:
        """Fit LDA to SNP data."""
        results = {}
        
        # Standardize data
        X_snp_scaled = self.scaler_snp.fit_transform(snp_matrix)
        
        if sample_labels is not None:
            # Supervised LDA
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(sample_labels)
            
            # Adapt n_components to data constraints
            max_components = min(
                self.n_components,
                len(np.unique(y_encoded)) - 1,  # LDA constraint: max n_classes - 1
                X_snp_scaled.shape[0] - 1,      # Can't have more components than samples
                X_snp_scaled.shape[1]           # Can't have more components than features
            )
            
            self.lda_snp = LinearDiscriminantAnalysis(
                n_components=max(1, max_components),  # At least 1 component
                solver=self.solver,
                shrinkage=self.shrinkage if self.solver == 'lsqr' else None
            )
            
            X_lda = self.lda_snp.fit_transform(X_snp_scaled, y_encoded)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.lda_snp, X_snp_scaled, y_encoded, cv=5)
            
            results.update({
                'lda_components': X_lda,
                'explained_variance_ratio': self.lda_snp.explained_variance_ratio_,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': np.abs(self.lda_snp.coef_).mean(axis=0)
            })
        else:
            # Unsupervised dimensionality reduction using PCA-like approach
            from sklearn.decomposition import PCA
            max_components = min(
                self.n_components,
                X_snp_scaled.shape[0] - 1,  # Can't have more components than samples
                X_snp_scaled.shape[1]       # Can't have more components than features
            )
            pca = PCA(n_components=max(1, max_components))
            X_pca = pca.fit_transform(X_snp_scaled)
            
            results.update({
                'pca_components': X_pca,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
            })
        
        return results
    
    def _fit_lda_methylation(self, meth_matrix: pd.DataFrame, sample_labels: Optional[pd.Series]) -> Dict:
        """Fit LDA to methylation data."""
        results = {}
        
        # Standardize data
        X_meth_scaled = self.scaler_methylation.fit_transform(meth_matrix)
        
        if sample_labels is not None:
            # Supervised LDA
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(sample_labels)
            
            # Adapt n_components to data constraints
            max_components = min(
                self.n_components,
                len(np.unique(y_encoded)) - 1,  # LDA constraint: max n_classes - 1
                X_meth_scaled.shape[0] - 1,     # Can't have more components than samples
                X_meth_scaled.shape[1]          # Can't have more components than features
            )
            
            self.lda_methylation = LinearDiscriminantAnalysis(
                n_components=max(1, max_components),  # At least 1 component
                solver=self.solver,
                shrinkage=self.shrinkage if self.solver == 'lsqr' else None
            )
            
            X_lda = self.lda_methylation.fit_transform(X_meth_scaled, y_encoded)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.lda_methylation, X_meth_scaled, y_encoded, cv=5)
            
            results.update({
                'lda_components': X_lda,
                'explained_variance_ratio': self.lda_methylation.explained_variance_ratio_,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': np.abs(self.lda_methylation.coef_).mean(axis=0)
            })
        else:
            # Unsupervised dimensionality reduction
            from sklearn.decomposition import PCA
            max_components = min(
                self.n_components,
                X_meth_scaled.shape[0] - 1,  # Can't have more components than samples
                X_meth_scaled.shape[1]       # Can't have more components than features
            )
            pca = PCA(n_components=max(1, max_components))
            X_pca = pca.fit_transform(X_meth_scaled)
            
            results.update({
                'pca_components': X_pca,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
            })
        
        return results
    
    def _fit_combined_lda(self, snp_matrix: pd.DataFrame, meth_matrix: pd.DataFrame,
                         sample_labels: Optional[pd.Series]) -> Dict:
        """Fit LDA to combined SNP and methylation data."""
        results = {}
        
        # Combine features (may need feature selection due to high dimensionality)
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        
        # Standardize both datasets
        X_snp_scaled = self.scaler_snp.transform(snp_matrix)
        X_meth_scaled = self.scaler_methylation.transform(meth_matrix)
        
        # Combine features
        X_combined = np.hstack([X_snp_scaled, X_meth_scaled])
        
        if sample_labels is not None:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(sample_labels)
            
            # Feature selection to avoid curse of dimensionality
            selector = SelectKBest(score_func=f_classif, k=min(1000, X_combined.shape[1]))
            X_selected = selector.fit_transform(X_combined, y_encoded)
            
            # Fit LDA with adaptive components
            max_components = min(
                self.n_components,
                len(np.unique(y_encoded)) - 1,  # LDA constraint: max n_classes - 1
                X_selected.shape[0] - 1,        # Can't have more components than samples
                X_selected.shape[1]             # Can't have more components than features
            )
            
            lda_combined = LinearDiscriminantAnalysis(
                n_components=max(1, max_components),  # At least 1 component
                solver=self.solver,  # Use configured solver
                shrinkage=self.shrinkage if self.solver == 'lsqr' else None
            )
            
            X_lda = lda_combined.fit_transform(X_selected, y_encoded)
            
            # Cross-validation
            cv_scores = cross_val_score(lda_combined, X_selected, y_encoded, cv=5)
            
            results.update({
                'lda_components': X_lda,
                'explained_variance_ratio': lda_combined.explained_variance_ratio_,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'selected_features': selector.get_support(),
                'feature_scores': selector.scores_
            })
        
        return results


class HMMAnalyzer:
    """
    Hidden Markov Model analyzer for genomic sequences and patterns.
    
    HMM can be used for:
    1. Identifying methylation domains and patterns
    2. Detecting genomic regions with correlated SNP-methylation patterns
    3. Segmentation of chromosomes into functional regions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_components = config.get('n_components', 5)
        self.covariance_type = config.get('covariance_type', 'full')
        self.n_iter = config.get('n_iter', 100)
        self.random_state = config.get('random_state', 42)
        
        self.hmm_methylation = None
        self.hmm_snp = None
    
    def fit_and_analyze(self, snp_data: pd.DataFrame, methylation_data: pd.DataFrame) -> Dict:
        """
        Fit HMM models to identify patterns and domains.
        
        Args:
            snp_data: SNP data with genomic positions
            methylation_data: Methylation data
            
        Returns:
            Dictionary with HMM analysis results
        """
        results = {'methylation_domains': {}, 'snp_patterns': {}, 'combined_analysis': {}}
        
        # Analyze methylation domains
        results['methylation_domains'] = self._analyze_methylation_domains(methylation_data)
        
        # Analyze SNP patterns
        results['snp_patterns'] = self._analyze_snp_patterns(snp_data)
        
        # Combined genomic segmentation
        results['combined_analysis'] = self._combined_genomic_segmentation(snp_data, methylation_data)
        
        return results
    
    def _analyze_methylation_domains(self, methylation_data: pd.DataFrame) -> Dict:
        """
        Use HMM to identify methylation domains across samples.
        """
        results = {}
        
        # Prepare data: use mean methylation across samples for each CpG site
        mean_methylation = methylation_data.mean(axis=1).values.reshape(-1, 1)
        
        # Fit Gaussian HMM to identify methylation states
        self.hmm_methylation = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        try:
            self.hmm_methylation.fit(mean_methylation)
            
            # Predict states
            states = self.hmm_methylation.predict(mean_methylation)
            
            # Get state parameters
            means = self.hmm_methylation.means_.flatten()
            covars = self.hmm_methylation.covars_.flatten()
            
            # Identify domain boundaries
            domain_changes = np.where(np.diff(states) != 0)[0]
            
            results.update({
                'states': states,
                'state_means': means,
                'state_variances': covars,
                'domain_boundaries': domain_changes,
                'transition_matrix': self.hmm_methylation.transmat_,
                'log_likelihood': self.hmm_methylation.score(mean_methylation)
            })
            
            # Characterize domains
            domain_stats = []
            boundaries_extended = np.concatenate([[0], domain_changes + 1, [len(states)]])
            
            for i in range(len(boundaries_extended) - 1):
                start_idx = boundaries_extended[i]
                end_idx = boundaries_extended[i + 1]
                domain_state = stats.mode(states[start_idx:end_idx])[0][0]
                domain_length = end_idx - start_idx
                domain_mean_meth = np.mean(mean_methylation[start_idx:end_idx])
                
                domain_stats.append({
                    'domain_id': i,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'length': domain_length,
                    'dominant_state': domain_state,
                    'mean_methylation': domain_mean_meth
                })
            
            results['domain_statistics'] = domain_stats
            
        except Exception as e:
            logger.warning(f"HMM methylation analysis failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _analyze_snp_patterns(self, snp_data: pd.DataFrame) -> Dict:
        """
        Use HMM to identify patterns in SNP data along chromosomes.
        """
        results = {}
        
        # Group SNPs by chromosome and sort by position
        chromosomes = snp_data['chr'].unique()
        
        for chrom in chromosomes:
            chrom_snps = snp_data[snp_data['chr'] == chrom].sort_values('pos')
            
            if len(chrom_snps) < 10:  # Skip chromosomes with too few SNPs
                continue
            
            # Calculate mean allele frequency across samples
            genotype_cols = [col for col in chrom_snps.columns if col.startswith('genotype_')]
            if not genotype_cols:
                continue
                
            mean_allele_freq = chrom_snps[genotype_cols].mean(axis=1).values.reshape(-1, 1)
            
            # Fit HMM
            hmm_chrom = GaussianHMM(
                n_components=min(3, len(mean_allele_freq) // 5),  # Fewer states for smaller regions
                covariance_type='spherical',  # Simpler model
                n_iter=50,
                random_state=self.random_state
            )
            
            try:
                hmm_chrom.fit(mean_allele_freq)
                states = hmm_chrom.predict(mean_allele_freq)
                
                # Find state transitions (potential recombination hotspots)
                transitions = np.where(np.diff(states) != 0)[0]
                transition_positions = chrom_snps.iloc[transitions]['pos'].values
                
                results[chrom] = {
                    'states': states,
                    'positions': chrom_snps['pos'].values,
                    'transition_points': transition_positions,
                    'state_means': hmm_chrom.means_.flatten(),
                    'log_likelihood': hmm_chrom.score(mean_allele_freq)
                }
                
            except Exception as e:
                logger.warning(f"HMM analysis failed for chromosome {chrom}: {e}")
                continue
        
        return results
    
    def _combined_genomic_segmentation(self, snp_data: pd.DataFrame, 
                                     methylation_data: pd.DataFrame) -> Dict:
        """
        Perform combined segmentation using both SNP and methylation data.
        """
        results = {}
        
        try:
            # This is a simplified approach - in practice, you'd need to align
            # SNPs and CpG sites by genomic coordinates
            
            # Extract features for combined analysis
            genotype_cols = [col for col in snp_data.columns if col.startswith('genotype_')]
            if not genotype_cols:
                return {'error': 'No genotype data found'}
            
            # Use mean values across samples for simplicity
            mean_snp = snp_data[genotype_cols].mean(axis=1).values
            mean_meth = methylation_data.mean(axis=1).values
            
            # Take subset for computational efficiency
            n_features = min(len(mean_snp), len(mean_meth), 1000)
            
            # Create combined feature vector
            combined_features = np.column_stack([
                mean_snp[:n_features],
                mean_meth[:n_features]
            ])
            
            # Fit multivariate HMM
            hmm_combined = GaussianHMM(
                n_components=self.n_components,
                covariance_type='tied',  # Shared covariance for stability
                n_iter=50,
                random_state=self.random_state
            )
            
            hmm_combined.fit(combined_features)
            states = hmm_combined.predict(combined_features)
            
            # Analyze state characteristics
            state_characteristics = []
            for state in range(self.n_components):
                state_mask = states == state
                if np.sum(state_mask) > 0:
                    state_snp_mean = np.mean(mean_snp[:n_features][state_mask])
                    state_meth_mean = np.mean(mean_meth[:n_features][state_mask])
                    
                    state_characteristics.append({
                        'state': state,
                        'count': np.sum(state_mask),
                        'snp_mean': state_snp_mean,
                        'methylation_mean': state_meth_mean
                    })
            
            results.update({
                'states': states,
                'state_characteristics': state_characteristics,
                'transition_matrix': hmm_combined.transmat_,
                'log_likelihood': hmm_combined.score(combined_features)
            })
            
        except Exception as e:
            logger.warning(f"Combined HMM analysis failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def predict_genomic_domains(self, new_data: np.ndarray, model_type: str = 'methylation') -> np.ndarray:
        """
        Predict genomic domains for new data using fitted HMM models.
        
        Args:
            new_data: New genomic data
            model_type: Type of model to use ('methylation' or 'snp')
            
        Returns:
            Predicted states
        """
        if model_type == 'methylation' and self.hmm_methylation is not None:
            return self.hmm_methylation.predict(new_data.reshape(-1, 1))
        elif model_type == 'snp' and self.hmm_snp is not None:
            return self.hmm_snp.predict(new_data.reshape(-1, 1))
        else:
            raise ValueError(f"Model {model_type} not fitted or not available")