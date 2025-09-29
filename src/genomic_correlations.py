"""
Genomic Region Correlation Analysis for Multi-Omics Data.

This module analyzes correlations between different genomic assays by examining
overlapping regions, enrichment patterns, and treatment effects.
Specifically designed for analyzing END-seq breaks in relation to other genomic features.
"""
# import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from scipy import stats
from scipy.stats import fisher_exact, hypergeom
from itertools import combinations
import warnings

logger = logging.getLogger(__name__)

class GenomicRegionAnalyzer:
    """
    Analyzes overlaps and correlations between genomic regions from different assays.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.overlap_threshold = config.get('analysis', {}).get('overlap_threshold', 1)  # Minimum overlap in bp
        self.enrichment_threshold = config.get('analysis', {}).get('enrichment_p_threshold', 0.05)
        
    def calculate_region_overlaps(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                                 name1: str, name2: str) -> Dict:
        """
        Calculate overlaps between two sets of genomic regions.
        
        Args:
            data1, data2: DataFrames with 'chr', 'start', 'end' columns
            name1, name2: Names for the datasets
            
        Returns:
            Dictionary with overlap statistics
        """
        logger.info(f"Calculating overlaps between {name1} and {name2}")
        
        overlaps = []
        overlap_details = []
        
        # Group by chromosome for efficiency
        for chr_name in set(data1['chr'].unique()) & set(data2['chr'].unique()):
            chr_data1 = data1[data1['chr'] == chr_name].copy()
            chr_data2 = data2[data2['chr'] == chr_name].copy()
            
            # Calculate overlaps for this chromosome
            chr_overlaps = self._calculate_chromosome_overlaps(chr_data1, chr_data2, chr_name)
            overlaps.extend(chr_overlaps)
            
            # Store detailed overlap information
            for overlap in chr_overlaps:
                overlap_details.append({
                    'chr': chr_name,
                    'dataset1': name1,
                    'dataset2': name2,
                    **overlap
                })
        
        # Calculate summary statistics
        total_overlaps = len([o for o in overlaps if o['overlap_length'] > 0])
        total_possible = len(data1) * len(data2)  # Maximum possible overlaps
        
        # Calculate region-based overlap statistics
        regions1_with_overlap = len(set([o['region1_idx'] for o in overlaps if o['overlap_length'] > 0]))
        regions2_with_overlap = len(set([o['region2_idx'] for o in overlaps if o['overlap_length'] > 0]))
        
        overlap_stats = {
            'dataset1': name1,
            'dataset2': name2,
            'total_regions1': len(data1),
            'total_regions2': len(data2),
            'total_overlaps': total_overlaps,
            'regions1_with_overlap': regions1_with_overlap,
            'regions2_with_overlap': regions2_with_overlap,
            'overlap_fraction1': regions1_with_overlap / len(data1) if len(data1) > 0 else 0,
            'overlap_fraction2': regions2_with_overlap / len(data2) if len(data2) > 0 else 0,
            'overlap_details': overlap_details
        }
        
        # Add enrichment analysis
        enrichment = self._calculate_enrichment(data1, data2, overlaps)
        overlap_stats['enrichment'] = enrichment
        
        return overlap_stats
    
    def _calculate_chromosome_overlaps(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                                     chr_name: str) -> List[Dict]:
        """Calculate overlaps within a single chromosome."""
        overlaps = []
        
        for idx1, row1 in data1.iterrows():
            for idx2, row2 in data2.iterrows():
                # Calculate overlap
                overlap_start = max(row1['start'], row2['start'])
                overlap_end = min(row1['end'], row2['end'])
                overlap_length = max(0, overlap_end - overlap_start)
                
                if overlap_length >= self.overlap_threshold:
                    overlaps.append({
                        'region1_idx': idx1,
                        'region2_idx': idx2,
                        'region1_start': row1['start'],
                        'region1_end': row1['end'],
                        'region2_start': row2['start'], 
                        'region2_end': row2['end'],
                        'overlap_start': overlap_start,
                        'overlap_end': overlap_end,
                        'overlap_length': overlap_length,
                        'region1_length': row1['end'] - row1['start'],
                        'region2_length': row2['end'] - row2['start'],
                        'overlap_fraction1': overlap_length / (row1['end'] - row1['start']),
                        'overlap_fraction2': overlap_length / (row2['end'] - row2['start']),
                        'region1_feature': row1.get('feature_type', 'unknown'),
                        'region2_feature': row2.get('feature_type', 'unknown'),
                        'region1_gene': row1.get('gene_name', 'unknown'),
                        'region2_gene': row2.get('gene_name', 'unknown')
                    })
        
        return overlaps
    
    def _calculate_enrichment(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                            overlaps: List[Dict]) -> Dict:
        """
        Calculate statistical enrichment of overlaps using hypergeometric test.
        """
        # Count overlapping regions
        overlapping_regions1 = len(set([o['region1_idx'] for o in overlaps if o['overlap_length'] > 0]))
        overlapping_regions2 = len(set([o['region2_idx'] for o in overlaps if o['overlap_length'] > 0]))
        
        total_regions1 = len(data1)
        total_regions2 = len(data2)
        
        # Estimate expected overlaps based on random expectation
        # This is a simplified approach - could be improved with genome-wide background
        genome_size = 3e9  # Approximate human genome size
        
        # Calculate average region sizes
        avg_size1 = data1['length'].mean() if 'length' in data1.columns else (data1['end'] - data1['start']).mean()
        avg_size2 = data2['length'].mean() if 'length' in data2.columns else (data2['end'] - data2['start']).mean()
        
        # Expected overlap probability (simplified)
        expected_prob = (avg_size1 * avg_size2) / (genome_size ** 2)
        expected_overlaps = expected_prob * total_regions1 * total_regions2
        
        # Hypergeometric test
        # Population size = total possible region pairs
        # Successes in population = expected overlaps
        # Sample size = total_regions1
        # Observed successes = overlapping_regions1
        
        if expected_overlaps > 0 and overlapping_regions1 > 0:
            try:
                p_value = hypergeom.sf(overlapping_regions1 - 1, 
                                     total_regions1 * total_regions2,
                                     expected_overlaps,
                                     total_regions1)
                fold_enrichment = overlapping_regions1 / expected_overlaps
            except:
                p_value = 1.0
                fold_enrichment = 1.0
        else:
            p_value = 1.0
            fold_enrichment = 1.0
        
        return {
            'observed_overlaps': overlapping_regions1,
            'expected_overlaps': expected_overlaps,
            'fold_enrichment': fold_enrichment,
            'p_value': p_value,
            'is_significant': p_value < self.enrichment_threshold
        }
    
    def analyze_treatment_correlations(self, treatment_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """
        Analyze correlations between different assays in response to AQR24h treatment.
        
        Args:
            treatment_data: Dictionary with structure {assay_type: {'gain': DataFrame, 'loss': DataFrame}}
            
        Returns:
            Correlation analysis results
        """
        logger.info("Analyzing treatment effect correlations between assays")
        
        results = {
            'pairwise_correlations': {},
            'feature_enrichment': {},
            'summary_statistics': {}
        }
        
        # Get all assay types with gain/loss data
        assay_types = list(treatment_data.keys())
        
        # Analyze pairwise correlations
        for assay1, assay2 in combinations(assay_types, 2):
            if 'gain' in treatment_data[assay1] and 'gain' in treatment_data[assay2]:
                correlation_result = self._analyze_assay_pair_correlation(
                    treatment_data[assay1], treatment_data[assay2], assay1, assay2
                )
                results['pairwise_correlations'][f"{assay1}_vs_{assay2}"] = correlation_result
        
        # Analyze feature enrichment across all assays
        results['feature_enrichment'] = self._analyze_feature_enrichment(treatment_data)
        
        # Calculate summary statistics
        results['summary_statistics'] = self._calculate_treatment_summary(treatment_data)
        
        return results
    
    def _analyze_assay_pair_correlation(self, data1: Dict[str, pd.DataFrame], 
                                      data2: Dict[str, pd.DataFrame],
                                      name1: str, name2: str) -> Dict:
        """Analyze correlation between two assay types."""
        
        correlation_result = {
            'assay1': name1,
            'assay2': name2,
            'gain_vs_gain': {},
            'loss_vs_loss': {},
            'gain_vs_loss': {},
            'cross_correlation': {}
        }
        
        # Analyze gain vs gain
        if 'gain' in data1 and 'gain' in data2:
            correlation_result['gain_vs_gain'] = self.calculate_region_overlaps(
                data1['gain'], data2['gain'], f"{name1}_gain", f"{name2}_gain"
            )
        
        # Analyze loss vs loss
        if 'loss' in data1 and 'loss' in data2:
            correlation_result['loss_vs_loss'] = self.calculate_region_overlaps(
                data1['loss'], data2['loss'], f"{name1}_loss", f"{name2}_loss"
            )
        
        # Analyze gain vs loss (anti-correlation)
        if 'gain' in data1 and 'loss' in data2:
            correlation_result['gain_vs_loss'] = self.calculate_region_overlaps(
                data1['gain'], data2['loss'], f"{name1}_gain", f"{name2}_loss"
            )
        
        # Calculate cross-correlation metrics
        correlation_result['cross_correlation'] = self._calculate_cross_correlation_metrics(
            correlation_result
        )
        
        return correlation_result
    
    def _calculate_cross_correlation_metrics(self, correlation_data: Dict) -> Dict:
        """Calculate summary metrics for cross-correlation analysis."""
        
        metrics = {}
        
        # Extract enrichment values
        gain_gain_enrichment = correlation_data.get('gain_vs_gain', {}).get('enrichment', {}).get('fold_enrichment', 1.0)
        loss_loss_enrichment = correlation_data.get('loss_vs_loss', {}).get('enrichment', {}).get('fold_enrichment', 1.0)
        gain_loss_enrichment = correlation_data.get('gain_vs_loss', {}).get('enrichment', {}).get('fold_enrichment', 1.0)
        
        # Calculate correlation strength
        positive_correlation_strength = (gain_gain_enrichment + loss_loss_enrichment) / 2
        negative_correlation_strength = gain_loss_enrichment
        
        metrics['positive_correlation'] = positive_correlation_strength
        metrics['negative_correlation'] = negative_correlation_strength
        metrics['correlation_direction'] = 'positive' if positive_correlation_strength > negative_correlation_strength else 'negative'
        metrics['correlation_strength'] = max(positive_correlation_strength, negative_correlation_strength)
        
        # Statistical significance
        gain_gain_pval = correlation_data.get('gain_vs_gain', {}).get('enrichment', {}).get('p_value', 1.0)
        loss_loss_pval = correlation_data.get('loss_vs_loss', {}).get('enrichment', {}).get('p_value', 1.0)
        
        metrics['min_p_value'] = min(gain_gain_pval, loss_loss_pval)
        metrics['is_significant'] = metrics['min_p_value'] < self.enrichment_threshold
        
        return metrics
    
    def _analyze_feature_enrichment(self, treatment_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """Analyze which genomic features are enriched in correlated regions."""
        
        feature_analysis = {}
        
        # Collect all regions by feature type and treatment effect
        all_features = {}
        
        for assay_type, conditions in treatment_data.items():
            for condition, data in conditions.items():
                if 'feature_type' in data.columns:
                    for feature_type in data['feature_type'].unique():
                        if feature_type not in all_features:
                            all_features[feature_type] = {'gain': [], 'loss': [], 'assays': set()}
                        
                        feature_regions = data[data['feature_type'] == feature_type]
                        
                        if 'gain' in condition:
                            all_features[feature_type]['gain'].extend(feature_regions.to_dict('records'))
                        elif 'loss' in condition:
                            all_features[feature_type]['loss'].extend(feature_regions.to_dict('records'))
                        
                        all_features[feature_type]['assays'].add(assay_type)
        
        # Calculate enrichment for each feature type
        for feature_type, feature_data in all_features.items():
            if len(feature_data['assays']) > 1:  # Only analyze features present in multiple assays
                
                gain_df = pd.DataFrame(feature_data['gain'])
                loss_df = pd.DataFrame(feature_data['loss'])
                
                if len(gain_df) > 0 and len(loss_df) > 0:
                    # Calculate overlap between gain and loss regions of the same feature type
                    overlap_stats = self.calculate_region_overlaps(
                        gain_df, loss_df, f"{feature_type}_gain", f"{feature_type}_loss"
                    )
                    
                    feature_analysis[feature_type] = {
                        'present_in_assays': list(feature_data['assays']),
                        'total_gain_regions': len(gain_df),
                        'total_loss_regions': len(loss_df),
                        'overlap_analysis': overlap_stats,
                        'consistency_score': self._calculate_feature_consistency(feature_data)
                    }
        
        return feature_analysis
    
    def _calculate_feature_consistency(self, feature_data: Dict) -> float:
        """Calculate how consistently a feature responds across assays."""
        
        total_assays = len(feature_data['assays'])
        gain_count = len(feature_data['gain'])
        loss_count = len(feature_data['loss'])
        
        if total_assays == 0:
            return 0.0
        
        # Consistency is higher when the feature shows the same direction across assays
        direction_consistency = abs(gain_count - loss_count) / (gain_count + loss_count) if (gain_count + loss_count) > 0 else 0
        
        return direction_consistency
    
    def _calculate_treatment_summary(self, treatment_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """Calculate summary statistics for treatment effects."""
        
        summary = {
            'total_assays': len(treatment_data),
            'assay_statistics': {},
            'overall_patterns': {}
        }
        
        total_gain_regions = 0
        total_loss_regions = 0
        
        for assay_type, conditions in treatment_data.items():
            assay_stats = {
                'gain_regions': len(conditions.get('gain', pd.DataFrame())),
                'loss_regions': len(conditions.get('loss', pd.DataFrame())),
                'total_regions': 0
            }
            
            assay_stats['total_regions'] = assay_stats['gain_regions'] + assay_stats['loss_regions']
            assay_stats['gain_fraction'] = assay_stats['gain_regions'] / assay_stats['total_regions'] if assay_stats['total_regions'] > 0 else 0
            
            summary['assay_statistics'][assay_type] = assay_stats
            
            total_gain_regions += assay_stats['gain_regions']
            total_loss_regions += assay_stats['loss_regions']
        
        summary['overall_patterns'] = {
            'total_gain_regions': total_gain_regions,
            'total_loss_regions': total_loss_regions,
            'overall_gain_fraction': total_gain_regions / (total_gain_regions + total_loss_regions) if (total_gain_regions + total_loss_regions) > 0 else 0
        }
        
        return summary

    def analyze_endseq_focus(self, treatment_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """
        Focused analysis on what causes END-seq breaks with AQR24h treatment.
        
        This method specifically examines END-seq data and correlates it with other assays
        to understand the mechanistic causes of transcription termination defects.
        """
        logger.info("Analyzing END-seq break patterns and correlations with AQR24h treatment")
        
        endseq_analysis = {
            'endseq_patterns': {},
            'mechanistic_correlations': {},
            'feature_specific_analysis': {},
            'causality_inference': {}
        }
        
        # Find END-seq data
        endseq_data = None
        endseq_key = None
        for key in treatment_data.keys():
            if 'END' in key.upper():
                endseq_data = treatment_data[key]
                endseq_key = key
                break
        
        if endseq_data is None:
            logger.warning("No END-seq data found for analysis")
            return endseq_analysis
        
        # Analyze END-seq patterns
        endseq_analysis['endseq_patterns'] = self._analyze_endseq_patterns(endseq_data, endseq_key)
        
        # Correlate with other assays to infer mechanisms
        other_assays = {k: v for k, v in treatment_data.items() if k != endseq_key}
        
        for assay_name, assay_data in other_assays.items():
            correlation = self._analyze_endseq_mechanism(endseq_data, assay_data, endseq_key, assay_name)
            endseq_analysis['mechanistic_correlations'][assay_name] = correlation
        
        # Feature-specific analysis (promoters, gene bodies, terminators)
        endseq_analysis['feature_specific_analysis'] = self._analyze_endseq_by_features(
            endseq_data, other_assays
        )
        
        # Infer causality relationships
        endseq_analysis['causality_inference'] = self._infer_endseq_causality(
            endseq_analysis['mechanistic_correlations']
        )
        
        return endseq_analysis
    
    def _analyze_endseq_patterns(self, endseq_data: Dict[str, pd.DataFrame], assay_name: str) -> Dict:
        """Analyze END-seq specific patterns."""
        
        patterns = {
            'assay_name': assay_name,
            'region_counts': {},
            'feature_distribution': {},
            'gene_distribution': {}
        }
        
        for condition, data in endseq_data.items():
            patterns['region_counts'][condition] = len(data)
            
            # Feature type distribution
            if 'feature_type' in data.columns:
                feature_counts = data['feature_type'].value_counts().to_dict()
                patterns['feature_distribution'][condition] = feature_counts
            
            # Gene distribution
            if 'gene_name' in data.columns:
                top_genes = data['gene_name'].value_counts().head(10).to_dict()
                patterns['gene_distribution'][condition] = top_genes
        
        return patterns
    
    def _analyze_endseq_mechanism(self, endseq_data: Dict[str, pd.DataFrame],
                                 other_assay_data: Dict[str, pd.DataFrame],
                                 endseq_name: str, other_name: str) -> Dict:
        """Analyze potential mechanisms by correlating END-seq with other assays."""
        
        mechanism_analysis = {
            'other_assay': other_name,
            'correlation_strength': 0,
            'temporal_relationship': 'unknown',
            'overlap_analysis': {},
            'mechanistic_hypothesis': ''
        }
        
        # Calculate overlaps between END-seq gain and other assay changes
        if 'gain' in endseq_data and 'gain' in other_assay_data:
            gain_overlap = self.calculate_region_overlaps(
                endseq_data['gain'], other_assay_data['gain'],
                f"{endseq_name}_gain", f"{other_name}_gain"
            )
            mechanism_analysis['overlap_analysis']['gain_vs_gain'] = gain_overlap
        
        if 'gain' in endseq_data and 'loss' in other_assay_data:
            gain_loss_overlap = self.calculate_region_overlaps(
                endseq_data['gain'], other_assay_data['loss'],
                f"{endseq_name}_gain", f"{other_name}_loss"
            )
            mechanism_analysis['overlap_analysis']['gain_vs_loss'] = gain_loss_overlap
        
        # Infer mechanistic relationships
        mechanism_analysis['mechanistic_hypothesis'] = self._generate_mechanistic_hypothesis(
            endseq_name, other_name, mechanism_analysis['overlap_analysis']
        )
        
        return mechanism_analysis
    
    def _generate_mechanistic_hypothesis(self, endseq_name: str, other_name: str, 
                                       overlap_analysis: Dict) -> str:
        """Generate mechanistic hypothesis based on overlap patterns."""
        
        hypotheses = []
        
        # Check for co-occurrence (both increase)
        gain_gain = overlap_analysis.get('gain_vs_gain', {})
        if gain_gain.get('enrichment', {}).get('is_significant', False):
            if 'sDRIP' in other_name:
                hypotheses.append("R-loop formation may contribute to transcription termination defects")
            elif 'EU' in other_name:
                hypotheses.append("Nascent RNA processing defects may cause termination failures")
            elif 'RNA' in other_name:
                hypotheses.append("RNA stability changes correlate with termination efficiency")
        
        # Check for anti-correlation (END-seq gain, other loss)
        gain_loss = overlap_analysis.get('gain_vs_loss', {})
        if gain_loss.get('enrichment', {}).get('is_significant', False):
            if 'sDRIP' in other_name:
                hypotheses.append("Loss of R-loops associated with increased termination read-through")
            elif 'EU' in other_name:
                hypotheses.append("Decreased nascent transcription where termination fails")
        
        if not hypotheses:
            hypotheses.append(f"No clear mechanistic relationship with {other_name}")
        
        return "; ".join(hypotheses)
    
    def _analyze_endseq_by_features(self, endseq_data: Dict[str, pd.DataFrame],
                                  other_assays: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """Analyze END-seq patterns by genomic features."""
        
        feature_analysis = {}
        
        if 'gain' not in endseq_data:
            return feature_analysis
        
        endseq_gain = endseq_data['gain']
        
        # Group by feature type
        if 'feature_type' in endseq_gain.columns:
            for feature_type in endseq_gain['feature_type'].unique():
                feature_regions = endseq_gain[endseq_gain['feature_type'] == feature_type]
                
                feature_analysis[feature_type] = {
                    'endseq_count': len(feature_regions),
                    'correlations_with_other_assays': {}
                }
                
                # Check correlations with other assays for this feature
                for assay_name, assay_data in other_assays.items():
                    if 'gain' in assay_data:
                        other_gain = assay_data['gain']
                        if 'feature_type' in other_gain.columns:
                            other_feature_regions = other_gain[other_gain['feature_type'] == feature_type]
                            
                            if len(other_feature_regions) > 0:
                                overlap_stats = self.calculate_region_overlaps(
                                    feature_regions, other_feature_regions,
                                    f"ENDseq_{feature_type}", f"{assay_name}_{feature_type}"
                                )
                                
                                feature_analysis[feature_type]['correlations_with_other_assays'][assay_name] = overlap_stats
        
        return feature_analysis
    
    def _infer_endseq_causality(self, mechanistic_correlations: Dict) -> Dict:
        """Infer likely causal relationships for END-seq breaks."""
        
        causality_analysis = {
            'primary_causes': [],
            'secondary_effects': [],
            'confidence_scores': {}
        }
        
        # Rank potential causes by correlation strength and biological plausibility
        cause_scores = {}
        
        for assay_name, correlation_data in mechanistic_correlations.items():
            score = 0
            
            # Check overlap enrichment
            gain_gain = correlation_data.get('overlap_analysis', {}).get('gain_vs_gain', {})
            if gain_gain.get('enrichment', {}).get('is_significant', False):
                score += gain_gain.get('enrichment', {}).get('fold_enrichment', 1) * 2
            
            # Weight by biological plausibility
            if 'sDRIP' in assay_name:
                score *= 1.5  # R-loops are known to affect termination
            elif 'EU' in assay_name:
                score *= 1.3  # Nascent RNA processing is relevant
            elif 'RNA' in assay_name:
                score *= 1.1  # RNA levels are downstream effects
            
            cause_scores[assay_name] = score
        
        # Categorize causes
        sorted_causes = sorted(cause_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (assay_name, score) in enumerate(sorted_causes):
            causality_analysis['confidence_scores'][assay_name] = score
            
            if i == 0 and score > 2.0:  # Strong primary cause
                causality_analysis['primary_causes'].append(assay_name)
            elif score > 1.5:  # Moderate correlation
                causality_analysis['secondary_effects'].append(assay_name)
        
        return causality_analysis