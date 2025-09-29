"""
Multi-omics visualization module for genomic correlation analysis.

Specialized plotting functions for analyzing correlations between different genomic assays.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MultiOmicsPlotter:
    """
    Specialized plotter for multi-omics genomic correlation analysis.
    """
    
    def __init__(self, config: Dict, results_dir: Path):
        self.config = config
        self.results_dir = results_dir
        self.color_palette = config.get('visualization', {}).get('color_palette', 'Set1')
        
    def create_endseq_correlation_report(self, genomic_analysis_results: Dict) -> Dict:
        """
        Create comprehensive visualization report for END-seq correlation analysis.
        
        Args:
            genomic_analysis_results: Results from GenomicRegionAnalyzer
            
        Returns:
            Dictionary with paths to created plots
        """
        plot_paths = {}
        
        # 1. Overview correlation matrix
        plot_paths['correlation_matrix'] = self._plot_correlation_matrix(
            genomic_analysis_results.get('pairwise_correlations', {})
        )
        
        # 2. Feature enrichment analysis
        plot_paths['feature_enrichment'] = self._plot_feature_enrichment(
            genomic_analysis_results.get('feature_enrichment', {})
        )
        
        # 3. END-seq specific analysis
        if 'endseq_analysis' in genomic_analysis_results:
            plot_paths.update(self._plot_endseq_analysis(
                genomic_analysis_results['endseq_analysis']
            ))
        
        # 4. Treatment effect summary
        plot_paths['treatment_summary'] = self._plot_treatment_summary(
            genomic_analysis_results.get('summary_statistics', {})
        )
        
        return plot_paths
    
    def _plot_correlation_matrix(self, pairwise_correlations: Dict) -> str:
        """Create correlation matrix heatmap."""
        if not pairwise_correlations:
            return None
        
        # Extract correlation strengths
        assay_pairs = []
        correlation_strengths = []
        p_values = []
        
        for pair_name, correlation_data in pairwise_correlations.items():
            cross_corr = correlation_data.get('cross_correlation', {})
            
            assay_pairs.append(pair_name.replace('_vs_', ' vs '))
            correlation_strengths.append(cross_corr.get('correlation_strength', 0))
            p_values.append(cross_corr.get('min_p_value', 1.0))
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation strength heatmap
        correlation_matrix = np.array(correlation_strengths).reshape(-1, 1)
        im1 = ax1.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto')
        ax1.set_yticks(range(len(assay_pairs)))
        ax1.set_yticklabels(assay_pairs)
        ax1.set_xlabel('Correlation Strength')
        ax1.set_title('Assay Correlation Strengths')
        plt.colorbar(im1, ax=ax1)
        
        # Significance heatmap
        significance_matrix = np.array([-np.log10(p + 1e-10) for p in p_values]).reshape(-1, 1)
        im2 = ax2.imshow(significance_matrix, cmap='Reds', aspect='auto')
        ax2.set_yticks(range(len(assay_pairs)))
        ax2.set_yticklabels(assay_pairs)
        ax2.set_xlabel('-log10(p-value)')
        ax2.set_title('Statistical Significance')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        filename = 'multiomics_correlation_matrix'
        filepath = self.results_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_feature_enrichment(self, feature_enrichment: Dict) -> str:
        """Plot genomic feature enrichment analysis."""
        if not feature_enrichment:
            return None
        
        features = list(feature_enrichment.keys())
        consistency_scores = []
        gain_counts = []
        loss_counts = []
        
        for feature, data in feature_enrichment.items():
            consistency_scores.append(data.get('consistency_score', 0))
            gain_counts.append(data.get('total_gain_regions', 0))
            loss_counts.append(data.get('total_loss_regions', 0))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Consistency scores
        bars1 = ax1.bar(features, consistency_scores, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Consistency Score')
        ax1.set_title('Feature Response Consistency Across Assays')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Gain vs Loss counts
        x_pos = np.arange(len(features))
        width = 0.35
        
        bars2 = ax2.bar(x_pos - width/2, gain_counts, width, label='Gain', color='red', alpha=0.7)
        bars3 = ax2.bar(x_pos + width/2, loss_counts, width, label='Loss', color='blue', alpha=0.7)
        ax2.set_ylabel('Region Count')
        ax2.set_title('Gain vs Loss Regions by Feature Type')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(features, rotation=45)
        ax2.legend()
        
        # 3. Feature distribution pie chart
        feature_totals = [g + l for g, l in zip(gain_counts, loss_counts)]
        if sum(feature_totals) > 0:
            ax3.pie(feature_totals, labels=features, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Distribution of Affected Regions by Feature Type')
        
        # 4. Gain/Loss ratio
        ratios = [g / (l + 1) for g, l in zip(gain_counts, loss_counts)]  # +1 to avoid division by zero
        bars4 = ax4.bar(features, ratios, color='purple', alpha=0.7)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Gain/Loss Ratio')
        ax4.set_title('Gain/Loss Ratio by Feature Type')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        filename = 'feature_enrichment_analysis'
        filepath = self.results_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_endseq_analysis(self, endseq_analysis: Dict) -> Dict:
        """Create END-seq specific analysis plots."""
        plot_paths = {}
        
        # 1. END-seq pattern analysis
        plot_paths['endseq_patterns'] = self._plot_endseq_patterns(
            endseq_analysis.get('endseq_patterns', {})
        )
        
        # 2. Mechanistic correlations
        plot_paths['mechanistic_correlations'] = self._plot_mechanistic_correlations(
            endseq_analysis.get('mechanistic_correlations', {})
        )
        
        # 3. Causality inference
        plot_paths['causality_analysis'] = self._plot_causality_analysis(
            endseq_analysis.get('causality_inference', {})
        )
        
        return plot_paths
    
    def _plot_endseq_patterns(self, endseq_patterns: Dict) -> str:
        """Plot END-seq specific patterns."""
        if not endseq_patterns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Region counts by condition
        region_counts = endseq_patterns.get('region_counts', {})
        if region_counts:
            conditions = list(region_counts.keys())
            counts = list(region_counts.values())
            
            bars1 = ax1.bar(conditions, counts, color='darkred', alpha=0.7)
            ax1.set_ylabel('Number of Regions')
            ax1.set_title('END-seq Regions by Condition')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars1, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                        str(count), ha='center', va='bottom')
        
        # 2. Feature distribution for gain condition
        feature_dist = endseq_patterns.get('feature_distribution', {})
        if 'gain' in feature_dist:
            features = list(feature_dist['gain'].keys())
            counts = list(feature_dist['gain'].values())
            
            if features and counts:
                ax2.pie(counts, labels=features, autopct='%1.1f%%', startangle=90)
                ax2.set_title('END-seq Gain: Feature Type Distribution')
        
        # 3. Top affected genes
        gene_dist = endseq_patterns.get('gene_distribution', {})
        if 'gain' in gene_dist:
            genes = list(gene_dist['gain'].keys())[:10]  # Top 10
            counts = list(gene_dist['gain'].values())[:10]
            
            if genes and counts:
                bars3 = ax3.barh(genes, counts, color='orange', alpha=0.7)
                ax3.set_xlabel('Number of Regions')
                ax3.set_title('Top 10 Genes with END-seq Gains')
        
        # 4. Treatment effect comparison
        if len(region_counts) >= 2:
            conditions = list(region_counts.keys())
            values = list(region_counts.values())
            
            # Create comparison plot
            x_pos = np.arange(len(conditions))
            bars4 = ax4.bar(x_pos, values, color=['red', 'blue', 'green'][:len(conditions)], alpha=0.7)
            ax4.set_ylabel('Number of Regions')
            ax4.set_title('END-seq Treatment Effects')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(conditions, rotation=45)
        
        plt.tight_layout()
        
        filename = 'endseq_pattern_analysis'
        filepath = self.results_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_mechanistic_correlations(self, mechanistic_correlations: Dict) -> str:
        """Plot mechanistic correlation analysis."""
        if not mechanistic_correlations:
            return None
        
        assays = list(mechanistic_correlations.keys())
        
        # Extract enrichment data
        enrichment_data = []
        hypothesis_data = []
        
        for assay, data in mechanistic_correlations.items():
            overlap_analysis = data.get('overlap_analysis', {})
            
            # Get gain vs gain enrichment
            gain_gain = overlap_analysis.get('gain_vs_gain', {})
            enrichment = gain_gain.get('enrichment', {})
            fold_enrichment = enrichment.get('fold_enrichment', 1.0)
            p_value = enrichment.get('p_value', 1.0)
            
            enrichment_data.append({
                'assay': assay,
                'fold_enrichment': fold_enrichment,
                'neg_log_p': -np.log10(p_value + 1e-10),
                'is_significant': enrichment.get('is_significant', False)
            })
            
            hypothesis_data.append({
                'assay': assay,
                'hypothesis': data.get('mechanistic_hypothesis', 'No clear relationship')
            })
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Enrichment volcano plot
        enrichment_df = pd.DataFrame(enrichment_data)
        if not enrichment_df.empty:
            colors = ['red' if sig else 'blue' for sig in enrichment_df['is_significant']]
            
            scatter = ax1.scatter(enrichment_df['fold_enrichment'], enrichment_df['neg_log_p'],
                                c=colors, alpha=0.7, s=100)
            
            # Add significance thresholds
            ax1.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
            ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
            
            # Label points
            for i, row in enrichment_df.iterrows():
                ax1.annotate(row['assay'], (row['fold_enrichment'], row['neg_log_p']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax1.set_xlabel('Fold Enrichment')
            ax1.set_ylabel('-log10(p-value)')
            ax1.set_title('END-seq Mechanistic Correlations')
        
        # 2. Mechanistic hypothesis summary
        ax2.axis('off')
        y_pos = 0.9
        
        ax2.text(0.05, y_pos, 'Mechanistic Hypotheses:', fontsize=14, weight='bold',
                transform=ax2.transAxes)
        y_pos -= 0.1
        
        for i, hyp_data in enumerate(hypothesis_data):
            assay_text = f"{hyp_data['assay']}:"
            hypothesis_text = hyp_data['hypothesis']
            
            # Wrap long text
            if len(hypothesis_text) > 60:
                hypothesis_text = hypothesis_text[:60] + "..."
            
            ax2.text(0.05, y_pos, assay_text, fontsize=10, weight='bold',
                    transform=ax2.transAxes)
            y_pos -= 0.05
            ax2.text(0.1, y_pos, hypothesis_text, fontsize=9,
                    transform=ax2.transAxes, wrap=True)
            y_pos -= 0.12
        
        plt.tight_layout()
        
        filename = 'mechanistic_correlations'
        filepath = self.results_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_causality_analysis(self, causality_inference: Dict) -> str:
        """Plot causality inference analysis."""
        if not causality_inference:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Confidence scores
        confidence_scores = causality_inference.get('confidence_scores', {})
        if confidence_scores:
            assays = list(confidence_scores.keys())
            scores = list(confidence_scores.values())
            
            # Color code by causality category
            primary_causes = set(causality_inference.get('primary_causes', []))
            secondary_effects = set(causality_inference.get('secondary_effects', []))
            
            colors = []
            for assay in assays:
                if assay in primary_causes:
                    colors.append('red')
                elif assay in secondary_effects:
                    colors.append('orange')
                else:
                    colors.append('gray')
            
            bars1 = ax1.barh(assays, scores, color=colors, alpha=0.7)
            ax1.set_xlabel('Causality Confidence Score')
            ax1.set_title('Inferred Causal Relationships with END-seq Breaks')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label='Primary Cause'),
                Patch(facecolor='orange', alpha=0.7, label='Secondary Effect'),
                Patch(facecolor='gray', alpha=0.7, label='Weak/No Evidence')
            ]
            ax1.legend(handles=legend_elements, loc='lower right')
        
        # 2. Causality network diagram (simplified)
        ax2.axis('off')
        
        # Create a simple network visualization
        if confidence_scores:
            # Central node (END-seq)
            center_x, center_y = 0.5, 0.5
            circle = plt.Circle((center_x, center_y), 0.1, color='darkred', alpha=0.8)
            ax2.add_patch(circle)
            ax2.text(center_x, center_y, 'END-seq\nBreaks', ha='center', va='center',
                    fontsize=8, color='white', weight='bold')
            
            # Surrounding nodes for other assays
            n_assays = len(assays)
            angles = np.linspace(0, 2*np.pi, n_assays, endpoint=False)
            
            for i, (assay, score) in enumerate(zip(assays, scores)):
                angle = angles[i]
                x = center_x + 0.3 * np.cos(angle)
                y = center_y + 0.3 * np.sin(angle)
                
                # Color and size based on confidence
                if assay in primary_causes:
                    color = 'red'
                    size = 0.08
                elif assay in secondary_effects:
                    color = 'orange'
                    size = 0.06
                else:
                    color = 'lightgray'
                    size = 0.04
                
                circle = plt.Circle((x, y), size, color=color, alpha=0.7)
                ax2.add_patch(circle)
                ax2.text(x, y, assay, ha='center', va='center', fontsize=6, weight='bold')
                
                # Draw connection line with width proportional to confidence
                line_width = max(1, score * 3)
                ax2.plot([center_x, x], [center_y, y], 'k-', alpha=0.5, linewidth=line_width)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Causal Network for END-seq Breaks')
        
        plt.tight_layout()
        
        filename = 'causality_analysis'
        filepath = self.results_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_treatment_summary(self, summary_statistics: Dict) -> str:
        """Plot treatment effect summary."""
        if not summary_statistics:
            return None
        
        assay_stats = summary_statistics.get('assay_statistics', {})
        if not assay_stats:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        assays = list(assay_stats.keys())
        gain_regions = [stats.get('gain_regions', 0) for stats in assay_stats.values()]
        loss_regions = [stats.get('loss_regions', 0) for stats in assay_stats.values()]
        total_regions = [stats.get('total_regions', 0) for stats in assay_stats.values()]
        gain_fractions = [stats.get('gain_fraction', 0) for stats in assay_stats.values()]
        
        # 1. Total regions per assay
        bars1 = ax1.bar(assays, total_regions, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Total Regions')
        ax1.set_title('Total Affected Regions by Assay Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, total in zip(bars1, total_regions):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(total_regions),
                    str(total), ha='center', va='bottom')
        
        # 2. Gain vs Loss stacked bar
        x_pos = np.arange(len(assays))
        bars2a = ax2.bar(x_pos, gain_regions, label='Gain', color='red', alpha=0.7)
        bars2b = ax2.bar(x_pos, loss_regions, bottom=gain_regions, label='Loss', color='blue', alpha=0.7)
        
        ax2.set_ylabel('Number of Regions')
        ax2.set_title('Gain vs Loss Regions by Assay')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(assays, rotation=45)
        ax2.legend()
        
        # 3. Gain fraction comparison
        bars3 = ax3.bar(assays, gain_fractions, color='purple', alpha=0.7)
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Gain Fraction')
        ax3.set_title('Proportion of Gain vs Loss by Assay')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
        
        # 4. Overall treatment effect pie chart
        overall_patterns = summary_statistics.get('overall_patterns', {})
        total_gain = overall_patterns.get('total_gain_regions', 0)
        total_loss = overall_patterns.get('total_loss_regions', 0)
        
        if total_gain > 0 or total_loss > 0:
            ax4.pie([total_gain, total_loss], labels=['Gain', 'Loss'],
                   colors=['red', 'blue'], autopct='%1.1f%%', alpha=0.7)
            ax4.set_title('Overall AQR24h Treatment Effect')
        
        plt.tight_layout()
        
        filename = 'treatment_effect_summary'
        filepath = self.results_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)