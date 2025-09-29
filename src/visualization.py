"""
Visualization module for genomics correlation analysis.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class GenomicsPlotter:
    """
    Main visualization class for genomics correlation analysis.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_formats = config['visualization']['output_format']
        self.dpi = config['visualization']['dpi']
        self.figure_size = config['visualization']['figure_size']
        self.color_palette = config['visualization']['color_palette']
        self.results_dir = Path(config['output']['results_dir'])
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette(self.color_palette)
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize multi-omics plotter
        from .multiomics_visualization import MultiOmicsPlotter
        self.multiomics_plotter = MultiOmicsPlotter(config, self.results_dir)
        
    def create_comprehensive_report(self, analysis_results: Dict, 
                                  snp_data: pd.DataFrame, 
                                  methylation_data: pd.DataFrame) -> Dict:
        """
        Create a comprehensive visualization report of all analysis results.
        
        Args:
            analysis_results: Results from correlation analysis
            snp_data: Original SNP data
            methylation_data: Original methylation data
            
        Returns:
            Dictionary with paths to generated plots
        """
        plot_paths = {}
        
        logger.info("Creating comprehensive visualization report...")
        
        # Statistical correlation plots
        if 'statistical' in analysis_results:
            stat_plots = self._create_statistical_plots(analysis_results['statistical'])
            plot_paths.update(stat_plots)
        
        # Machine learning plots
        if 'machine_learning' in analysis_results:
            ml_plots = self._create_ml_plots(analysis_results['machine_learning'])
            plot_paths.update(ml_plots)
        
        # Window-based analysis plots
        if 'window_based' in analysis_results:
            window_plots = self._create_window_plots(analysis_results['window_based'])
            plot_paths.update(window_plots)
        
        # Data overview plots
        overview_plots = self._create_data_overview_plots(snp_data, methylation_data)
        plot_paths.update(overview_plots)
        
        # Summary dashboard
        dashboard_path = self._create_summary_dashboard(analysis_results)
        if dashboard_path:
            plot_paths['summary_dashboard'] = dashboard_path
        
        return plot_paths
    
    def _create_statistical_plots(self, statistical_results: Dict) -> Dict:
        """Create plots for statistical correlation analysis."""
        plot_paths = {}
        
        # Pairwise correlation heatmap
        if 'pairwise_correlations' in statistical_results:
            pairwise_data = statistical_results['pairwise_correlations']
            
            if isinstance(pairwise_data, pd.DataFrame) and len(pairwise_data) > 0:
                # Correlation distribution plot
                path = self._plot_correlation_distribution(pairwise_data)
                if path:
                    plot_paths['correlation_distribution'] = path
                
                # Manhattan plot
                path = self._plot_manhattan(pairwise_data)
                if path:
                    plot_paths['manhattan_plot'] = path
                
                # Top correlations heatmap
                path = self._plot_top_correlations_heatmap(pairwise_data)
                if path:
                    plot_paths['top_correlations_heatmap'] = path
        
        # Global correlation patterns
        if 'global_patterns' in statistical_results:
            global_data = statistical_results['global_patterns']
            
            # Principal component correlation plot
            path = self._plot_pc_correlations(global_data)
            if path:
                plot_paths['pc_correlations'] = path
        
        # Distance-based correlations
        if 'distance_correlations' in statistical_results:
            distance_data = statistical_results['distance_correlations']
            
            # LD decay plot
            path = self._plot_ld_decay(distance_data)
            if path:
                plot_paths['ld_decay'] = path
        
        return plot_paths
    
    def _create_ml_plots(self, ml_results: Dict) -> Dict:
        """Create plots for machine learning analysis."""
        plot_paths = {}
        
        # Feature importance plots
        if 'feature_importance' in ml_results:
            fi_data = ml_results['feature_importance']
            
            # Feature importance bar plot
            path = self._plot_feature_importance(fi_data)
            if path:
                plot_paths['feature_importance'] = path
        
        # Mutual information plots
        if 'mutual_information' in ml_results:
            mi_data = ml_results['mutual_information']
            
            # MI distribution plot
            path = self._plot_mutual_information(mi_data)
            if path:
                plot_paths['mutual_information'] = path
        
        # Non-linear associations
        if 'nonlinear_associations' in ml_results:
            nl_data = ml_results['nonlinear_associations']
            
            # Non-linear model performance
            path = self._plot_nonlinear_performance(nl_data)
            if path:
                plot_paths['nonlinear_performance'] = path
        
        return plot_paths
    
    def _create_window_plots(self, window_results: Dict) -> Dict:
        """Create plots for window-based analysis."""
        plot_paths = {}
        
        if 'chromosome_analysis' in window_results:
            # Genomic landscape plot
            path = self._plot_genomic_landscape(window_results)
            if path:
                plot_paths['genomic_landscape'] = path
            
            # Chromosome-specific plots
            path = self._plot_chromosome_correlations(window_results['chromosome_analysis'])
            if path:
                plot_paths['chromosome_correlations'] = path
        
        return plot_paths
    
    def _create_data_overview_plots(self, snp_data: pd.DataFrame, 
                                  methylation_data: pd.DataFrame) -> Dict:
        """Create overview plots of the input data."""
        plot_paths = {}
        
        # SNP data overview
        path = self._plot_snp_overview(snp_data)
        if path:
            plot_paths['snp_overview'] = path
        
        # Methylation data overview
        path = self._plot_methylation_overview(methylation_data)
        if path:
            plot_paths['methylation_overview'] = path
        
        # Data quality metrics
        path = self._plot_data_quality(snp_data, methylation_data)
        if path:
            plot_paths['data_quality'] = path
        
        return plot_paths
    
    def _plot_correlation_distribution(self, pairwise_data: pd.DataFrame) -> Optional[str]:
        """Plot distribution of correlation coefficients."""
        try:
            # Find the primary correlation column
            corr_cols = [col for col in pairwise_data.columns if col.endswith('_corr')]
            if not corr_cols:
                return None
            
            primary_corr_col = corr_cols[0]
            correlations = pairwise_data[primary_corr_col].dropna()
            
            if len(correlations) == 0:
                return None
            
            # Create subplot with histogram and box plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, height_ratios=[3, 1])
            
            # Histogram
            ax1.hist(correlations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(correlations.mean(), color='red', linestyle='--', 
                       label=f'Mean: {correlations.mean():.3f}')
            ax1.axvline(correlations.median(), color='green', linestyle='--', 
                       label=f'Median: {correlations.median():.3f}')
            ax1.set_xlabel('Correlation Coefficient')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of SNP-Methylation Correlations')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(correlations, vert=False, widths=0.7)
            ax2.set_xlabel('Correlation Coefficient')
            ax2.set_yticks([])
            
            plt.tight_layout()
            
            # Save plot
            filename = 'correlation_distribution'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create correlation distribution plot: {e}")
            return None
    
    def _plot_manhattan(self, pairwise_data: pd.DataFrame) -> Optional[str]:
        """Create Manhattan plot of correlation p-values."""
        try:
            # Check for required columns
            if 'snp_chr' not in pairwise_data.columns or 'snp_pos' not in pairwise_data.columns:
                return None
            
            pval_cols = [col for col in pairwise_data.columns if col.endswith('_pval')]
            if not pval_cols:
                return None
            
            primary_pval_col = pval_cols[0]
            
            # Prepare data
            plot_data = pairwise_data[['snp_chr', 'snp_pos', primary_pval_col]].copy()
            plot_data = plot_data.dropna()
            
            if len(plot_data) == 0:
                return None
            
            # Calculate -log10 p-values
            plot_data['neg_log10_p'] = -np.log10(plot_data[primary_pval_col].clip(lower=1e-50))
            
            # Create Manhattan plot using plotly
            fig = px.scatter(
                plot_data, 
                x='snp_pos', 
                y='neg_log10_p',
                color='snp_chr',
                title='Manhattan Plot of SNP-Methylation Associations',
                labels={
                    'snp_pos': 'Genomic Position',
                    'neg_log10_p': '-log10(p-value)',
                    'snp_chr': 'Chromosome'
                },
                hover_data=['snp_chr', 'snp_pos']
            )
            
            # Add significance line
            significance_threshold = -np.log10(0.05)
            fig.add_hline(y=significance_threshold, line_dash="dash", 
                         annotation_text="p=0.05")
            
            fig.update_layout(
                width=1200,
                height=600,
                showlegend=True
            )
            
            # Save plot
            filename = 'manhattan_plot'
            path = self._save_plotly_figure(fig, filename)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create Manhattan plot: {e}")
            return None
    
    def _plot_top_correlations_heatmap(self, pairwise_data: pd.DataFrame) -> Optional[str]:
        """Create heatmap of top correlations."""
        try:
            # Get top correlations
            if 'abs_correlation' not in pairwise_data.columns:
                return None
            
            top_correlations = pairwise_data.nlargest(50, 'abs_correlation')
            
            if len(top_correlations) == 0:
                return None
            
            # Create pivot table for heatmap
            corr_col = [col for col in top_correlations.columns if col.endswith('_corr')][0]
            
            # Since we might have many-to-many relationships, create a matrix
            pivot_data = top_correlations.pivot_table(
                values=corr_col, 
                index='snp_id', 
                columns='cpg_id', 
                aggfunc='first'
            )
            
            if pivot_data.empty:
                return None
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sns.heatmap(
                pivot_data, 
                annot=False, 
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax
            )
            
            ax.set_title('Top SNP-Methylation Correlations')
            ax.set_xlabel('CpG Sites')
            ax.set_ylabel('SNPs')
            
            plt.tight_layout()
            
            # Save plot
            filename = 'top_correlations_heatmap'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create top correlations heatmap: {e}")
            return None
    
    def _plot_pc_correlations(self, global_data: Dict) -> Optional[str]:
        """Plot principal component correlations."""
        try:
            if 'pc_correlations' not in global_data:
                return None
            
            pc_corr_matrix = global_data['pc_correlations']
            
            # Create heatmap of PC correlations
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(
                pc_corr_matrix,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation'},
                ax=ax
            )
            
            ax.set_title('Correlation between SNP and Methylation Principal Components')
            ax.set_xlabel('Methylation PCs')
            ax.set_ylabel('SNP PCs')
            
            plt.tight_layout()
            
            # Save plot
            filename = 'pc_correlations'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create PC correlation plot: {e}")
            return None
    
    def _plot_ld_decay(self, distance_data: Dict) -> Optional[str]:
        """Plot linkage disequilibrium decay with distance."""
        try:
            if 'distance_correlation_decay' not in distance_data:
                return None
            
            decay_data = distance_data['distance_correlation_decay']
            
            # Create LD decay plot
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot mean correlation vs distance
            x_pos = range(len(decay_data))
            ax.errorbar(x_pos, decay_data['mean'], yerr=decay_data['std'], 
                       marker='o', capsize=5, capthick=2, linewidth=2)
            
            ax.set_xlabel('Distance Bin')
            ax.set_ylabel('Mean Correlation')
            ax.set_title('Linkage Disequilibrium Decay with Genomic Distance')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(interval) for interval in decay_data['distance_bin']], 
                              rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            filename = 'ld_decay'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create LD decay plot: {e}")
            return None
    
    def _plot_feature_importance(self, fi_data: Dict) -> Optional[str]:
        """Plot feature importance from machine learning analysis."""
        try:
            if 'feature_importance_df' not in fi_data:
                return None
            
            importance_df = fi_data['feature_importance_df']
            
            # Plot top features
            top_features = importance_df.head(20)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            bars = ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 20 SNP Features by Importance (Random Forest)')
            
            # Color bars by importance
            colors = plt.cm.viridis(top_features['importance'] / top_features['importance'].max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            
            # Save plot
            filename = 'feature_importance'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create feature importance plot: {e}")
            return None
    
    def _plot_mutual_information(self, mi_data: Dict) -> Optional[str]:
        """Plot mutual information results."""
        try:
            if 'mutual_information_df' not in mi_data:
                return None
            
            mi_df = mi_data['mutual_information_df']
            
            # Distribution of MI scores
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram of MI scores
            ax1.hist(mi_df['mutual_information'], bins=30, alpha=0.7, color='lightcoral')
            ax1.set_xlabel('Mutual Information Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Mutual Information Scores')
            ax1.grid(True, alpha=0.3)
            
            # Top MI features
            top_mi = mi_df.head(15)
            bars = ax2.barh(range(len(top_mi)), top_mi['mutual_information'])
            ax2.set_yticks(range(len(top_mi)))
            ax2.set_yticklabels(top_mi['feature'])
            ax2.set_xlabel('Mutual Information Score')
            ax2.set_title('Top 15 Features by Mutual Information')
            
            # Color bars
            colors = plt.cm.plasma(top_mi['mutual_information'] / top_mi['mutual_information'].max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            
            # Save plot
            filename = 'mutual_information'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create mutual information plot: {e}")
            return None
    
    def _plot_nonlinear_performance(self, nl_data: Dict) -> Optional[str]:
        """Plot non-linear model performance."""
        try:
            if 'nonlinear_models' not in nl_data:
                return None
            
            model_data = nl_data['nonlinear_models']
            
            if not model_data:
                return None
            
            # Extract R2 scores
            r2_scores = [model['r2_score'] for model in model_data]
            n_features = [model['n_selected_features'] for model in model_data]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # R2 scores
            ax1.bar(range(len(r2_scores)), r2_scores, color='lightgreen')
            ax1.set_xlabel('Target Index')
            ax1.set_ylabel('R² Score')
            ax1.set_title('Non-linear Model Performance (Elastic Net)')
            ax1.grid(True, alpha=0.3)
            
            # Number of selected features
            ax2.bar(range(len(n_features)), n_features, color='orange')
            ax2.set_xlabel('Target Index')
            ax2.set_ylabel('Number of Selected Features')
            ax2.set_title('Feature Selection (Non-linear Models)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filename = 'nonlinear_performance'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create non-linear performance plot: {e}")
            return None
    
    def _plot_genomic_landscape(self, window_results: Dict) -> Optional[str]:
        """Plot genomic landscape of correlations."""
        try:
            if 'chromosome_analysis' not in window_results:
                return None
            
            chrom_data = window_results['chromosome_analysis']
            
            # Collect all window data
            all_windows = []
            for chrom, data in chrom_data.items():
                if 'windows' in data:
                    for window in data['windows']:
                        window['chromosome'] = chrom
                        all_windows.append(window)
            
            if not all_windows:
                return None
            
            # Create DataFrame
            windows_df = pd.DataFrame(all_windows)
            
            # Create genomic landscape plot using plotly
            fig = px.scatter(
                windows_df,
                x='start_pos',
                y='max_correlation',
                color='chromosome',
                size='n_snps',
                title='Genomic Landscape of SNP-Methylation Correlations',
                labels={
                    'start_pos': 'Genomic Position',
                    'max_correlation': 'Maximum Correlation in Window',
                    'chromosome': 'Chromosome',
                    'n_snps': 'Number of SNPs'
                },
                hover_data=['end_pos', 'n_cpgs', 'mean_correlation']
            )
            
            fig.update_layout(
                width=1200,
                height=600,
                showlegend=True
            )
            
            # Save plot
            filename = 'genomic_landscape'
            path = self._save_plotly_figure(fig, filename)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create genomic landscape plot: {e}")
            return None
    
    def _plot_chromosome_correlations(self, chrom_analysis: Dict) -> Optional[str]:
        """Plot chromosome-specific correlation patterns."""
        try:
            # Create subplot for each chromosome
            n_chroms = len(chrom_analysis)
            if n_chroms == 0:
                return None
            
            cols = min(3, n_chroms)
            rows = (n_chroms + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            if n_chroms == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            axes_flat = axes.flatten() if n_chroms > 1 else axes
            
            for i, (chrom, data) in enumerate(chrom_analysis.items()):
                if i >= len(axes_flat):
                    break
                    
                ax = axes_flat[i]
                
                if 'windows' in data and data['windows']:
                    windows = data['windows']
                    positions = [w['start_pos'] for w in windows]
                    correlations = [w['max_correlation'] for w in windows]
                    
                    ax.plot(positions, correlations, 'o-', alpha=0.7)
                    ax.set_xlabel('Genomic Position')
                    ax.set_ylabel('Max Correlation')
                    ax.set_title(f'Chromosome {chrom}')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'Chromosome {chrom}')
            
            # Hide empty subplots
            for i in range(n_chroms, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filename = 'chromosome_correlations'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create chromosome correlation plot: {e}")
            return None
    
    def _plot_snp_overview(self, snp_data: pd.DataFrame) -> Optional[str]:
        """Create SNP data overview plot."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # MAF distribution
            if 'maf' in snp_data.columns:
                axes[0, 0].hist(snp_data['maf'].dropna(), bins=30, alpha=0.7, color='skyblue')
                axes[0, 0].set_xlabel('Minor Allele Frequency')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('MAF Distribution')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Quality score distribution
            if 'qual' in snp_data.columns:
                axes[0, 1].hist(snp_data['qual'].dropna(), bins=30, alpha=0.7, color='lightcoral')
                axes[0, 1].set_xlabel('Quality Score')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Quality Score Distribution')
                axes[0, 1].grid(True, alpha=0.3)
            
            # SNPs per chromosome
            if 'chr' in snp_data.columns:
                chrom_counts = snp_data['chr'].value_counts().sort_index()
                axes[1, 0].bar(range(len(chrom_counts)), chrom_counts.values, color='lightgreen')
                axes[1, 0].set_xlabel('Chromosome')
                axes[1, 0].set_ylabel('Number of SNPs')
                axes[1, 0].set_title('SNPs per Chromosome')
                axes[1, 0].set_xticks(range(len(chrom_counts)))
                axes[1, 0].set_xticklabels(chrom_counts.index, rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            
            # Missing data plot
            genotype_cols = [col for col in snp_data.columns if col.startswith('genotype_')]
            if genotype_cols:
                missing_rates = snp_data[genotype_cols].isnull().sum(axis=1) / len(genotype_cols)
                axes[1, 1].hist(missing_rates, bins=30, alpha=0.7, color='orange')
                axes[1, 1].set_xlabel('Missing Data Rate')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Missing Data Distribution')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filename = 'snp_overview'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create SNP overview plot: {e}")
            return None
    
    def _plot_methylation_overview(self, methylation_data: pd.DataFrame) -> Optional[str]:
        """Create methylation data overview plot."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Beta value distribution
            all_betas = methylation_data.values.flatten()
            all_betas = all_betas[~np.isnan(all_betas)]
            
            axes[0, 0].hist(all_betas, bins=50, alpha=0.7, color='purple')
            axes[0, 0].set_xlabel('Beta Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Methylation Beta Value Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Mean methylation per sample
            sample_means = methylation_data.mean(axis=0)
            axes[0, 1].hist(sample_means, bins=30, alpha=0.7, color='teal')
            axes[0, 1].set_xlabel('Mean Methylation Level')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Mean Methylation per Sample')
            axes[0, 1].grid(True, alpha=0.3)
            
            # CpG site variability
            cpg_stds = methylation_data.std(axis=1)
            axes[1, 0].hist(cpg_stds, bins=30, alpha=0.7, color='gold')
            axes[1, 0].set_xlabel('Standard Deviation')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('CpG Site Variability')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Missing data heatmap (sample)
            missing_by_sample = methylation_data.isnull().sum(axis=0)
            if len(missing_by_sample) > 0:
                axes[1, 1].bar(range(len(missing_by_sample)), missing_by_sample.values, 
                              color='red', alpha=0.7)
                axes[1, 1].set_xlabel('Sample Index')
                axes[1, 1].set_ylabel('Missing CpG Sites')
                axes[1, 1].set_title('Missing Data per Sample')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filename = 'methylation_overview'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create methylation overview plot: {e}")
            return None
    
    def _plot_data_quality(self, snp_data: pd.DataFrame, 
                          methylation_data: pd.DataFrame) -> Optional[str]:
        """Create data quality assessment plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Sample overlap
            genotype_cols = [col for col in snp_data.columns if col.startswith('genotype_')]
            snp_samples = [col.replace('genotype_', '') for col in genotype_cols]
            meth_samples = methylation_data.columns.tolist()
            
            common_samples = set(snp_samples) & set(meth_samples)
            snp_only = set(snp_samples) - common_samples
            meth_only = set(meth_samples) - common_samples
            
            # Venn diagram representation as bar plot
            categories = ['Common', 'SNP Only', 'Methylation Only']
            counts = [len(common_samples), len(snp_only), len(meth_only)]
            colors = ['green', 'blue', 'red']
            
            axes[0, 0].bar(categories, counts, color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('Number of Samples')
            axes[0, 0].set_title('Sample Overlap between Datasets')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Data completeness
            if genotype_cols:
                snp_completeness = 1 - (snp_data[genotype_cols].isnull().sum().sum() / 
                                      (len(snp_data) * len(genotype_cols)))
            else:
                snp_completeness = 0
            
            meth_completeness = 1 - (methylation_data.isnull().sum().sum() / 
                                   (methylation_data.shape[0] * methylation_data.shape[1]))
            
            completeness_data = ['SNP Data', 'Methylation Data']
            completeness_values = [snp_completeness, meth_completeness]
            
            bars = axes[0, 1].bar(completeness_data, completeness_values, 
                                 color=['lightblue', 'lightcoral'], alpha=0.7)
            axes[0, 1].set_ylabel('Completeness Rate')
            axes[0, 1].set_title('Data Completeness')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, completeness_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # Data dimensions
            dimensions = ['SNPs', 'CpG Sites', 'Samples (SNP)', 'Samples (Meth)']
            dim_values = [len(snp_data), len(methylation_data), 
                         len(snp_samples), len(meth_samples)]
            
            axes[1, 0].bar(dimensions, dim_values, color='gold', alpha=0.7)
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Dataset Dimensions')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Quality scores summary (if available)
            if 'qual' in snp_data.columns:
                qual_stats = snp_data['qual'].describe()
                
                # Simple quality summary
                axes[1, 1].text(0.1, 0.8, f"Quality Score Statistics:", 
                               transform=axes[1, 1].transAxes, fontweight='bold')
                axes[1, 1].text(0.1, 0.7, f"Mean: {qual_stats['mean']:.2f}", 
                               transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.6, f"Median: {qual_stats['50%']:.2f}", 
                               transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.5, f"Min: {qual_stats['min']:.2f}", 
                               transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.4, f"Max: {qual_stats['max']:.2f}", 
                               transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('SNP Quality Summary')
                axes[1, 1].set_xticks([])
                axes[1, 1].set_yticks([])
            else:
                axes[1, 1].text(0.5, 0.5, 'No quality data available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Quality Assessment')
                axes[1, 1].set_xticks([])
                axes[1, 1].set_yticks([])
            
            plt.tight_layout()
            
            # Save plot
            filename = 'data_quality'
            path = self._save_plot(fig, filename)
            plt.close(fig)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create data quality plot: {e}")
            return None
    
    def _create_summary_dashboard(self, analysis_results: Dict) -> Optional[str]:
        """Create an interactive summary dashboard using plotly."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Analysis Summary', 
                    'Top Correlations',
                    'Method Comparison', 
                    'Genomic Overview'
                ),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Analysis summary
            if 'summary' in analysis_results:
                summary = analysis_results['summary']
                
                # Method counts
                methods = ['Statistical', 'Machine Learning', 'Window-based']
                method_counts = [1, 1, 1]  # Placeholder - would be actual counts
                
                fig.add_trace(
                    go.Bar(x=methods, y=method_counts, name="Methods Used"),
                    row=1, col=1
                )
            
            # Top correlations scatter
            if ('statistical' in analysis_results and 
                'pairwise_correlations' in analysis_results['statistical']):
                
                pairwise = analysis_results['statistical']['pairwise_correlations']
                if isinstance(pairwise, pd.DataFrame) and len(pairwise) > 0:
                    top_corr = pairwise.head(20)
                    
                    corr_cols = [col for col in top_corr.columns if col.endswith('_corr')]
                    if corr_cols:
                        fig.add_trace(
                            go.Scatter(
                                x=range(len(top_corr)),
                                y=top_corr[corr_cols[0]],
                                mode='markers',
                                name="Top Correlations",
                                marker=dict(size=8)
                            ),
                            row=1, col=2
                        )
            
            # Update layout
            fig.update_layout(
                title="Genomics Correlation Analysis Dashboard",
                height=800,
                showlegend=True
            )
            
            # Save dashboard
            filename = 'summary_dashboard'
            path = self._save_plotly_figure(fig, filename)
            
            return path
            
        except Exception as e:
            logger.warning(f"Failed to create summary dashboard: {e}")
            return None
    
    def _save_plot(self, fig, filename: str) -> str:
        """Save matplotlib figure in multiple formats."""
        saved_paths = []
        
        for fmt in self.output_formats:
            if fmt in ['png', 'pdf', 'svg', 'jpg']:
                filepath = self.results_dir / f"{filename}.{fmt}"
                fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', format=fmt)
                saved_paths.append(str(filepath))
        
        return saved_paths[0] if saved_paths else None
    
    def _save_plotly_figure(self, fig, filename: str) -> str:
        """Save plotly figure in multiple formats."""
        saved_paths = []
        
        for fmt in self.output_formats:
            if fmt == 'html':
                filepath = self.results_dir / f"{filename}.html"
                fig.write_html(str(filepath))
                saved_paths.append(str(filepath))
            elif fmt == 'png':
                filepath = self.results_dir / f"{filename}.png"
                fig.write_image(str(filepath), width=1200, height=800)
                saved_paths.append(str(filepath))
            elif fmt == 'pdf':
                filepath = self.results_dir / f"{filename}.pdf"
                fig.write_image(str(filepath), width=1200, height=800)
                saved_paths.append(str(filepath))
        
        return saved_paths[0] if saved_paths else None