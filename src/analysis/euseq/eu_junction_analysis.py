#!/usr/bin/env python3
"""
EU and EU-junction Analysis Script
Focused analysis of nascent RNA processing defects with AQR24h treatment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import yaml

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data import BEDProcessor
from genomic_correlations import GenomicRegionAnalyzer

def run_eu_junction_analysis():
    """Run comprehensive EU and EU-junction analysis"""
    
    print("="*60)
    print("EU & EU-JUNCTION ANALYSIS WITH AQR24h")
    print("="*60)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processors
    bed_processor = BEDProcessor(config)
    analyzer = GenomicRegionAnalyzer(config)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    print("\n1. Loading EU and EU-junction data...")
    
    # Load treatment comparison data
    treatment_data = bed_processor.get_treatment_comparison_data()
    
    print(f"Available assays: {list(treatment_data.keys())}")
    
    # Focus on EU and EUjunc data
    eu_datasets = {}
    
    # Extract EU data (try different possible names)
    eu_names = ['EUcombined', 'EU', 'EUcombined_AQR24h_DMSO24h']
    for name in eu_names:
        if name in treatment_data:
            eu_datasets['EU_gain'] = pd.DataFrame(treatment_data[name].get('gain', []))
            eu_datasets['EU_loss'] = pd.DataFrame(treatment_data[name].get('loss', []))
            print(f"Found EU data under name: {name}")
            break
    
    # Extract EUjunc data
    eujunc_names = ['EUjunc', 'EUjunc_AQR24h_DMSO24h']
    for name in eujunc_names:
        if name in treatment_data:
            eu_datasets['EUjunc_gain'] = pd.DataFrame(treatment_data[name].get('gain', []))
            eu_datasets['EUjunc_loss'] = pd.DataFrame(treatment_data[name].get('loss', []))
            print(f"Found EUjunc data under name: {name}")
            break
    
    # Add END-seq for comparison
    endseq_names = ['ENDseq', 'ENDseq_AQR24h_DMSO24h']
    for name in endseq_names:
        if name in treatment_data:
            eu_datasets['END_gain'] = pd.DataFrame(treatment_data[name].get('gain', []))
            print(f"Found END-seq data under name: {name}")
            break
    
    # Print dataset sizes
    print("\nDataset sizes:")
    for name, df in eu_datasets.items():
        print(f"  {name}: {len(df):,} regions")
    
    print("\n2. Analyzing EU vs EU-junction correlations...")
    
    # Calculate overlaps between EU datasets
    eu_correlations = {}
    
    # EU gain vs EU loss (within-assay comparison)
    if not eu_datasets['EU_gain'].empty and not eu_datasets['EU_loss'].empty:
        overlap_stats = analyzer.calculate_region_overlaps(
            eu_datasets['EU_gain'], eu_datasets['EU_loss'], 
            'EU_gain', 'EU_loss'
        )
        eu_correlations['EU_gain_vs_EU_loss'] = overlap_stats
    
    # EUjunc gain vs EUjunc loss  
    if not eu_datasets['EUjunc_gain'].empty and not eu_datasets['EUjunc_loss'].empty:
        overlap_stats = analyzer.calculate_region_overlaps(
            eu_datasets['EUjunc_gain'], eu_datasets['EUjunc_loss'],
            'EUjunc_gain', 'EUjunc_loss'
        )
        eu_correlations['EUjunc_gain_vs_EUjunc_loss'] = overlap_stats
    
    # EU vs EUjunc (cross-assay comparisons)
    cross_comparisons = [
        ('EU_gain', 'EUjunc_gain'),
        ('EU_loss', 'EUjunc_loss'),
        ('EU_gain', 'EUjunc_loss'),  # Anti-correlation
        ('EU_loss', 'EUjunc_gain')   # Anti-correlation
    ]
    
    for comp1, comp2 in cross_comparisons:
        if not eu_datasets[comp1].empty and not eu_datasets[comp2].empty:
            overlap_stats = analyzer.calculate_region_overlaps(
                eu_datasets[comp1], eu_datasets[comp2],
                comp1, comp2
            )
            eu_correlations[f'{comp1}_vs_{comp2}'] = overlap_stats
    
    print("\n3. Analyzing genomic feature distributions...")
    
    # Analyze genomic features for each EU dataset
    def analyze_features(df, dataset_name):
        """Simple genomic feature analysis"""
        if df.empty or 'annotation' not in df.columns:
            return pd.DataFrame()
        
        # Count features
        feature_counts = df['annotation'].value_counts()
        total = len(df)
        
        # Create feature dataframe
        features_df = pd.DataFrame({
            'feature_type': feature_counts.index,
            'count': feature_counts.values,
            'percentage': (feature_counts.values / total) * 100
        })
        
        return features_df.reset_index(drop=True)
    
    feature_analysis = {}
    for name, df in eu_datasets.items():
        if not df.empty:
            features = analyze_features(df, name)
            feature_analysis[name] = features
    
    print("\n4. Creating comprehensive visualizations...")
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(20, 16))
    
    # Panel 1: Dataset sizes comparison
    ax1 = plt.subplot(3, 4, 1)
    sizes = [len(df) for df in eu_datasets.values()]
    names = list(eu_datasets.keys())
    colors = ['red' if 'gain' in name else 'blue' for name in names]
    
    bars = ax1.bar(range(len(names)), sizes, color=colors, alpha=0.7)
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Number of Regions')
    ax1.set_title('EU/EUjunc Dataset Sizes')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    
    # Add value labels
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                f'{size:,}', ha='center', va='bottom', fontsize=8)
    
    # Panel 2: Cross-correlation heatmap
    ax2 = plt.subplot(3, 4, 2)
    
    # Create correlation matrix
    corr_matrix = np.zeros((len(eu_datasets), len(eu_datasets)))
    dataset_names = list(eu_datasets.keys())
    
    for i, name1 in enumerate(dataset_names):
        for j, name2 in enumerate(dataset_names):
            if i == j:
                corr_matrix[i, j] = 1.0  # Self-correlation
            else:
                key = f'{name1}_vs_{name2}'
                if key in eu_correlations:
                    # Use overlap fraction as correlation measure
                    overlap_frac = eu_correlations[key].get('overlap_fraction_A', 0)
                    corr_matrix[i, j] = overlap_frac
    
    im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    ax2.set_xticks(range(len(dataset_names)))
    ax2.set_yticks(range(len(dataset_names)))
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.set_yticklabels(dataset_names)
    ax2.set_title('EU/EUjunc Cross-Correlations')
    
    # Add correlation values
    for i in range(len(dataset_names)):
        for j in range(len(dataset_names)):
            text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if corr_matrix[i, j] < 0.5 else "white")
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Panel 3: Feature distribution for EU_loss
    ax3 = plt.subplot(3, 4, 3)
    if 'EU_loss' in feature_analysis:
        features_df = feature_analysis['EU_loss']
        if not features_df.empty:
            ax3.pie(features_df['percentage'], labels=features_df['feature_type'], 
                   autopct='%1.1f%%', startangle=90)
            ax3.set_title('EU Loss: Genomic Features')
    
    # Panel 4: Feature distribution for EUjunc_loss  
    ax4 = plt.subplot(3, 4, 4)
    if 'EUjunc_loss' in feature_analysis:
        features_df = feature_analysis['EUjunc_loss']
        if not features_df.empty:
            ax4.pie(features_df['percentage'], labels=features_df['feature_type'],
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title('EUjunc Loss: Genomic Features')
    
    # Panel 5: Overlap analysis - EU gain vs loss
    ax5 = plt.subplot(3, 4, 5)
    if 'EU_gain_vs_EU_loss' in eu_correlations:
        stats = eu_correlations['EU_gain_vs_EU_loss']
        
        # Create Venn diagram-style visualization
        total_gain = len(eu_datasets['EU_gain'])
        total_loss = len(eu_datasets['EU_loss'])
        overlap_count = stats.get('overlap_count', 0)
        
        # Bar chart showing overlap
        categories = ['EU Gain\nOnly', 'Overlap', 'EU Loss\nOnly']
        values = [total_gain - overlap_count, overlap_count, total_loss - overlap_count]
        colors_venn = ['red', 'purple', 'blue']
        
        bars = ax5.bar(categories, values, color=colors_venn, alpha=0.7)
        ax5.set_ylabel('Number of Regions')
        ax5.set_title('EU Gain vs Loss Overlap')
        
        # Add percentage labels
        for bar, val in zip(bars, values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # Panel 6: Overlap analysis - EUjunc gain vs loss
    ax6 = plt.subplot(3, 4, 6)
    if 'EUjunc_gain_vs_EUjunc_loss' in eu_correlations:
        stats = eu_correlations['EUjunc_gain_vs_EUjunc_loss']
        
        total_gain = len(eu_datasets['EUjunc_gain'])
        total_loss = len(eu_datasets['EUjunc_loss'])
        overlap_count = stats.get('overlap_count', 0)
        
        categories = ['EUjunc Gain\nOnly', 'Overlap', 'EUjunc Loss\nOnly']
        values = [total_gain - overlap_count, overlap_count, total_loss - overlap_count]
        colors_venn = ['red', 'purple', 'blue']
        
        bars = ax6.bar(categories, values, color=colors_venn, alpha=0.7)
        ax6.set_ylabel('Number of Regions')
        ax6.set_title('EUjunc Gain vs Loss Overlap')
        
        for bar, val in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # Panel 7: EU vs EUjunc comparison (same direction)
    ax7 = plt.subplot(3, 4, 7)
    
    # Compare gain correlations
    gain_correlations = []
    loss_correlations = []
    
    if 'EU_gain_vs_EUjunc_gain' in eu_correlations:
        gain_correlations.append(eu_correlations['EU_gain_vs_EUjunc_gain'].get('overlap_fraction_A', 0))
    if 'EU_loss_vs_EUjunc_loss' in eu_correlations:
        loss_correlations.append(eu_correlations['EU_loss_vs_EUjunc_loss'].get('overlap_fraction_A', 0))
    
    categories = ['Gain\nCorrelation', 'Loss\nCorrelation']
    values = [gain_correlations[0] if gain_correlations else 0, 
             loss_correlations[0] if loss_correlations else 0]
    
    bars = ax7.bar(categories, values, color=['red', 'blue'], alpha=0.7)
    ax7.set_ylabel('Overlap Fraction')
    ax7.set_title('EU vs EUjunc Same-Direction\nCorrelations')
    ax7.set_ylim(0, 1)
    
    for bar, val in zip(bars, values):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Panel 8: Anti-correlations
    ax8 = plt.subplot(3, 4, 8)
    
    anti_correlations = []
    anti_labels = []
    
    if 'EU_gain_vs_EUjunc_loss' in eu_correlations:
        anti_correlations.append(eu_correlations['EU_gain_vs_EUjunc_loss'].get('overlap_fraction_A', 0))
        anti_labels.append('EU Gain vs\nEUjunc Loss')
    
    if 'EU_loss_vs_EUjunc_gain' in eu_correlations:
        anti_correlations.append(eu_correlations['EU_loss_vs_EUjunc_gain'].get('overlap_fraction_A', 0))
        anti_labels.append('EU Loss vs\nEUjunc Gain')
    
    if anti_correlations:
        bars = ax8.bar(range(len(anti_correlations)), anti_correlations, 
                      color=['orange', 'green'], alpha=0.7)
        ax8.set_xticks(range(len(anti_labels)))
        ax8.set_xticklabels(anti_labels)
        ax8.set_ylabel('Overlap Fraction')
        ax8.set_title('EU vs EUjunc\nAnti-Correlations')
        ax8.set_ylim(0, max(anti_correlations) * 1.2 if anti_correlations else 1)
        
        for bar, val in zip(bars, anti_correlations):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(anti_correlations)*0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Panel 9-12: Detailed feature comparisons
    feature_names = ['GENEBODY', 'PROMOTER', 'INTERGENIC', 'UTR5', 'UTR3']
    
    # Panel 9: Feature comparison bar chart
    ax9 = plt.subplot(3, 4, (9, 10))  # Span two columns
    
    feature_comparison_data = {}
    for dataset_name in ['EU_gain', 'EU_loss', 'EUjunc_gain', 'EUjunc_loss']:
        if dataset_name in feature_analysis:
            features_df = feature_analysis[dataset_name]
            feature_dict = dict(zip(features_df['feature_type'], features_df['percentage']))
            feature_comparison_data[dataset_name] = [feature_dict.get(f, 0) for f in feature_names]
    
    if feature_comparison_data:
        x = np.arange(len(feature_names))
        width = 0.2
        
        for i, (dataset, percentages) in enumerate(feature_comparison_data.items()):
            color = 'red' if 'gain' in dataset else 'blue'
            alpha = 0.8 if 'EU_' in dataset and 'junc' not in dataset else 0.5
            ax9.bar(x + i*width, percentages, width, label=dataset, color=color, alpha=alpha)
        
        ax9.set_xlabel('Genomic Features')
        ax9.set_ylabel('Percentage of Regions')
        ax9.set_title('Genomic Feature Distribution Comparison')
        ax9.set_xticks(x + width * 1.5)
        ax9.set_xticklabels(feature_names, rotation=45, ha='right')
        ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Panel 11: Summary statistics
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    # Create summary text
    summary_text = "Key EU/EUjunc Findings:\n\n"
    
    # Dataset sizes
    summary_text += "Dataset Sizes:\n"
    for name, df in eu_datasets.items():
        if name != 'END_gain':  # Skip END for EU analysis
            summary_text += f"• {name}: {len(df):,}\n"
    
    summary_text += "\nCorrelation Insights:\n"
    
    # Same-direction correlations
    if 'EU_gain_vs_EUjunc_gain' in eu_correlations:
        overlap_frac = eu_correlations['EU_gain_vs_EUjunc_gain'].get('overlap_fraction_A', 0)
        summary_text += f"• EU-EUjunc Gain Correlation: {overlap_frac:.1%}\n"
    
    if 'EU_loss_vs_EUjunc_loss' in eu_correlations:
        overlap_frac = eu_correlations['EU_loss_vs_EUjunc_loss'].get('overlap_fraction_A', 0)
        summary_text += f"• EU-EUjunc Loss Correlation: {overlap_frac:.1%}\n"
    
    # Feature insights
    summary_text += "\nGenomic Distribution:\n"
    if 'EU_loss' in feature_analysis:
        features_df = feature_analysis['EU_loss']
        if not features_df.empty:
            top_feature = features_df.iloc[0]
            summary_text += f"• EU Loss: {top_feature['percentage']:.1f}% in {top_feature['feature_type']}\n"
    
    if 'EUjunc_loss' in feature_analysis:
        features_df = feature_analysis['EUjunc_loss']
        if not features_df.empty:
            top_feature = features_df.iloc[0]
            summary_text += f"• EUjunc Loss: {top_feature['percentage']:.1f}% in {top_feature['feature_type']}\n"
    
    ax11.text(0.1, 0.9, summary_text, transform=ax11.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    # Panel 12: Mechanistic interpretation
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    interpretation = "Mechanistic Interpretation:\n\n"
    interpretation += "EU (nascent RNA) vs EUjunc (splicing):\n\n"
    
    # Calculate correlation strength
    if 'EU_loss_vs_EUjunc_loss' in eu_correlations:
        loss_corr = eu_correlations['EU_loss_vs_EUjunc_loss'].get('overlap_fraction_A', 0)
        if loss_corr > 0.1:
            interpretation += f"Strong correlation ({loss_corr:.1%}):\n"
            interpretation += "• AQR24h disrupts both nascent\n  RNA synthesis AND splicing\n"
            interpretation += "• Suggests coordinated disruption\n  of RNA processing machinery\n\n"
        else:
            interpretation += f"Weak correlation ({loss_corr:.1%}):\n"
            interpretation += "• EU and EUjunc represent\n  independent processes\n"
            interpretation += "• AQR24h may have distinct\n  effects on synthesis vs splicing\n\n"
    
    interpretation += "Therapeutic Implications:\n"
    interpretation += "• Target both RNA synthesis\n  and splicing machinery\n"
    interpretation += "• Focus on gene body regions\n  with dual defects\n"
    
    ax12.text(0.05, 0.95, interpretation, transform=ax12.transAxes, fontsize=9,
             verticalalignment='top', color='darkblue')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'eu_eujunc_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n5. Saving detailed results...")
    
    # Save correlation results
    eu_corr_df = pd.DataFrame()
    for comparison, stats in eu_correlations.items():
        row_data = {
            'comparison': comparison,
            'overlap_count': stats.get('overlap_count', 0),
            'overlap_fraction_A': stats.get('overlap_fraction_A', 0),
            'overlap_fraction_B': stats.get('overlap_fraction_B', 0),
            'fold_enrichment': stats.get('fold_enrichment', 0),
            'p_value': stats.get('p_value', np.nan)
        }
        eu_corr_df = pd.concat([eu_corr_df, pd.DataFrame([row_data])], ignore_index=True)
    
    eu_corr_df.to_csv(results_dir / 'eu_eujunc_correlations.csv', index=False)
    
    # Save feature analysis
    feature_summary = pd.DataFrame()
    for dataset, features_df in feature_analysis.items():
        if not features_df.empty:
            features_df_copy = features_df.copy()
            features_df_copy['dataset'] = dataset
            feature_summary = pd.concat([feature_summary, features_df_copy], ignore_index=True)
    
    feature_summary.to_csv(results_dir / 'eu_eujunc_features.csv', index=False)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to:")
    print(f"  - {results_dir / 'eu_eujunc_comprehensive_analysis.png'}")
    print(f"  - {results_dir / 'eu_eujunc_correlations.csv'}")
    print(f"  - {results_dir / 'eu_eujunc_features.csv'}")
    
    print("\n" + "="*60)
    print("EU/EUJUNC ANALYSIS SUMMARY")
    print("="*60)
    
    # Print key findings
    print("\nKEY FINDINGS:")
    
    # Dataset overview
    print(f"\n1. Dataset Overview:")
    for name, df in eu_datasets.items():
        if name != 'END_gain':
            print(f"   • {name}: {len(df):,} regions")
    
    # Correlation findings
    print(f"\n2. EU vs EUjunc Correlations:")
    if 'EU_gain_vs_EUjunc_gain' in eu_correlations:
        gain_corr = eu_correlations['EU_gain_vs_EUjunc_gain']['overlap_fraction_A']
        print(f"   • Gain-Gain correlation: {gain_corr:.1%}")
    
    if 'EU_loss_vs_EUjunc_loss' in eu_correlations:
        loss_corr = eu_correlations['EU_loss_vs_EUjunc_loss']['overlap_fraction_A']
        print(f"   • Loss-Loss correlation: {loss_corr:.1%}")
        
        if loss_corr > 0.2:
            print("   → Strong correlation suggests coordinated RNA processing defects")
        elif loss_corr > 0.1:
            print("   → Moderate correlation suggests related but distinct processes")
        else:
            print("   → Weak correlation suggests independent mechanisms")
    
    # Feature distribution insights
    print(f"\n3. Genomic Feature Insights:")
    if 'EU_loss' in feature_analysis and not feature_analysis['EU_loss'].empty:
        top_feature = feature_analysis['EU_loss'].iloc[0]
        print(f"   • EU losses primarily in: {top_feature['feature_type']} ({top_feature['percentage']:.1f}%)")
    
    if 'EUjunc_loss' in feature_analysis and not feature_analysis['EUjunc_loss'].empty:
        top_feature = feature_analysis['EUjunc_loss'].iloc[0]
        print(f"   • EUjunc losses primarily in: {top_feature['feature_type']} ({top_feature['percentage']:.1f}%)")
    
    return eu_correlations, feature_analysis

if __name__ == "__main__":
    correlations, features = run_eu_junction_analysis()