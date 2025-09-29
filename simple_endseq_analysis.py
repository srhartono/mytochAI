#!/usr/bin/env python3
"""
Simple END-seq Correlation Analysis Runner

This is a streamlined version of the END-seq analysis that you can run immediately
to understand what causes END-seq breaks with AQR24h treatment.

Usage:
    python simple_endseq_analysis.py
"""

import sys
sys.path.append('src')

from genomic_correlations import GenomicRegionAnalyzer
from data import BEDProcessor
from multiomics_visualization import MultiOmicsPlotter
import yaml
from pathlib import Path
import pandas as pd

def run_simple_endseq_analysis():
    """Run a simplified END-seq correlation analysis."""
    
    print("="*60)
    print("END-seq Correlation Analysis - What Causes AQR24h Breaks?")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n1. Loading BED file data...")
    bed_processor = BEDProcessor(config)
    treatment_data = bed_processor.get_treatment_comparison_data()
    
    print(f"   Found {len(treatment_data)} assay types with treatment data:")
    for assay, conditions in treatment_data.items():
        gain_count = len(conditions.get('gain', []))
        loss_count = len(conditions.get('loss', []))
        print(f"     {assay}: {gain_count} gain, {loss_count} loss regions")
    
    # Initialize analyzer
    analyzer = GenomicRegionAnalyzer(config)
    
    # Focused END-seq analysis
    print("\n2. Analyzing END-seq patterns and correlations...")
    
    # Find END-seq data
    endseq_data = None
    endseq_key = None
    for key in treatment_data.keys():
        if 'END' in key.upper():
            endseq_data = treatment_data[key]
            endseq_key = key
            break
    
    if not endseq_data:
        print("   ERROR: No END-seq data found!")
        return
    
    print(f"   Found {endseq_key} data with {len(endseq_data.get('gain', []))} gain regions")
    
    # Analyze correlations with other assays
    print("\n3. Calculating overlaps with other assays...")
    
    correlation_results = {}
    endseq_gain = endseq_data.get('gain', pd.DataFrame())
    
    if len(endseq_gain) == 0:
        print("   ERROR: No END-seq gain data found!")
        return
    
    for assay_name, assay_data in treatment_data.items():
        if assay_name == endseq_key:
            continue
        
        assay_gain = assay_data.get('gain', pd.DataFrame())
        assay_loss = assay_data.get('loss', pd.DataFrame())
        
        if len(assay_gain) > 0:
            # Calculate overlap with gain regions
            overlap_gain = analyzer.calculate_region_overlaps(
                endseq_gain, assay_gain, f"{endseq_key}_gain", f"{assay_name}_gain"
            )
            
            correlation_results[f"{assay_name}_gain"] = {
                'overlap_count': overlap_gain['total_overlaps'],
                'endseq_fraction': overlap_gain['overlap_fraction1'],
                'other_fraction': overlap_gain['overlap_fraction2'],
                'fold_enrichment': overlap_gain['enrichment']['fold_enrichment'],
                'p_value': overlap_gain['enrichment']['p_value'],
                'significant': overlap_gain['enrichment']['is_significant'],
                'type': 'co-occurrence'
            }
        
        if len(assay_loss) > 0:
            # Calculate overlap with loss regions (anti-correlation)
            overlap_loss = analyzer.calculate_region_overlaps(
                endseq_gain, assay_loss, f"{endseq_key}_gain", f"{assay_name}_loss"
            )
            
            correlation_results[f"{assay_name}_loss"] = {
                'overlap_count': overlap_loss['total_overlaps'],
                'endseq_fraction': overlap_loss['overlap_fraction1'],
                'other_fraction': overlap_loss['overlap_fraction2'],
                'fold_enrichment': overlap_loss['enrichment']['fold_enrichment'],
                'p_value': overlap_loss['enrichment']['p_value'],
                'significant': overlap_loss['enrichment']['is_significant'],
                'type': 'anti-correlation'
            }
    
    print(f"   Calculated overlaps with {len(correlation_results)} assay conditions")
    
    # Analyze results
    print("\n4. Results Analysis:")
    print("   " + "-"*50)
    
    # Sort by significance and enrichment
    significant_results = {k: v for k, v in correlation_results.items() 
                         if v['significant'] and not pd.isna(v['fold_enrichment'])}
    
    if not significant_results:
        print("   No statistically significant correlations found.")
        print("   This could indicate:")
        print("     - END-seq breaks occur independently of other measured processes")
        print("     - Different genomic scales or timing of effects")
        print("     - Need for larger sample sizes or different analysis parameters")
    else:
        print(f"   Found {len(significant_results)} significant correlations:")
        
        # Sort by fold enrichment
        sorted_results = sorted(significant_results.items(), 
                              key=lambda x: x[1]['fold_enrichment'], reverse=True)
        
        for assay_condition, result in sorted_results[:5]:  # Top 5
            assay_name = assay_condition.split('_')[0]
            condition_type = assay_condition.split('_')[-1]
            
            print(f"     {assay_name} ({condition_type}):")
            print(f"       - {result['overlap_count']} overlapping regions")
            print(f"       - {result['endseq_fraction']:.1%} of END-seq regions overlap")
            print(f"       - {result['fold_enrichment']:.1f}x enrichment")
            print(f"       - p-value: {result['p_value']:.2e}")
            
            # Generate hypothesis
            if condition_type == 'gain' and result['fold_enrichment'] > 2:
                if 'sDRIP' in assay_name:
                    print(f"       → Hypothesis: R-loop formation contributes to END-seq breaks")
                elif 'EU' in assay_name:
                    print(f"       → Hypothesis: Nascent RNA processing defects cause termination failures")
                elif 'RNA' in assay_name:
                    print(f"       → Hypothesis: RNA accumulation correlates with termination problems")
            elif condition_type == 'loss' and result['fold_enrichment'] > 2:
                print(f"       → Hypothesis: Loss of {assay_name} activity associated with termination defects")
    
    # Feature analysis
    print("\n5. Feature Type Analysis:")
    print("   " + "-"*50)
    
    feature_counts = endseq_gain['feature_type'].value_counts()
    total_regions = len(endseq_gain)
    
    print("   END-seq gains are enriched in:")
    for feature, count in feature_counts.head().items():
        percentage = (count / total_regions) * 100
        print(f"     {feature}: {count} regions ({percentage:.1f}%)")
        
        # Feature-specific interpretation
        if feature == 'GENEBODY' and percentage > 40:
            print(f"       → AQR24h primarily causes gene body termination defects")
        elif feature == 'TERMINAL' and percentage > 30:
            print(f"       → Normal termination sites are disrupted by AQR24h")
        elif feature == 'PROMOTER' and percentage > 20:
            print(f"       → Promoter-proximal termination is affected")
    
    # Save summary
    print("\n6. Saving Results...")
    
    # Save correlation summary
    corr_df = pd.DataFrame.from_dict(correlation_results, orient='index')
    corr_df.to_csv(results_dir / 'endseq_correlation_summary.csv')
    
    # Save feature analysis
    feature_df = pd.DataFrame({
        'feature_type': feature_counts.index,
        'count': feature_counts.values,
        'percentage': (feature_counts.values / total_regions) * 100
    })
    feature_df.to_csv(results_dir / 'endseq_feature_analysis.csv', index=False)
    
    print(f"   Results saved to {results_dir}/")
    print("     - endseq_correlation_summary.csv")
    print("     - endseq_feature_analysis.csv")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What Causes END-seq Breaks with AQR24h?")
    print("="*60)
    
    if significant_results:
        top_result = sorted_results[0]
        assay_name = top_result[0].split('_')[0]
        print(f"Primary correlation: {assay_name}")
        print(f"Strength: {top_result[1]['fold_enrichment']:.1f}x enrichment")
        
        if 'sDRIP' in assay_name:
            print("Interpretation: R-loop formation appears to be a major contributor")
            print("to transcription termination defects caused by AQR24h treatment.")
        elif 'EU' in assay_name:
            print("Interpretation: Nascent RNA processing changes correlate with")
            print("transcription termination efficiency after AQR24h treatment.")
    else:
        print("No strong correlations found. END-seq breaks may be:")
        print("- Independent of measured genomic processes")
        print("- Occurring at different scales or timepoints")
        print("- Requiring additional assay types for mechanistic understanding")
    
    print(f"\nMost affected genomic features: {', '.join(feature_counts.head(3).index)}")
    print("Analysis complete!")

if __name__ == "__main__":
    run_simple_endseq_analysis()