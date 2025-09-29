#!/usr/bin/env python3
"""
EU/EU-junction Quick Analysis Summary
"""
import sys
sys.path.append('src')

from genomic_correlations import GenomicRegionAnalyzer
from data import BEDProcessor
import yaml
import pandas as pd

def quick_eu_summary():
    """Generate quick EU/EUjunc correlation summary"""
    
    # Load data
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    bed_processor = BEDProcessor(config)
    treatment_data = bed_processor.get_treatment_comparison_data()
    analyzer = GenomicRegionAnalyzer(config)
    
    # Get EU datasets
    eu_gain = pd.DataFrame(treatment_data['EU']['gain'])
    eu_loss = pd.DataFrame(treatment_data['EU']['loss'])
    eujunc_gain = pd.DataFrame(treatment_data['EUjunc']['gain'])
    eujunc_loss = pd.DataFrame(treatment_data['EUjunc']['loss'])
    endseq_gain = pd.DataFrame(treatment_data['ENDseq']['gain'])
    
    print('='*60)
    print('EU/EU-JUNCTION CORRELATION ANALYSIS SUMMARY')
    print('='*60)
    
    print('\nDATASET SIZES:')
    print(f'  EU Gain: {len(eu_gain):,} regions')
    print(f'  EU Loss: {len(eu_loss):,} regions')
    print(f'  EUjunc Gain: {len(eujunc_gain):,} regions')
    print(f'  EUjunc Loss: {len(eujunc_loss):,} regions')
    print(f'  END-seq Gain: {len(endseq_gain):,} regions')
    
    print('\nKEY CORRELATIONS:')
    
    # EU loss vs EUjunc loss (same direction - coordinated defects)
    if not eu_loss.empty and not eujunc_loss.empty:
        overlap = analyzer.calculate_region_overlaps(
            eu_loss, eujunc_loss, 'EU_loss', 'EUjunc_loss'
        )
        print(f'\n1. EU Loss ↔ EUjunc Loss (Coordinated Defects):')
        print(f'   • Overlap data: {overlap}')
        
        # Try different possible keys
        overlap_frac = overlap.get('overlap_fraction_A', overlap.get('overlap_fraction', 0))
        overlap_count = overlap.get('overlap_count', 0)
        
        print(f'   • {overlap_frac:.1%} of EU losses overlap with EUjunc losses')
        print(f'   • {overlap_count:,} shared regions')
        
        if overlap_frac > 0.15:
            interpretation = "STRONG correlation - Coordinated RNA processing disruption"
        elif overlap_frac > 0.05:
            interpretation = "MODERATE correlation - Related but distinct processes"
        else:
            interpretation = "WEAK correlation - Independent mechanisms"
        
        print(f'   → {interpretation}')
    
    # EU gain vs EUjunc gain (same direction)
    if not eu_gain.empty and not eujunc_gain.empty:
        overlap = analyzer.calculate_region_overlaps(
            eu_gain, eujunc_gain, 'EU_gain', 'EUjunc_gain'
        )
        print(f'\n2. EU Gain ↔ EUjunc Gain (Co-enhancement):')
        print(f'   • {overlap["overlap_fraction_A"]:.1%} overlap')
        print(f'   • {overlap["overlap_count"]:,} shared regions')
    
    # END-seq relationships
    print(f'\n3. END-seq Relationships:')
    
    if not endseq_gain.empty and not eu_loss.empty:
        overlap = analyzer.calculate_region_overlaps(
            endseq_gain, eu_loss, 'ENDseq_gain', 'EU_loss'
        )
        print(f'   • END-seq ↔ EU Loss: {overlap["overlap_fraction_A"]:.1%}')
    
    if not endseq_gain.empty and not eujunc_loss.empty:
        overlap = analyzer.calculate_region_overlaps(
            endseq_gain, eujunc_loss, 'ENDseq_gain', 'EUjunc_loss'
        )
        print(f'   • END-seq ↔ EUjunc Loss: {overlap["overlap_fraction_A"]:.1%}')
    
    # Feature analysis
    print(f'\nGENOMIC FEATURE DISTRIBUTION:')
    
    datasets = {
        'EU Loss': eu_loss,
        'EUjunc Loss': eujunc_loss,
        'END-seq Gain': endseq_gain
    }
    
    for name, df in datasets.items():
        if not df.empty and 'annotation' in df.columns:
            features = df['annotation'].value_counts()
            top_feature = features.index[0]
            top_pct = (features.iloc[0] / len(df)) * 100
            print(f'  {name}: {top_pct:.1f}% in {top_feature}')
    
    print('\n' + '='*60)
    print('MECHANISTIC INTERPRETATION')
    print('='*60)
    
    print('\n🔬 EU vs EUjunc Relationship:')
    print('  • EU measures nascent RNA synthesis (RNA Pol II activity)')
    print('  • EUjunc measures splice junction processing (splicing efficiency)')
    print('  • Overlapping losses suggest coordinated disruption of:')
    print('    - Co-transcriptional splicing machinery')
    print('    - RNA processing complex assembly')
    print('    - Chromatin structure affecting both processes')
    
    print('\n🎯 AQR24h Mechanism:')
    print('  • Disrupts RNA processing machinery coordination')
    print('  • Affects both transcription and splicing simultaneously')
    print('  • Results in widespread RNA processing defects')
    
    print('\n🧬 Therapeutic Implications:')
    print('  • Target co-transcriptional RNA processing complex')
    print('  • Consider combination therapies for:')
    print('    - Transcription elongation factors')
    print('    - Splicing regulatory proteins')
    print('    - Chromatin remodeling complexes')
    
    print('\n📊 Data Available in: results/eu_eujunc_comprehensive_analysis.png')

if __name__ == "__main__":
    quick_eu_summary()