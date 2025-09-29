#!/usr/bin/env python3
"""
END-seq Correlation Analysis Script

This script analyzes correlations between END-seq breaks and other genomic assays
to understand what causes transcription termination defects with AQR24h treatment.

Usage:
    python endseq_analysis.py [--config config.yaml] [--output-dir results/]

Example:
    python endseq_analysis.py --config config.yaml
"""

import argparse
import sys
import logging
from pathlib import Path
import yaml
import pandas as pd
from typing import Dict, Any

sys.path.append('src/utils')
from mytochai_debug import debug

# Add src to path
sys.path.append('src')

from data import DataLoader
from genomic_correlations import GenomicRegionAnalyzer
from multiomics_visualization import MultiOmicsPlotter


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ENDseqAnalysisPipeline:
    """
    Complete analysis pipeline for END-seq correlation analysis.
    """
    
    def __init__(self, config_path: str, output_dir: str = None):
        """Initialize the analysis pipeline."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set output directory
        if output_dir:
            self.config['output']['results_dir'] = output_dir
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.region_analyzer = GenomicRegionAnalyzer(self.config)
        self.plotter = MultiOmicsPlotter(self.config, Path(self.config['output']['results_dir']))
        
        # Ensure output directory exists
        Path(self.config['output']['results_dir']).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config file {self.config_path}: {e}")
            raise
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete END-seq correlation analysis pipeline.
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting END-seq correlation analysis pipeline")
        
        results = {
            'config': self.config,
            'data_summary': {},
            'genomic_analysis': {},
            'endseq_focus': {},
            'visualizations': {}
        }
        
        try:
            # Step 1: Load BED file data
            logger.info("Step 1: Loading BED file data")
            bed_data = self.data_loader.load_bed_data()
            treatment_data = self.data_loader.load_treatment_comparison_data()
            
            results['data_summary'] = self._summarize_data(bed_data, treatment_data)
            
            # Step 2: Analyze treatment correlations
            logger.info("Step 2: Analyzing treatment effect correlations")
            genomic_analysis = self.region_analyzer.analyze_treatment_correlations(treatment_data)
            results['genomic_analysis'] = genomic_analysis
            
            # Step 3: Focused END-seq analysis
            logger.info("Step 3: Performing focused END-seq causality analysis")
            endseq_analysis = self.region_analyzer.analyze_endseq_focus(treatment_data)
            results['endseq_focus'] = endseq_analysis
            
            # Combine results for visualization
            combined_results = {**genomic_analysis, 'endseq_analysis': endseq_analysis}
            
            # Step 4: Generate visualizations
            logger.info("Step 4: Creating comprehensive visualization report")
            plot_paths = self.plotter.create_endseq_correlation_report(combined_results)
            results['visualizations'] = plot_paths
            
            # Step 5: Save results
            logger.info("Step 5: Saving analysis results")
            self._save_results(results)
            
            # Step 6: Generate summary report
            logger.info("Step 6: Generating summary report")
            self._generate_summary_report(results)
            
            logger.info("END-seq correlation analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            raise
        
        return results
    
    def _summarize_data(self, bed_data: Dict, treatment_data: Dict) -> Dict:
        """Create summary statistics for loaded data."""
        summary = {
            'total_assays': len(bed_data),
            'assays_found': list(bed_data.keys()),
            'treatment_assays': len(treatment_data),
            'assay_details': {}
        }
        
        for assay_name, conditions in bed_data.items():
            assay_summary = {
                'conditions': list(conditions.keys()),
                'total_regions': sum(len(df) for df in conditions.values())
            }
            
            # Add region counts by condition
            for condition, df in conditions.items():
                assay_summary[f'{condition}_regions'] = len(df)
            
            summary['assay_details'][assay_name] = assay_summary
        
        return summary
    
    def _save_results(self, results: Dict) -> None:
        """Save analysis results to files."""
        results_dir = Path(self.config['output']['results_dir'])
        
        # Save main results as YAML
        results_file = results_dir / 'endseq_analysis_results.yaml'
        with open(results_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = self._make_serializable(results)
            yaml.dump(serializable_results, f, default_flow_style=False)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save detailed correlation data as CSV if available
        genomic_analysis = results.get('genomic_analysis', {})
        pairwise_corr = genomic_analysis.get('pairwise_correlations', {})
        
        if pairwise_corr:
            corr_summary = []
            for pair_name, corr_data in pairwise_corr.items():
                cross_corr = corr_data.get('cross_correlation', {})
                corr_summary.append({
                    'assay_pair': pair_name,
                    'correlation_strength': cross_corr.get('correlation_strength', 0),
                    'correlation_direction': cross_corr.get('correlation_direction', 'unknown'),
                    'p_value': cross_corr.get('min_p_value', 1.0),
                    'is_significant': cross_corr.get('is_significant', False)
                })
            
            if corr_summary:
                corr_df = pd.DataFrame(corr_summary)
                corr_file = results_dir / 'correlation_summary.csv'
                corr_df.to_csv(corr_file, index=False)
                logger.info(f"Correlation summary saved to {corr_file}")
    
    def _make_serializable(self, obj):
        """Convert object to YAML-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, pd.DataFrame):
            return f"DataFrame with {len(obj)} rows and {len(obj.columns)} columns"
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _generate_summary_report(self, results: Dict) -> None:
        """Generate a human-readable summary report."""
        results_dir = Path(self.config['output']['results_dir'])
        report_file = results_dir / 'endseq_analysis_summary.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\\n")
            f.write("END-seq Correlation Analysis Summary Report\\n")
            f.write("=" * 80 + "\\n\\n")
            
            # Data summary
            data_summary = results.get('data_summary', {})
            f.write(f"Data Summary:\\n")
            f.write(f"- Total assays analyzed: {data_summary.get('total_assays', 0)}\\n")
            f.write(f"- Assays found: {', '.join(data_summary.get('assays_found', []))}\\n")
            f.write(f"- Treatment comparison assays: {data_summary.get('treatment_assays', 0)}\\n\\n")
            
            # Assay details
            f.write("Assay Details:\\n")
            for assay, details in data_summary.get('assay_details', {}).items():
                f.write(f"  {assay}:\\n")
                f.write(f"    - Total regions: {details.get('total_regions', 0)}\\n")
                f.write(f"    - Conditions: {', '.join(details.get('conditions', []))}\\n")
            f.write("\\n")
            
            # Correlation analysis
            genomic_analysis = results.get('genomic_analysis', {})
            pairwise_corr = genomic_analysis.get('pairwise_correlations', {})
            
            if pairwise_corr:
                f.write("Pairwise Correlation Analysis:\\n")
                for pair_name, corr_data in pairwise_corr.items():
                    cross_corr = corr_data.get('cross_correlation', {})
                    f.write(f"  {pair_name}:\\n")
                    f.write(f"    - Correlation strength: {cross_corr.get('correlation_strength', 0):.3f}\\n")
                    f.write(f"    - Direction: {cross_corr.get('correlation_direction', 'unknown')}\\n")
                    f.write(f"    - Significant: {cross_corr.get('is_significant', False)}\\n")
                f.write("\\n")
            
            # END-seq specific findings
            endseq_analysis = results.get('endseq_focus', {})
            
            if endseq_analysis:
                f.write("END-seq Specific Analysis:\\n")
                
                # Mechanistic correlations
                mech_corr = endseq_analysis.get('mechanistic_correlations', {})
                if mech_corr:
                    f.write("  Mechanistic Correlations:\\n")
                    for assay, data in mech_corr.items():
                        hypothesis = data.get('mechanistic_hypothesis', 'No hypothesis')
                        f.write(f"    {assay}: {hypothesis}\\n")
                
                # Causality inference
                causality = endseq_analysis.get('causality_inference', {})
                if causality:
                    primary_causes = causality.get('primary_causes', [])
                    secondary_effects = causality.get('secondary_effects', [])
                    
                    f.write("  Inferred Causal Relationships:\\n")
                    if primary_causes:
                        f.write(f"    Primary causes: {', '.join(primary_causes)}\\n")
                    if secondary_effects:
                        f.write(f"    Secondary effects: {', '.join(secondary_effects)}\\n")
            
            # Visualization files
            viz_files = results.get('visualizations', {})
            if viz_files:
                f.write("\\nGenerated Visualizations:\\n")
                for plot_type, filepath in viz_files.items():
                    if filepath:
                        f.write(f"  - {plot_type}: {Path(filepath).name}\\n")
        
        logger.info(f"Summary report saved to {report_file}")


def main():
    """Main entry point for END-seq analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze correlations between END-seq breaks and other genomic assays"
    )
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for results (overrides config)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    debug()
 
    try:
        # Initialize and run analysis
        pipeline = ENDseqAnalysisPipeline(args.config, args.output_dir)
        results = pipeline.run_complete_analysis()
        
        Path.mkdir(Path(pipeline.config['output']['results_dir']), exist_ok=True)

        sys.exit(0)

        print("\\n" + "="*50)
        print("END-seq Correlation Analysis Completed!")
        print("="*50)
        print(f"Results saved to: {pipeline.config['output']['results_dir']}")
        
        # Print key findings
        data_summary = results.get('data_summary', {})
        print(f"\\nAnalyzed {data_summary.get('total_assays', 0)} assay types:")
        for assay in data_summary.get('assays_found', []):
            print(f"  - {assay}")
        
        print("\\nCheck the results directory for:")
        print("  - endseq_analysis_summary.txt (human-readable summary)")
        print("  - endseq_analysis_results.yaml (complete results)")
        print("  - correlation_summary.csv (correlation data)")
        print("  - Various visualization plots (.png files)")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()