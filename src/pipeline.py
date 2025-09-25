"""
Main pipeline for genomics correlation analysis.

"""

import yaml
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import argparse
import traceback

# Import framework modules
from .data import DataLoader, align_samples
from .classic_models import ClassicModels
from .ai_models import AIModels
from .analysis import CorrelationAnalyzer
from .visualization import GenomicsPlotter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genomics_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class GenomicsPipeline:
    """
    Main pipeline class that orchestrates the entire genomics correlation analysis.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the genomics analysis pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.classic_models = ClassicModels(self.config)
        self.ai_models = AIModels(self.config)
        self.correlation_analyzer = CorrelationAnalyzer(self.config)
        self.plotter = GenomicsPlotter(self.config)
        
        # Create results directory
        results_dir = Path(self.config['output']['results_dir'])
        results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized Genomics Pipeline with config: {config_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def run_full_pipeline(self, sample_labels: Optional[Dict] = None) -> Dict:
        """
        Run the complete genomics correlation analysis pipeline.
        
        Args:
            sample_labels: Optional dictionary mapping sample names to labels
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("=" * 80)
        logger.info("STARTING GENOMICS CORRELATION ANALYSIS PIPELINE")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load and preprocess data
            logger.info("Step 1: Loading and preprocessing data...")
            snp_data, methylation_data, annotation_data = self._load_data()
            
            # Step 2: Align samples between datasets
            logger.info("Step 2: Aligning samples between datasets...")
            snp_data, methylation_data = self._align_data(snp_data, methylation_data)
            
            # Step 3: Run classical ML models
            logger.info("Step 3: Running classical machine learning models...")
            classic_results = self._run_classic_models(snp_data, methylation_data, sample_labels)
            
            # Step 4: Run AI models
            logger.info("Step 4: Running AI models...")
            ai_results = self._run_ai_models(snp_data, methylation_data)
            
            # Step 5: Correlation analysis
            logger.info("Step 5: Running correlation analysis...")
            correlation_results = self._run_correlation_analysis(snp_data, methylation_data)
            
            # Step 6: Generate visualizations
            logger.info("Step 6: Generating visualizations...")
            visualization_results = self._generate_visualizations(
                correlation_results, snp_data, methylation_data
            )
            
            # Step 7: Compile final results
            logger.info("Step 7: Compiling final results...")
            final_results = self._compile_results(
                snp_data, methylation_data, annotation_data,
                classic_results, ai_results, correlation_results, visualization_results
            )
            
            # Step 8: Save results
            logger.info("Step 8: Saving results...")
            self._save_results(final_results)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("=" * 80)
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {duration}")
            logger.info("=" * 80)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_data(self) -> tuple:
        """Load all required data."""
        try:
            snp_data, methylation_data, annotation_data = self.data_loader.load_all_data()
            
            logger.info(f"Loaded {len(snp_data)} SNPs")
            logger.info(f"Loaded {len(methylation_data)} CpG sites")
            logger.info(f"Loaded {len(annotation_data)} annotations")
            
            return snp_data, methylation_data, annotation_data
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def _align_data(self, snp_data, methylation_data) -> tuple:
        """Align samples between SNP and methylation datasets."""
        try:
            aligned_snp_data, aligned_methylation_data = align_samples(snp_data, methylation_data)
            
            # Extract common sample info
            genotype_cols = [col for col in aligned_snp_data.columns if col.startswith('genotype_')]
            n_common_samples = len(genotype_cols)
            
            logger.info(f"Aligned datasets with {n_common_samples} common samples")
            
            return aligned_snp_data, aligned_methylation_data
            
        except Exception as e:
            logger.error(f"Data alignment failed: {e}")
            raise
    
    def _run_classic_models(self, snp_data, methylation_data, sample_labels) -> Dict:
        """Run classical ML models (LDA, HMM)."""
        try:
            # Convert sample_labels to pandas Series if provided
            labels_series = None
            if sample_labels:
                # Get common samples
                genotype_cols = [col for col in snp_data.columns if col.startswith('genotype_')]
                common_samples = [col.replace('genotype_', '') for col in genotype_cols]
                
                # Create labels series
                import pandas as pd
                labels_list = [sample_labels.get(sample, 'unknown') for sample in common_samples]
                labels_series = pd.Series(labels_list, index=common_samples)
            
            classic_results = self.classic_models.fit_all_models(
                snp_data, methylation_data, labels_series
            )
            
            logger.info("Classical ML models completed successfully")
            return classic_results
            
        except Exception as e:
            logger.warning(f"Classical models failed: {e}")
            return {'error': str(e)}
    
    def _run_ai_models(self, snp_data, methylation_data) -> Dict:
        """Run AI models (BERT, GPT)."""
        try:
            ai_results = self.ai_models.analyze_with_ai_models(snp_data, methylation_data)
            
            logger.info("AI models completed successfully")
            return ai_results
            
        except Exception as e:
            logger.warning(f"AI models failed: {e}")
            return {'error': str(e)}
    
    def _run_correlation_analysis(self, snp_data, methylation_data) -> Dict:
        """Run comprehensive correlation analysis."""
        try:
            correlation_results = self.correlation_analyzer.run_comprehensive_analysis(
                snp_data, methylation_data
            )
            
            logger.info("Correlation analysis completed successfully")
            return correlation_results
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            raise
    
    def _generate_visualizations(self, correlation_results, snp_data, methylation_data) -> Dict:
        """Generate all visualizations."""
        try:
            visualization_results = self.plotter.create_comprehensive_report(
                correlation_results, snp_data, methylation_data
            )
            
            logger.info(f"Generated {len(visualization_results)} visualization plots")
            return visualization_results
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
            return {'error': str(e)}
    
    def _compile_results(self, snp_data, methylation_data, annotation_data,
                        classic_results, ai_results, correlation_results, 
                        visualization_results) -> Dict:
        """Compile all results into final output."""
        
        final_results = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'config_used': self.config,
                'data_summary': {
                    'n_snps': len(snp_data),
                    'n_cpg_sites': len(methylation_data),
                    'n_annotations': len(annotation_data),
                    'n_samples_snp': len([col for col in snp_data.columns if col.startswith('genotype_')]),
                    'n_samples_methylation': methylation_data.shape[1]
                }
            },
            'correlation_analysis': correlation_results,
            'classical_models': classic_results,
            'ai_models': ai_results,
            'visualizations': visualization_results
        }
        
        # Add summary statistics
        if 'summary' in correlation_results:
            final_results['executive_summary'] = self._create_executive_summary(correlation_results['summary'])
        
        return final_results
    
    def _create_executive_summary(self, summary_data: Dict) -> Dict:
        """Create executive summary of key findings."""
        
        executive_summary = {
            'key_findings': [],
            'recommendations': [],
            'significant_associations': summary_data.get('significant_associations', 0),
            'analysis_methods_used': ['Statistical Correlation', 'Machine Learning', 'Window-based Analysis']
        }
        
        # Key findings
        n_significant = summary_data.get('significant_associations', 0)
        if n_significant > 0:
            executive_summary['key_findings'].append(
                f"Found {n_significant} statistically significant SNP-methylation associations"
            )
        
        if 'top_associations' in summary_data and summary_data['top_associations']:
            top_corr = summary_data['top_associations'][0]
            executive_summary['key_findings'].append(
                f"Strongest association: SNP {top_corr.get('snp_id', 'unknown')} with "
                f"correlation coefficient {top_corr.get('abs_correlation', 0):.3f}"
            )
        
        # Recommendations
        if n_significant == 0:
            executive_summary['recommendations'].append(
                "No significant associations found. Consider: (1) increasing sample size, "
                "(2) adjusting quality filters, (3) focusing on specific genomic regions"
            )
        elif n_significant < 10:
            executive_summary['recommendations'].append(
                "Few significant associations found. Consider validation in independent cohort"
            )
        else:
            executive_summary['recommendations'].append(
                "Multiple significant associations identified. Prioritize for functional validation"
            )
        
        return executive_summary
    
    def _save_results(self, results: Dict) -> None:
        """Save results to files."""
        results_dir = Path(self.config['output']['results_dir'])
        
        # Save main results as YAML
        results_file = results_dir / 'analysis_results.yaml'
        try:
            # Convert numpy arrays and other non-serializable objects to lists/dicts
            serializable_results = self._make_serializable(results)
            
            with open(results_file, 'w') as f:
                yaml.dump(serializable_results, f, default_flow_style=False, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save YAML results: {e}")
        
        # Save summary as text
        summary_file = results_dir / 'analysis_summary.txt'
        try:
            with open(summary_file, 'w') as f:
                self._write_text_summary(results, f)
            
            logger.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save text summary: {e}")
    
    def _make_serializable(self, obj):
        """Convert object to be YAML serializable."""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        else:
            return obj
    
    def _write_text_summary(self, results: Dict, f):
        """Write human-readable summary to text file."""
        f.write("GENOMICS CORRELATION ANALYSIS SUMMARY\\n")
        f.write("=" * 50 + "\\n\\n")
        
        # Metadata
        if 'metadata' in results:
            metadata = results['metadata']
            f.write(f"Analysis Date: {metadata['analysis_date']}\\n")
            
            if 'data_summary' in metadata:
                data_sum = metadata['data_summary']
                f.write(f"Dataset Size:\\n")
                f.write(f"  - SNPs: {data_sum.get('n_snps', 0):,}\\n")
                f.write(f"  - CpG Sites: {data_sum.get('n_cpg_sites', 0):,}\\n")
                f.write(f"  - Samples: {data_sum.get('n_samples_snp', 0):,}\\n\\n")
        
        # Executive summary
        if 'executive_summary' in results:
            exec_sum = results['executive_summary']
            f.write("KEY FINDINGS:\\n")
            for finding in exec_sum.get('key_findings', []):
                f.write(f"  • {finding}\\n")
            
            f.write("\\nRECOMMENDATIONS:\\n")
            for rec in exec_sum.get('recommendations', []):
                f.write(f"  • {rec}\\n")
            
            f.write(f"\\nSignificant Associations: {exec_sum.get('significant_associations', 0)}\\n")
        
        # Analysis methods
        f.write("\\nANALYSIS METHODS COMPLETED:\\n")
        
        if 'correlation_analysis' in results:
            f.write("  ✓ Statistical Correlation Analysis\\n")
        
        if 'classical_models' in results and 'error' not in results['classical_models']:
            f.write("  ✓ Classical ML Models (LDA, HMM)\\n")
        
        if 'ai_models' in results and 'error' not in results['ai_models']:
            f.write("  ✓ AI Models (BERT, GPT)\\n")
        
        if 'visualizations' in results:
            n_plots = len([v for v in results['visualizations'].values() if v is not None])
            f.write(f"  ✓ Visualizations ({n_plots} plots generated)\\n")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='Genomics Correlation Analysis Pipeline')
    parser.add_argument('config', help='Path to configuration YAML file')
    parser.add_argument('--labels', help='Path to sample labels file (optional)', default=None)
    parser.add_argument('--output-dir', help='Override output directory', default=None)
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    
    # Override output directory if specified
    if args.output_dir:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['output']['results_dir'] = args.output_dir
        
        # Save modified config
        temp_config_path = 'temp_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        config_path = temp_config_path
    
    # Load sample labels if provided
    sample_labels = None
    if args.labels:
        try:
            import pandas as pd
            labels_df = pd.read_csv(args.labels, index_col=0)
            sample_labels = labels_df.iloc[:, 0].to_dict()
            logger.info(f"Loaded labels for {len(sample_labels)} samples")
        except Exception as e:
            logger.warning(f"Failed to load sample labels: {e}")
    
    # Run pipeline
    try:
        pipeline = GenomicsPipeline(config_path)
        results = pipeline.run_full_pipeline(sample_labels)
        
        print("\\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        if 'executive_summary' in results:
            exec_sum = results['executive_summary']
            print(f"\\nSignificant associations found: {exec_sum.get('significant_associations', 0)}")
            
            if exec_sum.get('key_findings'):
                print("\\nKey findings:")
                for finding in exec_sum['key_findings'][:3]:  # Show top 3
                    print(f"  • {finding}")
        
        results_dir = Path(results['metadata']['config_used']['output']['results_dir'])
        print(f"\\nResults saved to: {results_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())