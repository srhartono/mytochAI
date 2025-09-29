"""
Data loading and preprocessing module for genomic data analysis.

This module handles SNP data (VCF format) and DNA methylation data,
providing unified interfaces for data loading, quality control, and preprocessing.
# TODO: sDRIP, EU-eq, END-seq, eCLIP-seq, Histone, TFBS, NET-seq, RNA-seq, MRD, ATAC-seq, Hi-C
# Benchmark performanecs

"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import warnings
from pathlib import Path
import logging

# Optional bioinformatics imports
try:
    import pysam
    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False
    warnings.warn("pysam not available. Some VCF processing features will be limited.")

try:
    import vcf
    VCF_AVAILABLE = True
except ImportError:
    VCF_AVAILABLE = False
    warnings.warn("pyvcf not available. VCF parsing will use alternative methods.")

# BioPython integration
try:
    from .biopython_integration import integrate_sequence_features, SequenceAnalyzer
    BIOPYTHON_INTEGRATION_AVAILABLE = True
except ImportError:
    BIOPYTHON_INTEGRATION_AVAILABLE = False
    warnings.warn("BioPython integration not available.")

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Main data loader class for genomic correlation analysis.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.snp_processor = SNPProcessor(config)
        self.methylation_processor = MethylationProcessor(config)
        self.bed_processor = BEDProcessor(config)  # NEW: BED file processor
        
        # Initialize BioPython integration if available
        self.use_biopython = BIOPYTHON_INTEGRATION_AVAILABLE and config.get('enable_biopython', True)
        if self.use_biopython:
            logger.info("BioPython integration enabled")
        else:
            logger.info("BioPython integration disabled or not available")
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess all required data.
        
        Returns:
            Tuple of (snp_data, methylation_data, annotation_data)
        """
        logger.info("Loading SNP data...")
        snp_data = self.snp_processor.load_and_process()
        
        logger.info("Loading methylation data...")
        methylation_data = self.methylation_processor.load_and_process()
        
        logger.info("Loading annotation data...")
        annotation_data = self.load_annotations()
        
        # Apply BioPython enhancements if enabled
        if self.use_biopython:
            logger.info("Applying BioPython sequence analysis...")
            reference_fasta = self.config.get('data', {}).get('reference_fasta')
            snp_data, methylation_data = integrate_sequence_features(
                snp_data, methylation_data, reference_fasta, self.config
            )
        
        return snp_data, methylation_data, annotation_data
    
    def load_bed_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all BED8 genomic interval data.
        
        Returns:
            Dictionary with structure: {assay_type: {condition: DataFrame}}
        """
        logger.info("Loading BED8 files for multi-omics analysis...")
        return self.bed_processor.load_all_bed_files()
    
    def load_treatment_comparison_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load BED data organized for treatment comparison analysis (AQR24h vs DMSO24h).
        
        Returns:
            Dictionary with structure: {assay_type: {'gain': DataFrame, 'loss': DataFrame}}
        """
        logger.info("Loading treatment comparison data...")
        return self.bed_processor.get_treatment_comparison_data()
    
    def load_annotations(self) -> pd.DataFrame:
        """Load genomic annotations from BED file."""
        annotation_file = self.config['data']['annotation_file']
        
        if not Path(annotation_file).exists():
            logger.warning(f"Annotation file {annotation_file} not found. Creating dummy annotations.")
            return pd.DataFrame(columns=['chr', 'start', 'end', 'feature'])
        
        # Standard BED format columns
        columns = ['chr', 'start', 'end', 'feature', 'score', 'strand']
        annotations = pd.read_csv(annotation_file, sep='\t', header=None, names=columns[:4])
        
        return annotations


class SNPProcessor:
    """
    Processor for SNP data in VCF format.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.quality_threshold = config['data']['quality_threshold']
        self.maf_threshold = config['data']['minor_allele_frequency']
        self.missing_threshold = config['data']['missing_data_threshold']
    
    def load_and_process(self) -> pd.DataFrame:
        """
        Load SNP data from VCF file and apply quality filters.
        
        Returns:
            DataFrame with processed SNP data
        """
        vcf_file = self.config['data']['snp_file']
        
        if not Path(vcf_file).exists():
            logger.warning(f"SNP file {vcf_file} not found. Generating sample data.")
            return self._generate_sample_snp_data()
        
        # Load VCF file
        vcf_reader = vcf.Reader(open(vcf_file, 'r'))
        
        snp_records = []
        for record in vcf_reader:
            # Quality filtering
            if record.QUAL is None or record.QUAL < self.quality_threshold:
                continue
                
            # Extract genotype information
            genotypes = []
            for sample in record.samples:
                if sample.called:
                    # Convert genotype to numeric format (0, 1, 2 for dosage)
                    gt = sample.gt_nums
                    if gt:
                        dosage = sum([int(x) for x in gt if x is not None])
                        genotypes.append(dosage)
                    else:
                        genotypes.append(np.nan)
                else:
                    genotypes.append(np.nan)
            
            # Check missing data threshold
            missing_rate = np.isnan(genotypes).sum() / len(genotypes)
            if missing_rate > self.missing_threshold:
                continue
            
            # Calculate minor allele frequency
            allele_counts = [x for x in genotypes if not np.isnan(x)]
            if len(allele_counts) == 0:
                continue
                
            total_alleles = sum(allele_counts)
            total_samples = len(allele_counts) * 2  # diploid
            maf = min(total_alleles / total_samples, 1 - total_alleles / total_samples)
            
            if maf < self.maf_threshold:
                continue
            
            # Store record
            snp_record = {
                'chr': record.CHROM,
                'pos': record.POS,
                'ref': record.REF,
                'alt': ','.join([str(alt) for alt in record.ALT]),
                'qual': record.QUAL,
                'maf': maf,
                'snp_id': f"{record.CHROM}_{record.POS}_{record.REF}_{record.ALT[0]}"
            }
            
            # Add sample genotypes
            sample_names = [sample.sample for sample in record.samples]
            for i, sample_name in enumerate(sample_names):
                snp_record[f'genotype_{sample_name}'] = genotypes[i]
            
            snp_records.append(snp_record)
        
        snp_df = pd.DataFrame(snp_records)
        logger.info(f"Loaded {len(snp_df)} SNPs after quality filtering")
        
        return snp_df
    
    def _generate_sample_snp_data(self) -> pd.DataFrame:
        """Generate sample SNP data with intentional correlation structure."""
        np.random.seed(42)
        
        n_snps = 1000
        n_samples = 100
        
        snp_data = {
            'chr': np.random.choice(['chr1', 'chr2', 'chr3'], n_snps),
            'pos': np.random.randint(1000000, 100000000, n_snps),
            'ref': np.random.choice(['A', 'T', 'C', 'G'], n_snps),
            'alt': np.random.choice(['A', 'T', 'C', 'G'], n_snps),
            'qual': np.random.uniform(30, 100, n_snps),
            'maf': np.random.uniform(0.05, 0.5, n_snps)
        }
        
        # Add SNP IDs
        snp_data['snp_id'] = [f"{chr}_{pos}_{ref}_{alt}" 
                             for chr, pos, ref, alt in 
                             zip(snp_data['chr'], snp_data['pos'], 
                                 snp_data['ref'], snp_data['alt'])]
        
        # Create structured genotype data with population stratification patterns
        # This will create correlations that methylation can respond to
        
        # Create 3 population groups with different allele frequencies
        group1_samples = range(0, 33)  # European-like
        group2_samples = range(33, 66)  # African-like  
        group3_samples = range(66, 100)  # Asian-like
        
        # Add sample genotypes with population structure
        for i in range(n_samples):
            sample_name = f'sample_{i:03d}'
            
            if i in group1_samples:
                # Group 1: Higher frequency of alt alleles for first 200 SNPs
                genotypes = []
                for snp_idx in range(n_snps):
                    if snp_idx < 200:  # "Causal" SNPs
                        prob = [0.3, 0.5, 0.2]  # Higher alt allele freq
                    else:
                        prob = [0.6, 0.3, 0.1]  # Background
                    genotypes.append(np.random.choice([0, 1, 2], p=prob))
                    
            elif i in group2_samples:
                # Group 2: Intermediate frequencies
                genotypes = []
                for snp_idx in range(n_snps):
                    if snp_idx < 200:
                        prob = [0.4, 0.4, 0.2]
                    else:
                        prob = [0.6, 0.3, 0.1]
                    genotypes.append(np.random.choice([0, 1, 2], p=prob))
                    
            else:
                # Group 3: Lower alt allele frequencies for causal SNPs
                genotypes = []
                for snp_idx in range(n_snps):
                    if snp_idx < 200:
                        prob = [0.7, 0.25, 0.05]  # Lower alt allele freq
                    else:
                        prob = [0.6, 0.3, 0.1]
                    genotypes.append(np.random.choice([0, 1, 2], p=prob))
            
            snp_data[f'genotype_{sample_name}'] = genotypes
        
        return pd.DataFrame(snp_data)


class MethylationProcessor:
    """
    Processor for DNA methylation data.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.beta_threshold = config['data']['methylation_beta_threshold']
        self.missing_threshold = config['data']['missing_data_threshold']
    
    def load_and_process(self) -> pd.DataFrame:
        """
        Load methylation data and apply quality filters.
        
        Returns:
            DataFrame with processed methylation data
        """
        methylation_file = self.config['data']['methylation_file']
        
        if not Path(methylation_file).exists():
            logger.warning(f"Methylation file {methylation_file} not found. Generating sample data.")
            return self._generate_sample_methylation_data()
        
        # Load methylation data (assume tab-separated format)
        try:
            methylation_df = pd.read_csv(methylation_file, sep='\t', index_col=0)
        except Exception as e:
            logger.error(f"Error loading methylation file: {e}")
            return self._generate_sample_methylation_data()
        
        # Apply quality filters
        methylation_df = self._apply_methylation_filters(methylation_df)
        
        logger.info(f"Loaded {len(methylation_df)} methylation sites after filtering")
        
        return methylation_df
    
    def _apply_methylation_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to methylation data."""
        
        # Remove sites with too much missing data
        missing_rates = df.isnull().sum(axis=1) / df.shape[1]
        df = df[missing_rates <= self.missing_threshold]
        
        # Remove sites with extreme beta values (potential artifacts)
        # Keep sites with reasonable variation
        mean_beta = df.mean(axis=1)
        std_beta = df.std(axis=1)
        
        df = df[(mean_beta >= self.beta_threshold) & 
                (mean_beta <= (1 - self.beta_threshold)) &
                (std_beta >= 0.1)]  # Remove invariant sites
        
        return df
    
    def _generate_sample_methylation_data(self) -> pd.DataFrame:
        """Generate sample methylation data with intentional SNP correlations."""
        np.random.seed(42)
        
        n_cpg_sites = 5000
        n_samples = 100
        
        # Generate CpG site IDs
        cpg_ids = [f"cg{i:07d}" for i in range(n_cpg_sites)]
        
        # Generate sample names
        sample_names = [f'sample_{i:03d}' for i in range(n_samples)]
        
        # Create methylation patterns that correlate with SNP population groups
        methylation_data = {}
        
        # Define the same population groups as in SNP data
        group1_samples = range(0, 33)  # European-like
        group2_samples = range(33, 66)  # African-like  
        group3_samples = range(66, 100)  # Asian-like
        
        for i, sample_name in enumerate(sample_names):
            beta_values = np.zeros(n_cpg_sites)
            
            # Create correlated methylation patterns based on population group
            if i in group1_samples:
                # Group 1: Higher methylation at first 500 CpG sites (correlated with high SNP alt freq)
                for cpg_idx in range(n_cpg_sites):
                    if cpg_idx < 500:  # "Responsive" CpG sites
                        # Higher baseline methylation + noise
                        beta_values[cpg_idx] = np.clip(np.random.beta(4, 2) + np.random.normal(0, 0.1), 0, 1)
                    elif cpg_idx < 1000:  # Moderately responsive
                        beta_values[cpg_idx] = np.clip(np.random.beta(3, 3) + np.random.normal(0, 0.1), 0, 1)
                    else:  # Background methylation
                        beta_values[cpg_idx] = np.clip(np.random.beta(2, 5) + np.random.normal(0, 0.1), 0, 1)
                        
            elif i in group2_samples:
                # Group 2: Intermediate methylation patterns
                for cpg_idx in range(n_cpg_sites):
                    if cpg_idx < 500:
                        beta_values[cpg_idx] = np.clip(np.random.beta(3, 3) + np.random.normal(0, 0.1), 0, 1)
                    elif cpg_idx < 1000:
                        beta_values[cpg_idx] = np.clip(np.random.beta(3, 4) + np.random.normal(0, 0.1), 0, 1)
                    else:
                        beta_values[cpg_idx] = np.clip(np.random.beta(2, 5) + np.random.normal(0, 0.1), 0, 1)
                        
            else:
                # Group 3: Lower methylation at responsive sites (anti-correlated with SNPs)
                for cpg_idx in range(n_cpg_sites):
                    if cpg_idx < 500:  # Lower methylation where SNPs have lower alt freq
                        beta_values[cpg_idx] = np.clip(np.random.beta(2, 4) + np.random.normal(0, 0.1), 0, 1)
                    elif cpg_idx < 1000:
                        beta_values[cpg_idx] = np.clip(np.random.beta(2, 4) + np.random.normal(0, 0.1), 0, 1)
                    else:
                        beta_values[cpg_idx] = np.clip(np.random.beta(2, 5) + np.random.normal(0, 0.1), 0, 1)
            
            # Add additional correlation structure: some CpG sites directly respond to genotype
            # Simulate cis-eQTM effects (SNPs affecting nearby methylation)
            for j in range(0, min(100, n_cpg_sites), 10):  # Every 10th of first 100 CpGs
                if i < 50:  # First half of samples
                    # Positive correlation: more alt alleles -> more methylation
                    correlation_effect = 0.3 * (i / 50.0)  # Gradient effect
                    beta_values[j] = np.clip(beta_values[j] + correlation_effect, 0, 1)
                else:
                    # Negative correlation: more alt alleles -> less methylation  
                    correlation_effect = -0.2 * ((i - 50) / 50.0)
                    beta_values[j] = np.clip(beta_values[j] + correlation_effect, 0, 1)
            
            methylation_data[sample_name] = beta_values
        
        methylation_df = pd.DataFrame(methylation_data, index=cpg_ids)
        
        logger.info("Generated methylation data with intentional SNP correlations:")
        logger.info(f"  - Group 1 (samples 0-32): High methylation at CpGs 0-499")
        logger.info(f"  - Group 2 (samples 33-65): Intermediate methylation")
        logger.info(f"  - Group 3 (samples 66-99): Lower methylation at responsive sites")
        logger.info(f"  - Gradient correlation effects at CpGs 0, 10, 20, ..., 90")
        
        return methylation_df
    
    def convert_beta_to_m_values(self, beta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert beta values to M-values for statistical analysis.
        
        M-value = log2(beta / (1 - beta))
        """
        # Avoid division by zero and log of zero
        beta_clipped = np.clip(beta_df, 1e-6, 1 - 1e-6)
        m_values = np.log2(beta_clipped / (1 - beta_clipped))
        
        return pd.DataFrame(m_values, index=beta_df.index, columns=beta_df.columns)


def align_samples(snp_df: pd.DataFrame, methylation_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align samples between SNP and methylation datasets.
    
    Args:
        snp_df: SNP DataFrame with genotype columns
        methylation_df: Methylation DataFrame with sample columns
        
    Returns:
        Tuple of aligned (snp_df, methylation_df)
    """
    # Extract sample names from SNP genotype columns
    snp_sample_cols = [col for col in snp_df.columns if col.startswith('genotype_')]
    snp_samples = [col.replace('genotype_', '') for col in snp_sample_cols]
    
    # Get methylation sample names
    methylation_samples = methylation_df.columns.tolist()
    
    # Find common samples
    common_samples = list(set(snp_samples) & set(methylation_samples))
    logger.info(f"Found {len(common_samples)} common samples between SNP and methylation data")
    
    if len(common_samples) == 0:
        logger.warning("No common samples found between datasets!")
        return snp_df, methylation_df
    
    # Select only common samples
    snp_genotype_cols = [f'genotype_{sample}' for sample in common_samples]
    snp_metadata_cols = [col for col in snp_df.columns if not col.startswith('genotype_')]
    
    aligned_snp_df = snp_df[snp_metadata_cols + snp_genotype_cols].copy()
    aligned_methylation_df = methylation_df[common_samples].copy()
    
    return aligned_snp_df, aligned_methylation_df


class BEDProcessor:
    """
    Processor for BED8 files containing genomic interval data from various assays
    (END-seq, EU-seq, sDRIP-seq, RNA-seq, etc.)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config.get('data', {}).get('data_dir', 'data'))
        self.bed_files = {}
        
    def load_all_bed_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all BED8 files from the data directory and categorize them by assay type and condition.
        
        Returns:
            Dictionary with structure: {assay_type: {condition: DataFrame}}
        """
        # Discover all BED8 files in data directory
        bed_files = list(self.data_dir.glob("*.bed8"))
        
        if not bed_files:
            logger.warning(f"No BED8 files found in {self.data_dir}")
            return {}
        
        categorized_data = {}
        
        for bed_file in bed_files:
            assay_info = self._parse_filename(bed_file.name)
            if not assay_info:
                continue
                
            # Load the BED file
            bed_data = self._load_bed8_file(bed_file)
            
            # Categorize by assay type
            assay_type = assay_info['assay_type']
            condition = assay_info['condition']
            
            if assay_type not in categorized_data:
                categorized_data[assay_type] = {}
            
            categorized_data[assay_type][condition] = bed_data
            
            logger.info(f"Loaded {len(bed_data)} regions from {bed_file.name}")
        
        self.bed_files = categorized_data
        return categorized_data
    
    def _parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Parse BED file name to extract assay type and experimental condition.
        
        Expected patterns:
        - ENDseq_AQR24h_all_gain.bed8 -> assay: ENDseq, condition: AQR24h_all_gain
        - EUjunc_AQR24h_DMSO24h_gain.bed8 -> assay: EUjunc, condition: AQR24h_DMSO24h_gain
        - sDRIPcombined_AQR24h_DMSO24h_loss.bed8 -> assay: sDRIP, condition: AQR24h_DMSO24h_loss
        """
        if not filename.endswith('.bed8'):
            return None
        
        name_parts = filename.replace('.bed8', '').split('_')
        
        if len(name_parts) < 2:
            return None
        
        # Extract assay type (first part)
        assay_type = name_parts[0]
        
        # Handle special cases
        if assay_type == 'sDRIPcombined':
            assay_type = 'sDRIP'
        elif assay_type == 'EUcombined':
            assay_type = 'EU'
        elif 'HCT116' in name_parts[0]:
            assay_type = 'ENDseq_CPT'
        
        # Extract condition (remaining parts)
        condition = '_'.join(name_parts[1:])
        
        return {
            'assay_type': assay_type,
            'condition': condition,
            'filename': filename
        }
    
    def _load_bed8_file(self, bed_file: Path) -> pd.DataFrame:
        """
        Load a BED8 format file.
        
        BED8 columns: chr, start, end, name, score, strand, transcript_id, gene_name
        """
        try:
            bed_data = pd.read_csv(
                bed_file, 
                sep='\t', 
                header=None,
                names=['chr', 'start', 'end', 'annotation', 'score', 'strand', 'transcript_id', 'gene_name']
            )
            
            # Add filename info
            bed_data['source_file'] = bed_file.name
            
            # Parse annotation field for genomic features
            bed_data = self._parse_annotation_field(bed_data)
            
            # Calculate region length
            bed_data['length'] = bed_data['end'] - bed_data['start']
            
            return bed_data
            
        except Exception as e:
            logger.error(f"Error loading BED file {bed_file}: {e}")
            return pd.DataFrame()
    
    def _parse_annotation_field(self, bed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the annotation field to extract genomic feature information.
        
        Examples:
        - "GENEBODY_NEG218,POS231;ENST00000378609.9;GNB1" -> feature_type: GENEBODY
        - "PROMOTER_NEG_11;ENST00000354700.10;ACAP3" -> feature_type: PROMOTER
        """
        feature_types = []
        feature_positions = []
        
        for annotation in bed_data['annotation']:
            if pd.isna(annotation):
                feature_types.append('UNKNOWN')
                feature_positions.append('')
                continue
            
            parts = str(annotation).split(';')
            if len(parts) > 0:
                # Extract feature type from first part
                feature_info = parts[0]
                
                if 'GENEBODY' in feature_info:
                    feature_types.append('GENEBODY')
                elif 'PROMOTER' in feature_info:
                    feature_types.append('PROMOTER')
                elif 'TERMINAL' in feature_info:
                    feature_types.append('TERMINAL')
                elif 'TERMINTER' in feature_info:
                    feature_types.append('TERMINTER')
                elif 'ANTIPROMOTER' in feature_info:
                    feature_types.append('ANTIPROMOTER')
                elif 'JUNC' in feature_info:
                    feature_types.append('JUNCTION')
                elif 'INTERGENIC' in feature_info:
                    feature_types.append('INTERGENIC')
                elif 'PEAK' in feature_info:
                    feature_types.append('PEAK')
                else:
                    feature_types.append('OTHER')
                
                # Extract position information if present
                feature_positions.append(feature_info)
            else:
                feature_types.append('UNKNOWN')
                feature_positions.append('')
        
        bed_data['feature_type'] = feature_types
        bed_data['feature_position'] = feature_positions
        
        return bed_data
    
    def get_treatment_comparison_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Organize data for AQR24h treatment vs control comparisons.
        
        Returns:
            Dictionary with structure: {assay_type: {'gain': DataFrame, 'loss': DataFrame, 'both': DataFrame}}
        """
        if not self.bed_files:
            self.load_all_bed_files()
        
        treatment_data = {}
        
        for assay_type, conditions in self.bed_files.items():
            treatment_data[assay_type] = {}
            
            # Look for gain/loss/both patterns in AQR24h treatment
            for condition, data in conditions.items():
                if 'AQR24h' in condition and 'DMSO24h' in condition:
                    if 'gain' in condition:
                        treatment_data[assay_type]['gain'] = data
                    elif 'loss' in condition:
                        treatment_data[assay_type]['loss'] = data
                    elif 'both' in condition:
                        treatment_data[assay_type]['both'] = data
                elif 'AQR24h' in condition and 'all' in condition:
                    treatment_data[assay_type]['all_gain'] = data
        
        return treatment_data


def get_expected_correlations() -> Dict:
    """
    Return information about expected correlations in sample data.
    
    This helps users understand what patterns should be detected in the demo.
    """
    return {
        'population_structure': {
            'group1_samples': list(range(0, 33)),
            'group2_samples': list(range(33, 66)), 
            'group3_samples': list(range(66, 100)),
            'description': 'Three population groups with different allele frequencies'
        },
        'snp_patterns': {
            'causal_snps': list(range(0, 200)),
            'background_snps': list(range(200, 1000)),
            'description': 'First 200 SNPs have population-specific allele frequencies'
        },
        'methylation_patterns': {
            'highly_responsive_cpgs': list(range(0, 500)),
            'moderately_responsive_cpgs': list(range(500, 1000)),
            'background_cpgs': list(range(1000, 5000)),
            'gradient_effect_cpgs': list(range(0, 100, 10)),
            'description': 'CpG sites with different correlation strengths to SNP groups'
        },
        'expected_associations': [
            {
                'snp_range': 'SNPs 0-199 (population-stratified)',
                'cpg_range': 'CpGs 0-499 (highly responsive)', 
                'correlation_type': 'Strong positive correlation in groups 1-2, negative in group 3',
                'effect_size': 'Large (0.6-0.8 correlation)'
            },
            {
                'snp_range': 'All SNPs', 
                'cpg_range': 'CpGs 0, 10, 20, ..., 90',
                'correlation_type': 'Gradient effect across samples',
                'effect_size': 'Medium (0.3-0.5 correlation)'
            },
            {
                'snp_range': 'SNPs 0-199',
                'cpg_range': 'CpGs 500-999', 
                'correlation_type': 'Moderate correlation',
                'effect_size': 'Medium (0.2-0.4 correlation)'
            }
        ]
    }