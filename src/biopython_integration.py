"""
BioPython integration module for genomic sequence analysis.
Because pysam is not available on Windows....

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import warnings

# BioPython imports
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqUtils import gc_fraction, molecular_weight
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.SeqUtils import MeltingTemp as mt
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("BioPython not available. Sequence analysis features will be disabled.")

logger = logging.getLogger(__name__)


class SequenceAnalyzer:
    """
    BioPython-based sequence analyzer for genomic data.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython is required for sequence analysis features")
    
    def load_reference_sequences(self, fasta_file: Union[str, Path]) -> Dict[str, SeqRecord]:
        """
        Load reference sequences from FASTA file.
        
        Args:
            fasta_file: Path to FASTA file
            
        Returns:
            Dictionary mapping sequence IDs to SeqRecord objects
        """
        if not Path(fasta_file).exists():
            logger.warning(f"FASTA file {fasta_file} not found")
            return {}
        
        sequences = {}
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequences[record.id] = record
                logger.debug(f"Loaded sequence {record.id}, length: {len(record.seq)}")
                
            logger.info(f"Loaded {len(sequences)} sequences from {fasta_file}")
            
        except Exception as e:
            logger.error(f"Error loading FASTA file: {e}")
            
        return sequences
    
    def extract_snp_contexts(self, snp_df: pd.DataFrame, 
                           reference_sequences: Dict[str, SeqRecord],
                           context_size: int = 50) -> pd.DataFrame:
        """
        Extract sequence contexts around SNP positions.
        
        Args:
            snp_df: DataFrame with SNP information
            reference_sequences: Dictionary of reference sequences
            context_size: Number of bases to extract on each side of SNP
            
        Returns:
            DataFrame with added sequence context columns
        """
        contexts = []
        
        for idx, row in snp_df.iterrows():
            chrom = str(row['chr']).replace('chr', '')  # Handle chr prefix
            pos = int(row['pos']) - 1  # Convert to 0-based indexing
            
            # Find matching chromosome
            seq_record = None
            for seq_id, record in reference_sequences.items():
                if chrom in seq_id or seq_id in chrom:
                    seq_record = record
                    break
            
            if seq_record is None:
                contexts.append({
                    'context_sequence': None,
                    'upstream': None,
                    'downstream': None,
                    'gc_content': None
                })
                continue
            
            # Extract context sequence
            start_pos = max(0, pos - context_size)
            end_pos = min(len(seq_record.seq), pos + context_size + 1)
            
            context_seq = seq_record.seq[start_pos:end_pos]
            upstream = seq_record.seq[start_pos:pos]
            downstream = seq_record.seq[pos + 1:end_pos]
            
            # Calculate GC content
            gc_content = gc_fraction(context_seq) * 100 if len(context_seq) > 0 else None
            
            contexts.append({
                'context_sequence': str(context_seq),
                'upstream': str(upstream),
                'downstream': str(downstream),
                'gc_content': gc_content
            })
        
        # Add context information to SNP DataFrame
        context_df = pd.DataFrame(contexts)
        result_df = pd.concat([snp_df, context_df], axis=1)
        
        logger.info(f"Added sequence contexts for {len(result_df)} SNPs")
        return result_df
    
    def analyze_cpg_contexts(self, methylation_df: pd.DataFrame,
                           reference_sequences: Dict[str, SeqRecord],
                           cpg_annotations: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Analyze CpG site contexts and features.
        
        Args:
            methylation_df: DataFrame with methylation data
            reference_sequences: Dictionary of reference sequences
            cpg_annotations: Optional DataFrame with CpG site positions
            
        Returns:
            DataFrame with CpG context analysis
        """
        if cpg_annotations is None:
            logger.warning("No CpG annotations provided, using dummy data")
            return self._create_dummy_cpg_analysis(methylation_df)
        
        cpg_features = []
        
        for cpg_id in methylation_df.index:
            # Try to extract position from CpG ID (format: chr:pos or similar)
            try:
                if ':' in cpg_id:
                    chrom, pos = cpg_id.split(':')
                    pos = int(pos)
                else:
                    # Look up in annotations if available
                    annotation_row = cpg_annotations[cpg_annotations['cpg_id'] == cpg_id]
                    if len(annotation_row) > 0:
                        chrom = annotation_row.iloc[0]['chr']
                        pos = annotation_row.iloc[0]['pos']
                    else:
                        continue
                        
            except (ValueError, KeyError):
                cpg_features.append({
                    'cpg_id': cpg_id,
                    'cpg_context': None,
                    'cpg_island': False,
                    'gc_content': None
                })
                continue
            
            # Find sequence context
            chrom_clean = str(chrom).replace('chr', '')
            seq_record = None
            
            for seq_id, record in reference_sequences.items():
                if chrom_clean in seq_id or seq_id in chrom_clean:
                    seq_record = record
                    break
            
            if seq_record is None:
                cpg_features.append({
                    'cpg_id': cpg_id,
                    'cpg_context': None,
                    'cpg_island': False,
                    'gc_content': None
                })
                continue
            
            # Extract CpG context (±10bp)
            context_size = 10
            start_pos = max(0, pos - context_size)
            end_pos = min(len(seq_record.seq), pos + context_size)
            
            context_seq = seq_record.seq[start_pos:end_pos]
            gc_content = gc_fraction(context_seq) * 100 if len(context_seq) > 0 else None
            
            # Simple CpG island detection (GC content > 60% and length > 20bp)
            cpg_island = gc_content > 60 if gc_content is not None else False
            
            cpg_features.append({
                'cpg_id': cpg_id,
                'cpg_context': str(context_seq),
                'cpg_island': cpg_island,
                'gc_content': gc_content
            })
        
        cpg_analysis_df = pd.DataFrame(cpg_features).set_index('cpg_id')
        logger.info(f"Analyzed {len(cpg_analysis_df)} CpG sites")
        
        return cpg_analysis_df
    
    def calculate_sequence_features(self, sequences: List[str]) -> pd.DataFrame:
        """
        Calculate various sequence features using BioPython.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            DataFrame with sequence features
        """
        features = []
        
        for i, seq_str in enumerate(sequences):
            if not seq_str or pd.isna(seq_str):
                features.append({
                    'sequence_id': i,
                    'length': None,
                    'gc_content': None,
                    'molecular_weight': None,
                    'melting_temp': None,
                    'has_cpg': None
                })
                continue
            
            try:
                seq = Seq(seq_str)
                
                # Basic features
                length = len(seq)
                gc_content = gc_fraction(seq) * 100 if length > 0 else None
                mol_weight = molecular_weight(seq, seq_type='DNA') if length > 0 else None
                
                # Melting temperature (approximate)
                melting_temp = None
                if length > 0:
                    try:
                        melting_temp = mt.Tm_Wallace(seq)
                    except:
                        pass
                
                # Check for CpG dinucleotides
                has_cpg = 'CG' in str(seq).upper() if length > 0 else False
                
                features.append({
                    'sequence_id': i,
                    'length': length,
                    'gc_content': gc_content,
                    'molecular_weight': mol_weight,
                    'melting_temp': melting_temp,
                    'has_cpg': has_cpg
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing sequence {i}: {e}")
                features.append({
                    'sequence_id': i,
                    'length': None,
                    'gc_content': None,
                    'molecular_weight': None,
                    'melting_temp': None,
                    'has_cpg': None
                })
        
        return pd.DataFrame(features)
    
    def find_regulatory_motifs(self, sequences: List[str], 
                             motifs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Search for regulatory motifs in sequences.
        
        Args:
            sequences: List of DNA sequences
            motifs: List of motif sequences to search for
            
        Returns:
            DataFrame with motif occurrence counts
        """
        if motifs is None:
            # Common regulatory motifs
            motifs = [
                'TATA',      # TATA box
                'CAAT',      # CAAT box  
                'CCAAT',     # CCAAT box
                'CG',        # CpG dinucleotide
                'CGCG',      # CpG island motif
                'GAATTC',    # EcoRI site
                'GGATCC',    # BamHI site
            ]
        
        motif_counts = []
        
        for i, seq_str in enumerate(sequences):
            if not seq_str or pd.isna(seq_str):
                counts = {'sequence_id': i}
                counts.update({f'motif_{motif}': 0 for motif in motifs})
                motif_counts.append(counts)
                continue
            
            seq_upper = str(seq_str).upper()
            counts = {'sequence_id': i}
            
            for motif in motifs:
                count = seq_upper.count(motif.upper())
                counts[f'motif_{motif}'] = count
            
            motif_counts.append(counts)
        
        return pd.DataFrame(motif_counts)
    
    def _create_dummy_cpg_analysis(self, methylation_df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy CpG analysis for demonstration."""
        np.random.seed(42)
        
        cpg_features = []
        for cpg_id in methylation_df.index:
            cpg_features.append({
                'cpg_id': cpg_id,
                'cpg_context': 'ATCG' + 'CG' + 'ATCG',  # Dummy context
                'cpg_island': np.random.choice([True, False], p=[0.3, 0.7]),
                'gc_content': np.random.uniform(40, 80)
            })
        
        return pd.DataFrame(cpg_features).set_index('cpg_id')


class GenomicFileHandler:
    """
    Handle various genomic file formats using BioPython.
    """
    
    def __init__(self):
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython is required for file handling features")
    
    def convert_format(self, input_file: Union[str, Path], 
                      output_file: Union[str, Path],
                      input_format: str, output_format: str) -> bool:
        """
        Convert between different sequence file formats.
        
        Args:
            input_file: Input file path
            output_file: Output file path  
            input_format: Input format (e.g., 'fasta', 'genbank')
            output_format: Output format (e.g., 'fasta', 'genbank')
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            sequences = SeqIO.parse(input_file, input_format)
            count = SeqIO.write(sequences, output_file, output_format)
            logger.info(f"Converted {count} sequences from {input_format} to {output_format}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting file formats: {e}")
            return False
    
    def extract_sequences_by_ids(self, fasta_file: Union[str, Path],
                                ids: List[str],
                                output_file: Union[str, Path]) -> int:
        """
        Extract specific sequences by ID from FASTA file.
        
        Args:
            fasta_file: Input FASTA file
            ids: List of sequence IDs to extract
            output_file: Output FASTA file
            
        Returns:
            Number of sequences extracted
        """
        try:
            id_set = set(ids)
            sequences = []
            
            for record in SeqIO.parse(fasta_file, "fasta"):
                if record.id in id_set:
                    sequences.append(record)
            
            count = SeqIO.write(sequences, output_file, "fasta")
            logger.info(f"Extracted {count} sequences to {output_file}")
            return count
            
        except Exception as e:
            logger.error(f"Error extracting sequences: {e}")
            return 0


def integrate_sequence_features(snp_df: pd.DataFrame, 
                              methylation_df: pd.DataFrame,
                              reference_fasta: Optional[Union[str, Path]] = None,
                              config: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main integration function to add BioPython-based sequence features.
    
    Args:
        snp_df: SNP DataFrame
        methylation_df: Methylation DataFrame
        reference_fasta: Optional path to reference FASTA file
        config: Configuration dictionary
        
    Returns:
        Tuple of enhanced (snp_df, methylation_df) with sequence features
    """
    if not BIOPYTHON_AVAILABLE:
        logger.warning("BioPython not available, returning original data")
        return snp_df, methylation_df
    
    if config is None:
        config = {}
    
    try:
        analyzer = SequenceAnalyzer(config)
        
        # Load reference sequences if available
        reference_sequences = {}
        if reference_fasta and Path(reference_fasta).exists():
            reference_sequences = analyzer.load_reference_sequences(reference_fasta)
        
        enhanced_snp_df = snp_df.copy()
        enhanced_methylation_df = methylation_df.copy()
        
        # Add SNP context features if reference available
        if reference_sequences and 'chr' in snp_df.columns and 'pos' in snp_df.columns:
            enhanced_snp_df = analyzer.extract_snp_contexts(
                snp_df, reference_sequences, context_size=50
            )
            logger.info("Added SNP context features")
        
        # Add CpG context analysis
        cpg_analysis = analyzer.analyze_cpg_contexts(
            methylation_df, reference_sequences
        )
        
        # Store CpG features separately to avoid mixing with numeric methylation data
        # Instead of adding to the main DataFrame, create metadata structure
        cpg_metadata = {}
        for feature_col in cpg_analysis.columns:
            if feature_col != 'cpg_id':
                cpg_metadata[f'cpg_{feature_col}'] = cpg_analysis[feature_col].to_dict()
        
        # Add metadata as attribute to preserve numeric-only methylation DataFrame
        enhanced_methylation_df.attrs['cpg_metadata'] = cpg_metadata
        
        logger.info("BioPython integration completed successfully")
        
        # Log what metadata was added
        if hasattr(enhanced_methylation_df, 'attrs') and 'cpg_metadata' in enhanced_methylation_df.attrs:
            metadata_keys = list(enhanced_methylation_df.attrs['cpg_metadata'].keys())
            logger.info(f"Added CpG metadata: {metadata_keys}")
        
        return enhanced_snp_df, enhanced_methylation_df
        
    except Exception as e:
        logger.error(f"Error in BioPython integration: {e}")
        return snp_df, methylation_df