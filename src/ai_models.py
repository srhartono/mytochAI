"""
AI Models for genomic analysis using BERT and GPT.

This module implements transformer-based models for analyzing genomic sequences,
SNP patterns, and methylation data using SOTA AI models
# TODO: Add DNA BERT, ENFORMER, others(?)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# Optional torch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. AI models will be disabled.")

# Optional transformers imports
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer,
        TrainingArguments, Trainer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. AI models will be disabled.")

logger = logging.getLogger(__name__)

class AIModels:
    """
    Container class for AI-based genomic analysis models.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.available = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE
        
        if self.available:
            self.bert_analyzer = BERTGenomicsAnalyzer(config['models']['ai']['bert'])
            self.gpt_analyzer = GPTGenomicsAnalyzer(config['models']['ai']['gpt'])
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            logger.warning("AI models not available due to missing dependencies")
        
    def analyze_with_ai_models(self, snp_data: pd.DataFrame, methylation_data: pd.DataFrame) -> Dict:
        """
        Perform AI-based analysis using BERT and GPT models.
        
        Args:
            snp_data: SNP genotype data
            methylation_data: DNA methylation data
            
        Returns:
            Dictionary containing AI model results
        """
        results = {'bert': {}, 'gpt': {}}
        
        if not self.available:
            logger.warning("AI models not available - returning empty results")
            results['bert']['error'] = "PyTorch/Transformers not available"
            results['gpt']['error'] = "PyTorch/Transformers not available"
            return results
        
        # Check data size
        if len(snp_data) < 5 or len(methylation_data) < 5:
            logger.warning("Dataset too small for AI analysis - returning empty results")
            results['bert']['error'] = "Dataset too small (need at least 5 samples/features)"
            results['gpt']['error'] = "Dataset too small (need at least 5 samples/features)"
            return results
        
        try:
            logger.info("Running BERT analysis...")
            results['bert'] = self.bert_analyzer.analyze_genomic_patterns(snp_data, methylation_data)
        except Exception as e:
            logger.error(f"BERT analysis failed: {e}")
            results['bert']['error'] = str(e)
        
        try:
            logger.info("Running GPT analysis...")
            results['gpt'] = self.gpt_analyzer.analyze_genomic_sequences(snp_data, methylation_data)
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            results['gpt']['error'] = str(e)
        
        return results


class BERTGenomicsAnalyzer:
    """
    BERT-based analyzer for genomic data patterns.
    
    Uses pre-trained BERT models (especially biomedical BERT) to:
    1. Encode genomic sequences and patterns
    2. Identify functional genomic regions
    3. Predict methylation states from sequence context
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get('model_name', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 16)
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded BERT model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load BERT model {self.model_name}: {e}")
            logger.info("Falling back to basic BERT model")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
    
    def analyze_genomic_patterns(self, snp_data: pd.DataFrame, methylation_data: pd.DataFrame) -> Dict:
        """
        Analyze genomic patterns using BERT embeddings.
        
        Args:
            snp_data: SNP data with genomic positions
            methylation_data: Methylation beta values
            
        Returns:
            Dictionary with BERT analysis results
        """
        results = {}
        
        # Create genomic sequence representations
        genomic_sequences = self._create_genomic_sequences(snp_data)
        
        # Generate BERT embeddings
        if genomic_sequences:
            embeddings = self._generate_bert_embeddings(genomic_sequences)
            results['sequence_embeddings'] = embeddings
            
            # Analyze methylation context
            methylation_context = self._analyze_methylation_context(methylation_data, embeddings)
            results['methylation_context'] = methylation_context
            
            # Functional region prediction
            functional_regions = self._predict_functional_regions(embeddings, snp_data)
            results['functional_regions'] = functional_regions
        
        return results
    
    def _create_genomic_sequences(self, snp_data: pd.DataFrame) -> List[str]:
        """
        Create genomic sequence representations from SNP data.
        
        This converts SNP information into sequence-like strings that BERT can process.
        """
        sequences = []
        
        # Group SNPs by chromosome and create sequence windows
        for chromosome in snp_data['chr'].unique():
            chrom_snps = snp_data[snp_data['chr'] == chromosome].sort_values('pos')
            
            if len(chrom_snps) < 2:
                continue
            
            # Create overlapping windows of SNPs
            window_size = 50  # SNPs per window
            step_size = 25    # Overlap
            
            for i in range(0, len(chrom_snps) - window_size + 1, step_size):
                window_snps = chrom_snps.iloc[i:i + window_size]
                
                # Create sequence representation
                # Format: "CHR:POS REF>ALT CHR:POS REF>ALT ..."
                sequence_parts = []
                for _, snp in window_snps.iterrows():
                    snp_str = f"{snp['chr']}:{snp['pos']} {snp['ref']}>{snp['alt']}"
                    sequence_parts.append(snp_str)
                
                sequence = " ".join(sequence_parts)
                sequences.append(sequence)
        
        return sequences[:100]  # Limit for computational efficiency
    
    def _generate_bert_embeddings(self, sequences: List[str]) -> np.ndarray:
        """
        Generate BERT embeddings for genomic sequences.
        """
        embeddings = []
        
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i + self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([])
    
    def _analyze_methylation_context(self, methylation_data: pd.DataFrame, 
                                   embeddings: np.ndarray) -> Dict:
        """
        Analyze methylation patterns in the context of BERT embeddings.
        """
        if embeddings.size == 0:
            return {'error': 'No embeddings available'}
        
        results = {}
        
        # Reduce dimensionality of embeddings for analysis
        # Adapt number of components to data size
        max_components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
        
        if embeddings.shape[1] > max_components and max_components > 0:
            pca = PCA(n_components=max_components)
            embeddings_reduced = pca.fit_transform(embeddings)
            results['explained_variance'] = pca.explained_variance_ratio_
        else:
            embeddings_reduced = embeddings
        
        # Correlate embeddings with methylation patterns
        # Use mean methylation across samples
        mean_methylation = methylation_data.mean(axis=1).values
        
        # Take subset matching embedding size
        n_common = min(len(embeddings_reduced), len(mean_methylation))
        
        if n_common > 10:
            from scipy.stats import pearsonr
            
            correlations = []
            for i in range(embeddings_reduced.shape[1]):
                corr, p_value = pearsonr(embeddings_reduced[:n_common, i], 
                                       mean_methylation[:n_common])
                correlations.append({'component': i, 'correlation': corr, 'p_value': p_value})
            
            results['embedding_methylation_correlations'] = correlations
            
            # Find most correlated embedding dimensions
            corr_df = pd.DataFrame(correlations)
            top_correlations = corr_df.nlargest(5, 'correlation')
            results['top_correlated_components'] = top_correlations.to_dict('records')
        
        return results
    
    def _predict_functional_regions(self, embeddings: np.ndarray, snp_data: pd.DataFrame) -> Dict:
        """
        Predict functional genomic regions using BERT embeddings.
        """
        if embeddings.size == 0:
            return {'error': 'No embeddings available'}
        
        results = {}
        
        # Cluster embeddings to identify functional regions
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Determine optimal number of clusters
        n_clusters_range = range(2, min(10, len(embeddings) // 5))
        silhouette_scores = []
        
        for n_clusters in n_clusters_range:
            if len(embeddings) < n_clusters:
                continue
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append({'n_clusters': n_clusters, 'score': score})
        
        if silhouette_scores:
            best_clustering = max(silhouette_scores, key=lambda x: x['score'])
            optimal_clusters = best_clustering['n_clusters']
            
            # Fit final clustering
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            final_clusters = kmeans.fit_predict(embeddings)
            
            results.update({
                'optimal_clusters': optimal_clusters,
                'cluster_labels': final_clusters,
                'cluster_centers': kmeans.cluster_centers_,
                'silhouette_score': best_clustering['score']
            })
            
            # Characterize clusters
            cluster_characteristics = []
            for cluster_id in range(optimal_clusters):
                cluster_mask = final_clusters == cluster_id
                cluster_size = np.sum(cluster_mask)
                
                cluster_characteristics.append({
                    'cluster_id': cluster_id,
                    'size': cluster_size,
                    'percentage': cluster_size / len(final_clusters) * 100
                })
            
            results['cluster_characteristics'] = cluster_characteristics
        
        return results


class GPTGenomicsAnalyzer:
    """
    GPT-based analyzer for genomic sequence generation and analysis.
    
    Uses GPT models to:
    1. Generate genomic sequence patterns
    2. Predict sequence properties
    3. Identify novel genomic motifs
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get('model_name', 'gpt2')
        self.max_length = config.get('max_length', 1024)
        self.temperature = config.get('temperature', 0.7)
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2Model.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded GPT model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load GPT model {self.model_name}: {e}")
            # Create a simple fallback
            self.model = None
            self.tokenizer = None
    
    def analyze_genomic_sequences(self, snp_data: pd.DataFrame, methylation_data: pd.DataFrame) -> Dict:
        """
        Analyze genomic sequences using GPT model.
        
        Args:
            snp_data: SNP data
            methylation_data: Methylation data
            
        Returns:
            Dictionary with GPT analysis results
        """
        if self.model is None:
            return {'error': 'GPT model not available'}
        
        results = {}
        
        # Create genomic text representations
        genomic_texts = self._create_genomic_text(snp_data, methylation_data)
        
        if genomic_texts:
            # Generate embeddings
            embeddings = self._generate_gpt_embeddings(genomic_texts)
            results['sequence_embeddings'] = embeddings
            
            # Sequence pattern analysis
            patterns = self._analyze_sequence_patterns(genomic_texts, embeddings)
            results['sequence_patterns'] = patterns
            
            # Generate new sequences (for motif discovery)
            generated_sequences = self._generate_sequences(genomic_texts[:5])  # Use first 5 as prompts
            results['generated_sequences'] = generated_sequences
        
        return results
    
    def _create_genomic_text(self, snp_data: pd.DataFrame, methylation_data: pd.DataFrame) -> List[str]:
        """
        Create text representations of genomic data for GPT analysis.
        """
        texts = []
        
        # Create narrative descriptions of genomic regions
        genotype_cols = [col for col in snp_data.columns if col.startswith('genotype_')]
        
        if not genotype_cols:
            return texts
        
        # Group by chromosome and create regional descriptions
        for chromosome in snp_data['chr'].unique()[:3]:  # Limit to first 3 chromosomes
            chrom_snps = snp_data[snp_data['chr'] == chromosome].sort_values('pos').head(20)
            
            if len(chrom_snps) < 5:
                continue
            
            # Create text description
            text_parts = [f"Genomic region on {chromosome}:"]
            
            for _, snp in chrom_snps.iterrows():
                # Calculate mean genotype
                mean_genotype = chrom_snps[genotype_cols].mean(axis=1).loc[snp.name]
                
                if mean_genotype < 0.5:
                    variant_desc = f"rare variant {snp['ref']}>{snp['alt']}"
                elif mean_genotype > 1.5:
                    variant_desc = f"common variant {snp['ref']}>{snp['alt']}"
                else:
                    variant_desc = f"heterozygous variant {snp['ref']}>{snp['alt']}"
                
                text_parts.append(f"Position {snp['pos']}: {variant_desc}")
            
            # Add methylation context if available
            if len(methylation_data) > 0:
                mean_meth = methylation_data.mean(axis=1).mean()
                if mean_meth > 0.7:
                    text_parts.append("High methylation region")
                elif mean_meth < 0.3:
                    text_parts.append("Low methylation region")
                else:
                    text_parts.append("Intermediate methylation region")
            
            text = ". ".join(text_parts) + "."
            texts.append(text)
        
        return texts
    
    def _generate_gpt_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate GPT embeddings for genomic texts.
        """
        embeddings = []
        
        for text in texts:
            # Tokenize
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use mean of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding)
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([])
    
    def _analyze_sequence_patterns(self, texts: List[str], embeddings: np.ndarray) -> Dict:
        """
        Analyze patterns in genomic sequences using GPT embeddings.
        """
        results = {}
        
        if embeddings.size == 0:
            return {'error': 'No embeddings available'}
        
        # Dimensionality reduction for visualization
        from sklearn.manifold import TSNE
        
        if len(embeddings) > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
            embeddings_2d = tsne.fit_transform(embeddings)
            
            results['tsne_coordinates'] = embeddings_2d
            
            # Cluster analysis
            from sklearn.cluster import DBSCAN
            
            clustering = DBSCAN(eps=0.5, min_samples=2)
            clusters = clustering.fit_predict(embeddings)
            
            results['clusters'] = clusters
            results['n_clusters'] = len(np.unique(clusters[clusters != -1]))
        
        # Text similarity analysis
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(embeddings)
        results['similarity_matrix'] = similarity_matrix
        
        # Find most similar sequence pairs
        n_sequences = len(texts)
        similarities = []
        
        for i in range(n_sequences):
            for j in range(i+1, n_sequences):
                similarities.append({
                    'sequence1_idx': i,
                    'sequence2_idx': j,
                    'similarity': similarity_matrix[i, j],
                    'sequence1': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                    'sequence2': texts[j][:100] + "..." if len(texts[j]) > 100 else texts[j]
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        results['top_similarities'] = similarities[:5]
        
        return results
    
    def _generate_sequences(self, prompt_texts: List[str]) -> List[str]:
        """
        Generate new genomic sequence descriptions using GPT.
        """
        if self.model is None or not prompt_texts:
            return []
        
        generated = []
        
        for prompt in prompt_texts:
            try:
                # Tokenize prompt
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=min(inputs.shape[1] + 50, self.max_length),
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the generated part (remove prompt)
                generated_part = generated_text[len(prompt):].strip()
                generated.append(generated_part)
                
            except Exception as e:
                logger.warning(f"Sequence generation failed: {e}")
                continue
        
        return generated


if TORCH_AVAILABLE:
    class GenomicsDataset(Dataset):
        """
        Custom dataset for training genomics-specific models.
        """
        
        def __init__(self, sequences: List[str], labels: Optional[List] = None, tokenizer=None, max_length: int = 512):
            self.sequences = sequences
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            
            # Tokenize
            encoding = self.tokenizer(
                sequence,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
            
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return item
else:
    class GenomicsDataset:
        """
        Placeholder dataset class when PyTorch is not available.
        """
        
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available - GenomicsDataset cannot be used")