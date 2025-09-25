# mytochAI - Vanilla

An AI framework for analyzing the link between multi-omics data (SNP, DNA methylation) using classical machine learning models (LDA, HMM) and SOTA AI models (GPT, BERT, Enformer).

## Features

### 🧬 Data Processing
- **SNP Data**: VCF file support with quality filtering and minor allele frequency thresholds
- **Methylation Data**: Beta value processing with missing data handling
- **Sample Alignment**: Automatic alignment of samples between SNP and methylation datasets

### 🤖 Machine Learning Models

#### Classical Models
- **Linear Discriminant Analysis (LDA)**: Dimensionality reduction and classification
- **Hidden Markov Models (HMM)**: Pattern recognition in genomic sequences and methylation domains

#### AI Models  
- **BERT**: Biomedical text analysis for genomic sequence patterns
- **GPT**: Sequence generation and pattern discovery
- **DNABERT**: (WIP)
- **Enformer**: (WIP)

### 📊 Analysis Methods
- **Statistical Correlations**: Pearson, Spearman, and Kendall correlations with multiple testing correction
- **Machine Learning**: Feature importance analysis using Random Forest and mutual information
- **Window-based Analysis**: Genomic window sliding analysis for local correlation patterns
- **Non-linear Associations**: Detection using polynomial features and regularization

### 📈 Visualization
- Manhattan plots for genome-wide association patterns
- Correlation heatmaps and distribution plots  
- Principal component analysis visualizations
- Interactive dashboards with Plotly
- Genomic landscape plots

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for AI models)

### Quick Setup with Virtual Environment

**Automated Setup (Recommended):**
```bash
# Clone or download the framework
cd mytochai

# Run automated virtual environment setup
python setup_venv.py
```

**Manual Setup:**

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

**Linux/macOS:**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Optional GPU Support
```bash
# Install PyTorch with CUDA support (after activating virtual environment)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Prepare Your Data

#### SNP Data (VCF format)
Place your VCF file in the `data/` directory:
```
data/snps.vcf
```

#### Methylation Data (Tab-separated)
Methylation beta values with CpG sites as rows and samples as columns:
```
data/methylation.txt
```

#### Annotations (BED format, optional)
```
data/annotations.bed
```

### 2. Configure Analysis

Edit `config.yaml` to specify your data files and analysis parameters:

```yaml
data:
  snp_file: "data/snps.vcf"
  methylation_file: "data/methylation.txt"
  quality_threshold: 30
  minor_allele_frequency: 0.05

analysis:
  correlation:
    methods: ["pearson", "spearman"]
    p_value_threshold: 0.05
    
models:
  classic:
    lda:
      n_components: 10
    hmm:
      n_components: 5
```

### 3. Run Analysis

**Important**: Always activate the virtual environment first!

**Windows:**
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run analysis
python main.py --config config.yaml

# Or run demo with sample data
python main.py --demo
```

**Linux/macOS:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Run analysis
python main.py --config config.yaml

# Or run demo with sample data
python main.py --demo
```

**Additional Options:**
```bash
# With custom output directory
python main.py --config config.yaml --output-dir my_results/

# With sample labels
python main.py --config config.yaml --labels data/sample_labels.csv
```

### 4. Programmatic Usage

```python
from src import GenomicsPipeline

# Initialize pipeline
pipeline = GenomicsPipeline('config.yaml')

# Run complete analysis
results = pipeline.run_full_pipeline()

# Access specific results
correlation_results = results['correlation_analysis']
ml_results = results['classical_models']
ai_results = results['ai_models']

# Generate additional plots
from src import GenomicsPlotter
plotter = GenomicsPlotter(pipeline.config)
plots = plotter.create_comprehensive_report(
    results['correlation_analysis'], 
    snp_data, 
    methylation_data
)
```

## Example Usage

### Basic Correlation Analysis
```python
from src import CorrelationAnalyzer, DataLoader

# Load data
loader = DataLoader(config)
snp_data, meth_data, annotations = loader.load_all_data()

# Run correlation analysis
analyzer = CorrelationAnalyzer(config)
results = analyzer.run_comprehensive_analysis(snp_data, meth_data)

print(f"Found {results['summary']['significant_associations']} significant associations")
```

### Classical ML Models
```python
from src import ClassicModels

# Initialize models
classic_models = ClassicModels(config)

# Fit LDA and HMM models
results = classic_models.fit_all_models(snp_data, meth_data, sample_labels)

# Access LDA results
lda_results = results['lda']['snp']
print(f"LDA explained variance: {lda_results['explained_variance_ratio']}")

# Access HMM methylation domains
hmm_results = results['hmm']['methylation_domains']
domains = hmm_results['domain_statistics']
```

### AI Models
```python
from src import AIModels

# Initialize AI models
ai_models = AIModels(config)

# Run BERT and GPT analysis
results = ai_models.analyze_with_ai_models(snp_data, meth_data)

# Access BERT embeddings
bert_embeddings = results['bert']['sequence_embeddings']

# Access generated sequences from GPT
generated_seqs = results['gpt']['generated_sequences']
```

## Configuration Options

### Data Processing
- `quality_threshold`: Minimum SNP quality score (default: 30)
- `minor_allele_frequency`: MAF threshold (default: 0.05)  
- `missing_data_threshold`: Maximum missing data rate (default: 0.1)

### Analysis Methods
- `correlation_methods`: Statistical correlation methods to use
- `p_value_threshold`: Significance threshold (default: 0.05)
- `fdr_correction`: Multiple testing correction method

### Model Parameters
- LDA: `n_components`, `solver`, `shrinkage`
- HMM: `n_components`, `covariance_type`, `n_iter`
- BERT: `model_name`, `max_length`, `batch_size`
- GPT: `model_name`, `max_length`, `temperature`

### Visualization
- `output_format`: Plot formats (png, pdf, html, svg)
- `dpi`: Resolution for static plots
- `color_palette`: Color scheme

## Output Files

The framework generates comprehensive output in the `results/` directory:

### Analysis Results
- `analysis_results.yaml`: Complete results in structured format
- `analysis_summary.txt`: Human-readable summary

### Visualizations
- `correlation_distribution.png`: Distribution of correlation coefficients
- `manhattan_plot.html`: Interactive Manhattan plot
- `top_correlations_heatmap.png`: Heatmap of strongest associations
- `feature_importance.png`: ML feature importance plots
- `genomic_landscape.html`: Interactive genomic correlation landscape
- `summary_dashboard.html`: Interactive summary dashboard

### Data Quality
- `snp_overview.png`: SNP data quality metrics
- `methylation_overview.png`: Methylation data overview
- `data_quality.png`: Cross-dataset quality assessment

## Advanced Usage

### Custom Analysis
```python
# Custom window analysis
from src import WindowBasedAnalyzer

window_analyzer = WindowBasedAnalyzer(config)
window_results = window_analyzer.analyze_genomic_windows(snp_data, meth_data)

# Access high-correlation windows
high_corr_windows = [w for w in window_results['chromosome_analysis']['chr1']['windows'] 
                     if w.get('high_correlation', False)]
```

### Feature Selection
```python
from src import MLCorrelationAnalyzer

ml_analyzer = MLCorrelationAnalyzer(config)
results = ml_analyzer.analyze_ml_associations(snp_data, meth_data)

# Get top features by mutual information
top_features = results['mutual_information']['top_mi_features']
```

### Custom Visualization
```python
from src import GenomicsPlotter
import matplotlib.pyplot as plt

plotter = GenomicsPlotter(config)

# Create custom correlation plot
fig, ax = plt.subplots(figsize=(10, 6))
# ... custom plotting code ...
plotter._save_plot(fig, 'custom_plot')
```

## Data Formats

### SNP Data (VCF)
Standard VCF format with genotype information:
```
##fileformat=VCFv4.2
#CHROM  POS     ID      REF ALT QUAL    FILTER  INFO    FORMAT  SAMPLE1 SAMPLE2
chr1    1000    rs1     A   T   60      PASS    .       GT      0/0     0/1
chr1    2000    rs2     G   C   45      PASS    .       GT      1/1     0/0
```

### Methylation Data
Tab-separated file with CpG sites as rows and samples as columns:
```
CpG_ID          SAMPLE1 SAMPLE2 SAMPLE3
cg00000029      0.123   0.456   0.789
cg00000165      0.234   0.567   0.890
cg00000236      0.345   0.678   0.901
```

### Sample Labels (Optional)
CSV file mapping sample IDs to phenotypes:
```
Sample,Label
SAMPLE1,case
SAMPLE2,control
SAMPLE3,case
```

## Performance Considerations

### Memory Usage
- Large datasets: Use `maxResults` parameters to limit analysis scope
- Enable sample data generation for testing: Files will be auto-generated if missing

### Computational Efficiency  
- SNP filtering reduces analysis scope significantly
- Window-based analysis can be memory intensive for whole-genome data
- AI models require substantial computational resources

### Optimization Tips
- Start with subset of chromosomes for testing
- Adjust window sizes based on available memory
- Use GPU acceleration for AI models when available

## Troubleshooting

### Common Issues

**Virtual Environment Issues**
- **Windows PowerShell execution policy**: 
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **Virtual environment not activating**: Try running `setup_venv.py` again
- **Import errors after installation**: Make sure you activated the environment and ran `pip install -e .`

**"No genotype columns found"**
- Ensure VCF file has proper FORMAT and sample columns
- Check that sample names don't contain special characters

**"Insufficient common samples"**
- Verify sample naming consistency between SNP and methylation data
- Check for case sensitivity in sample names

**"BERT/GPT model loading failed"**  
- Install transformers: `pip install transformers>=4.20.0`
- For biomedical BERT, ensure internet connection for model download

**Memory errors**
- Reduce dataset size or use sampling
- Increase system swap space
- Use more restrictive quality filters

### Performance Optimization

**For large datasets:**
```yaml
data:
  quality_threshold: 40        # Stricter filtering
  minor_allele_frequency: 0.1  # Higher MAF threshold

analysis:
  window_analysis:
    window_size: 5000000       # Larger windows
    step_size: 2500000         # Larger steps
```

**For testing:**
```yaml
models:
  classic:
    lda:
      n_components: 5          # Fewer components
  ai:
    bert:
      batch_size: 8            # Smaller batches
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```
mytochAI: A comprehensive AI toolkit for analyzing 
meta-omics associations using classical and AI-based methods. 
Version 1.0.0 (2024).
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in the `examples/` directory
- Review the configuration examples in `config.yaml`

---