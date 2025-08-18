# Mistral OCR - Advanced PDF Table Extraction

An intelligent PDF processing pipeline that combines selective text enhancement with advanced table extraction, optimized for financial documents using Mistral's OCR API.

## 🎯 Key Features

- **🔍 Selective Text Enhancement**: Automatically detects and enhances pages with table structures
- **💰 Financial Document Detection**: Specialized recognition for invoices, billing statements, and financial reports  
- **📊 Advanced Table Extraction**: Multiple extraction methods for various document layouts
- **🗂️ Large File Handling**: Automatic chunking for PDFs exceeding API limits
- **📈 Intelligent Processing**: Adaptive algorithms based on document content and structure
- **🔧 Multiple Output Formats**: JSON, Markdown, and CSV table exports

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amit-dingare/mistralocr.git
   cd mistralocr
   ```

2. **Install dependencies**:
   ```bash
   # Basic installation
   pip install -r requirements.txt
   
   # Enhanced installation (recommended for full functionality)
   pip install -r requirements_enhanced.txt
   ```

3. **Configure API key**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your Mistral API key
   # MISTRAL_API_KEY=your_actual_api_key_here
   ```
   
   Get your API key from [Mistral Platform](https://console.mistral.ai/api-keys/)

### Basic Usage

**Extract tables from a PDF**:
```bash
python unified_enhanced_extractor.py --pdf invoice.pdf
```

**Specify output directory**:
```bash
python unified_enhanced_extractor.py --pdf document.pdf --output-dir results
```

**Use API key from command line**:
```bash
python unified_enhanced_extractor.py --pdf document.pdf --api-key YOUR_KEY
```

## 📋 How It Works

### Processing Pipeline

```
PDF Input → Size Analysis → Page Enhancement → OCR Processing → Table Extraction → CSV Export
```

### Intelligent Enhancement

The system analyzes each page for:

1. **Financial Keywords**: Invoice numbers, dates, account information
2. **Table Indicators**: Edge density and line structure analysis  
3. **Document Structure**: Layout complexity and data organization

### Enhancement Thresholds

- **Edge Density**: `> 0.035` (detects table borders and structure)
- **Line Density**: `> 0.004` (identifies horizontal/vertical table lines)
- **Financial Keywords**: Automatic flagging for invoice/billing content

## 📊 Output Files

### Generated Artifacts

For a document named `invoice.pdf`, the system creates:

```
results/
├── invoice_enhanced_ocr.json    # Raw OCR data with page information
├── invoice_enhanced_ocr.md      # Human-readable text with table structures  
├── invoice_page_1.csv           # Table data from page 1
├── invoice_page_3.csv           # Table data from page 3
└── invoice_page_7.csv           # Table data from page 7
```

### File Descriptions

- **JSON**: Complete OCR response with metadata and coordinates
- **Markdown**: Formatted text output with preserved table structures
- **CSV**: Clean tabular data ready for analysis or import

## ⚙️ Configuration Options

### Environment Variables

```bash
# Required
MISTRAL_API_KEY=your_mistral_api_key

# Optional advanced settings  
EDGE_DENSITY_THRESHOLD=0.035
LINE_DENSITY_THRESHOLD=0.004
```

### Command Line Options

```bash
python unified_enhanced_extractor.py --help

Options:
  --pdf PDF                     Path to PDF file (required)
  --output-dir OUTPUT_DIR       Output directory (default: unified_enhanced_results)
  --api-key API_KEY            Mistral API key (can use .env file instead)
```

## 📏 Large File Handling

### Automatic Chunking

The system automatically handles large PDFs using intelligent chunking:

- **Threshold**: 400MB original file size or 600MB enhanced file size
- **Strategy**: Dynamic chunk sizing based on file size and page count
- **Processing**: Transparent chunking with unified output

### Chunk Size Strategy

| File Size | Chunks | Pages per Chunk |
|-----------|--------|-----------------|
| < 400MB   | 2      | 50% of pages    |
| 400-800MB | 4      | 25% of pages    |
| > 800MB   | 6      | ~17% of pages   |

## 🎨 Use Cases

### Ideal Documents

✅ **Perfect for**:
- Invoice backups and billing statements
- Financial reports with tabular data  
- Labor timesheets and expense reports
- Account statements and transaction records
- Construction and project billing documents

❌ **Not recommended for**:
- Image-heavy PDFs without structured data
- Purely text documents without tables
- Low-quality scanned documents

## 🔧 Technical Details

### Dependencies

**Core Requirements**:
- `mistralai>=1.5.0` - OCR API integration
- `python-dotenv>=1.0.0` - Environment configuration

**Enhanced Processing** (recommended):
- `opencv-python>=4.8.0` - Image processing and enhancement
- `pillow>=10.0.0` - Image manipulation  
- `PyMuPDF>=1.23.0` - PDF processing and manipulation
- `numpy>=1.24.0` - Numerical operations

### Performance

- **Processing Speed**: ~1-2 pages per second (varies by complexity)
- **Memory Usage**: ~100-200MB base + 50MB per enhanced page
- **API Efficiency**: Intelligent chunking minimizes API calls
- **File Size Impact**: Enhanced PDFs typically 50-100% larger

## 🛠️ Advanced Usage

### Custom Enhancement Thresholds

For fine-tuning detection sensitivity:

```python
from unified_enhanced_extractor import UnifiedEnhancedExtractor

# Initialize with custom settings
processor = UnifiedEnhancedExtractor(api_key="your_key")
result = processor.process_pdf_unified("document.pdf")

# Results include processing statistics
print(f"Pages processed: {result['pages_processed']}")
print(f"Tables found: {result['tables_found']}")
print(f"Enhancement applied: {result['enhancement_applied']}")
```

### Batch Processing

For processing multiple files:

```bash
# Process all PDFs in a directory
for pdf in *.pdf; do
    python unified_enhanced_extractor.py --pdf "$pdf" --output-dir "batch_results"
done
```

## 📈 Performance Optimization

### Tips for Best Results

1. **File Quality**: Use high-resolution PDFs when possible
2. **File Size**: Consider splitting very large files (>2GB) manually
3. **Content Type**: Works best with structured financial documents
4. **API Limits**: Monitor usage and implement rate limiting for batch jobs

### Troubleshooting

**Common Issues**:
- **API Key Error**: Ensure `.env` file is configured correctly
- **File Size Error**: Large files automatically chunk, but very large files may need manual splitting
- **No Tables Found**: Check if document contains structured tabular data
- **Poor Quality Results**: Verify PDF quality and table structure clarity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes  
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: Additional documentation in the `/docs` folder
- **API Reference**: [Mistral OCR Documentation](https://docs.mistral.ai/)

---

**Developed by**: AI-Powered Infrastructure Lab  
**Version**: 2.0.0  
**Last Updated**: August 2025