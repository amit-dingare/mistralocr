# Unified Enhanced Extractor Overview

## Purpose
A comprehensive PDF processing tool that combines intelligent text enhancement with advanced table extraction, specifically optimized for financial documents like invoices and billing statements.

## Core Functionality

### 🎯 Primary Features
1. **Selective Text Enhancement** - Intelligently darkens text on pages with table structures
2. **Financial Document Detection** - Automatically identifies invoice/billing documents
3. **Table Extraction** - Extracts structured data into CSV files
4. **Large File Handling** - Automatically chunks large PDFs to handle API limits
5. **OCR Integration** - Uses Mistral OCR for text recognition

### 🔧 Technical Architecture

#### Main Processing Pipeline
```
PDF Input → Size Check → Enhancement Analysis → OCR Processing → Table Extraction → CSV Output
```

#### Key Components

**1. File Size Management**
- Original PDF threshold: 400MB
- Enhanced PDF limit: 600MB
- Automatic chunking for large files
- Temporary file cleanup

**2. Selective Enhancement Logic**
```python
# Financial keyword detection
has_financial_keywords = ['INVOICE NO', 'TYPE', 'REFERENCE', 'INVOICE DATE', 
                         'DUE DATE', 'DEBIT', 'CREDIT', 'STATEMENT NO', 'ACCOUNT NUMBER']

# Threshold-based detection  
edge_density > 0.035 AND line_density > 0.004
```

**3. Enhancement Algorithms**
- **Conservative Table Enhancement**: Standard approach for clear table structures
- **Aggressive Table Enhancement**: Advanced processing for complex layouts
- **Image Analysis**: Edge detection and line density calculations

**4. Table Extraction Methods**
- **Financial Table Reconstruction**: Specialized for invoice/billing formats
- **Markdown Table Extraction**: General table structure parsing
- **Scattered Data Reconstruction**: Handles poorly structured data

## Dependencies

### Required Libraries
- `mistralai` - OCR API integration
- `cv2` (OpenCV) - Image processing
- `numpy` - Numerical operations  
- `PIL` (Pillow) - Image manipulation
- `fitz` (PyMuPDF) - PDF processing

### API Requirements
- Mistral API key for OCR services
- Internet connection for API calls

## Usage Patterns

### Command Line Interface
```bash
python unified_enhanced_extractor.py --pdf input.pdf --output-dir results [--api-key KEY]
```

### Processing Flow
1. **Input Validation** - Check file existence and size
2. **Enhancement Decision** - Analyze each page for table indicators
3. **Text Enhancement** - Apply selective darkening algorithms
4. **OCR Processing** - Extract text using Mistral API
5. **Table Detection** - Identify and extract tabular data
6. **Output Generation** - Create JSON, Markdown, and CSV files

## Output Files

### Generated Artifacts
- `{filename}_enhanced_ocr.json` - Raw OCR data
- `{filename}_enhanced_ocr.md` - Human-readable text with tables
- `{filename}_page_{N}.csv` - Individual table extractions per page

### File Structure
```
output_directory/
├── filename_enhanced_ocr.json
├── filename_enhanced_ocr.md
├── filename_page_1.csv
├── filename_page_2.csv
└── ...
```

## Performance Characteristics

### Processing Metrics
- **Enhancement Coverage**: Up to 23 pages (vs 19 in conservative mode)
- **File Size Impact**: ~10% increase in enhanced PDF size
- **Memory Usage**: Optimized with temporary file cleanup
- **API Efficiency**: Chunked processing for large documents

### Optimization Features
- Automatic size limit detection
- Intelligent chunking strategies
- Selective enhancement to reduce processing overhead
- Financial document prioritization

## Key Improvements (Current Version)

### Enhanced Detection
1. **Financial Keyword Recognition** - Automatic flagging of invoice/billing content
2. **Lowered Thresholds** - More sensitive table detection (42% lower)
3. **Moderate Structure Handling** - Special processing for borderline cases

### Processing Intelligence
- Pages with financial keywords are automatically enhanced
- Edge density thresholds: 0.05 → 0.035 (30% more sensitive)
- Line density thresholds: 0.005 → 0.004 (20% more sensitive)

## Use Cases

### Ideal Documents
- ✅ Invoice backups and billing statements
- ✅ Financial reports with tabular data
- ✅ Labor billing and timesheet documents
- ✅ Account statements and transaction records

### Document Types to Avoid
- ❌ Image-heavy PDFs without tables
- ❌ Purely text documents without structure
- ❌ Scanned documents with poor quality

## Error Handling

### Robust Recovery
- Graceful degradation when enhancement libraries unavailable
- Automatic fallback for oversized files
- Cleanup of temporary files on failure
- Comprehensive logging and status reporting

---

*Tool Version: Enhanced (Current)*  
*Optimization Focus: Financial Documents*  
*API Integration: Mistral OCR Latest*