#!/usr/bin/env python3
"""
Unified Enhanced Extractor - Applies text darkening and extracts tables in one process
This ensures the enhanced OCR results are used for table extraction
"""

import os
import json
import csv
import argparse
import tempfile
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk, TextChunk

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    import fitz  # PyMuPDF
    PROCESSING_AVAILABLE = True
except ImportError:
    PROCESSING_AVAILABLE = False

class UnifiedEnhancedExtractor:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.ocr_model = "mistral-ocr-latest"
    
    def process_pdf_unified(self, pdf_path: str, output_dir: str = "output") -> dict:
        """
        Process PDF with text enhancement and extract tables from enhanced results.
        Automatically splits large files to handle API size limits.
        """
        if not PROCESSING_AVAILABLE:
            return {"error": "Enhancement libraries not available. Install: pip install opencv-python pillow PyMuPDF"}
        
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"🚀 Processing {pdf_file.name} with unified enhancement + table extraction...")
        
        # Check if we need to split the original PDF
        original_size_mb = pdf_file.stat().st_size / (1024 * 1024)
        print(f"📏 Original PDF size: {original_size_mb:.1f}MB")
        
        if original_size_mb > 400:  # Conservative threshold for original PDF
            print(f"🔄 Large PDF detected, will process in chunks...")
            return self._process_large_pdf_in_chunks(pdf_file, output_dir)
        else:
            return self._process_single_pdf(pdf_file, output_dir)
    
    def _process_single_pdf(self, pdf_file: Path, output_dir: str) -> dict:
        """Process a single PDF file."""
        # Step 1: Create enhanced PDF
        enhanced_pdf_path = self._create_enhanced_pdf(pdf_file)
        
        try:
            # Check enhanced PDF size
            enhanced_size_mb = enhanced_pdf_path.stat().st_size / (1024 * 1024)
            if enhanced_size_mb > 600:
                print(f"⚠️  Enhanced PDF ({enhanced_size_mb:.1f}MB) exceeds API limit, splitting...")
                enhanced_pdf_path.unlink()  # Clean up oversized file
                return self._process_large_pdf_in_chunks(pdf_file, output_dir)
            
            # Step 2: Run OCR on enhanced PDF
            print("📄 Running OCR on enhanced document...")
            ocr_response = self._run_ocr(enhanced_pdf_path)
            
            # Step 3: Extract tables from enhanced OCR results
            print("📊 Extracting tables from enhanced OCR data...")
            page_tables = self._extract_tables_from_enhanced_ocr(ocr_response)
            
            # Step 4: Save results
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save enhanced OCR results
            self._save_ocr_results(ocr_response, pdf_file.stem, output_path)
            
            # Save CSV tables
            csv_files = []
            if page_tables:
                csv_files = self._save_tables_to_csv(page_tables, pdf_file.stem, output_path)
            
            return {
                "pdf_name": pdf_file.name,
                "pages_processed": len(ocr_response.pages),
                "tables_found": sum(len(tables) for tables in page_tables.values()) if page_tables else 0,
                "pages_with_tables": list(page_tables.keys()) if page_tables else [],
                "csv_files": csv_files,
                "enhancement_applied": True,
                "output_dir": str(output_path),
                "chunks_processed": 1
            }
            
        finally:
            # Clean up enhanced PDF
            if enhanced_pdf_path.exists():
                enhanced_pdf_path.unlink()
                temp_dir = enhanced_pdf_path.parent
                if temp_dir.name.startswith("enhanced_"):
                    try:
                        temp_dir.rmdir()
                    except:
                        pass
    
    def _process_large_pdf_in_chunks(self, pdf_file: Path, output_dir: str) -> dict:
        """Process a large PDF by splitting it into manageable chunks."""
        print(f"📑 Splitting {pdf_file.name} into chunks for processing...")
        
        # Split the original PDF into chunks
        chunks = self._split_pdf_into_chunks(pdf_file)
        
        # Initialize combined results
        all_page_tables = {}
        all_csv_files = []
        total_pages_processed = 0
        total_tables_found = 0
        combined_ocr_data = {"pages": []}
        combined_markdown_content = []
        global_page_num = 1
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Process each chunk
        for chunk_idx, chunk_path in enumerate(chunks, 1):
            try:
                print(f"\n🔧 Processing chunk {chunk_idx}/{len(chunks)}: {chunk_path.name}")
                
                # Process this chunk
                chunk_result = self._process_single_pdf(chunk_path, output_dir)
                
                if "error" not in chunk_result:
                    # Read the chunk's OCR results
                    chunk_json_path = output_path / f"{chunk_path.stem}_enhanced_ocr.json"
                    chunk_md_path = output_path / f"{chunk_path.stem}_enhanced_ocr.md"
                    
                    if chunk_json_path.exists():
                        with open(chunk_json_path, 'r', encoding='utf-8') as f:
                            chunk_data = json.load(f)
                        
                        # Update page numbers to be global and collect data
                        for page_data in chunk_data.get('pages', []):
                            page_data['page_number'] = global_page_num
                            combined_ocr_data['pages'].append(page_data)
                            global_page_num += 1
                        
                        # Remove chunk-specific JSON file
                        chunk_json_path.unlink()
                    
                    # Collect markdown content
                    if chunk_md_path.exists():
                        with open(chunk_md_path, 'r', encoding='utf-8') as f:
                            chunk_md_content = f.read()
                            # Remove the header and add content to combined
                            lines = chunk_md_content.split('\n')
                            if lines and lines[0].startswith('# Enhanced OCR Results'):
                                lines = lines[2:]  # Skip header and empty line
                            combined_markdown_content.extend(lines)
                        
                        # Remove chunk-specific MD file
                        chunk_md_path.unlink()
                    
                    # Process CSV files with global page numbers
                    for csv_file in chunk_result.get('csv_files', []):
                        csv_path = Path(csv_file)
                        if csv_path.exists():
                            # Extract original page number from filename and convert to global
                            import re
                            match = re.search(r'_page_(\d+)\.csv$', csv_path.name)
                            if match:
                                chunk_page_num = int(match.group(1))
                                # Calculate global page number
                                chunk_start_page = total_pages_processed + 1
                                global_page_number = chunk_start_page + chunk_page_num - 1
                                
                                # Create new filename with global page number
                                new_name = f"{pdf_file.stem}_page_{global_page_number}.csv"
                                new_csv_path = csv_path.parent / new_name
                                csv_path.rename(new_csv_path)
                                all_csv_files.append(str(new_csv_path))
                    
                    total_pages_processed += chunk_result.get('pages_processed', 0)
                    total_tables_found += chunk_result.get('tables_found', 0)
                
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_idx}: {e}")
                continue
            
            finally:
                # Clean up chunk file
                if chunk_path.exists():
                    chunk_path.unlink()
        
        # Clean up chunk directory
        if chunks:
            chunk_dir = chunks[0].parent
            try:
                chunk_dir.rmdir()
            except:
                pass
        
        # Save combined OCR results
        if combined_ocr_data['pages']:
            combined_json_path = output_path / f"{pdf_file.stem}_enhanced_ocr.json"
            with open(combined_json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_ocr_data, f, indent=2, ensure_ascii=False)
            
            # Save combined markdown
            combined_md_path = output_path / f"{pdf_file.stem}_enhanced_ocr.md"
            with open(combined_md_path, 'w', encoding='utf-8') as f:
                f.write("# Enhanced OCR Results\n\n")
                # Renumber pages in markdown content
                content = '\n'.join(combined_markdown_content)
                # Update page headers to use global numbering
                import re
                page_num = 1
                def replace_page_header(match):
                    nonlocal page_num
                    result = f"## Page {page_num}"
                    page_num += 1
                    return result
                content = re.sub(r'^## Page \d+', replace_page_header, content, flags=re.MULTILINE)
                f.write(content)
            
            print(f"📁 Combined OCR results saved: {combined_json_path.name}, {combined_md_path.name}")
        
        # Calculate pages with tables for summary
        pages_with_tables = []
        for csv_file in all_csv_files:
            match = re.search(r'_page_(\d+)\.csv$', Path(csv_file).name)
            if match:
                pages_with_tables.append(int(match.group(1)))
        
        return {
            "pdf_name": pdf_file.name,
            "pages_processed": total_pages_processed,
            "tables_found": total_tables_found,
            "pages_with_tables": sorted(pages_with_tables),
            "csv_files": all_csv_files,
            "enhancement_applied": True,
            "output_dir": str(output_path),
            "chunks_processed": len(chunks),
            "processing_method": "chunked"
        }
    
    def _needs_selective_enhancement(self, page, page_num: int) -> bool:
        """Detect if a page needs selective enhancement based on simple heuristics."""
        
        # Simplified image-based detection for potential table pages
        try:
            # Use lower resolution for faster analysis
            mat = fitz.Matrix(1.5, 1.5)  
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            import io
            pil_img = Image.open(io.BytesIO(img_data))
            
            # Convert to grayscale for analysis
            gray_img = pil_img.convert('L')
            img_array = np.asarray(gray_img)
            
            # Basic edge detection for table structures
            edges = cv2.Canny(img_array, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Simple line detection
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            line_density = (np.count_nonzero(horizontal_lines) + np.count_nonzero(vertical_lines)) / edges.size
            
            # Check for financial document indicators
            try:
                text_content = page.get_text()
                has_financial_keywords = any(keyword in text_content.upper() for keyword in 
                    ['INVOICE NO', 'TYPE', 'REFERENCE', 'INVOICE DATE', 'DUE DATE', 
                     'DEBIT', 'CREDIT', 'STATEMENT NO', 'ACCOUNT NUMBER'])
                
                if has_financial_keywords:
                    print(f"     💰 Page {page_num} has financial indicators - applying enhancement")
                    return True
            except:
                pass
            
            # Conservative heuristics for table detection  
            print(f"     📊 Page {page_num} analysis: edge_density={edge_density:.4f}, line_density={line_density:.4f}")
            
            # Lowered thresholds for better detection of financial tables
            if edge_density > 0.035 and line_density > 0.004:  # Lowered thresholds
                print(f"     🔍 Page {page_num} flagged for selective enhancement")
                return True
            
            # Special case for pages with moderate structure but financial content
            if 0.035 < edge_density < 0.05 and line_density > 0.008:
                print(f"     📋 Page {page_num} has moderate table structure - checking content")
                return True
            else:
                print(f"     ✅ Page {page_num} will use standard enhancement only")
                return False
                
        except Exception as e:
            print(f"     ⚠️ Image analysis failed for page {page_num}: {e}")
            return False

    def _create_enhanced_pdf(self, pdf_path: Path) -> Path:
        """Create enhanced PDF with selective text darkening."""
        temp_dir = Path(tempfile.mkdtemp(prefix="enhanced_"))
        enhanced_pdf_path = temp_dir / f"{pdf_path.stem}_enhanced.pdf"
        
        print("🔧 Applying selective text enhancement...")
        
        pdf_doc = fitz.open(pdf_path)
        enhanced_pdf = fitz.open()
        
        # Start with high resolution, reduce if file gets too large
        matrix_scale = 4.0
        dpi = 400
        selective_pages = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            print(f"   Analyzing page {page_num + 1}...")
            
            # Check if this page needs selective enhancement
            needs_enhancement = self._needs_selective_enhancement(page, page_num + 1)
            if needs_enhancement:
                selective_pages.append(page_num + 1)
                print(f"     ✅ Page {page_num + 1} flagged for selective enhancement")
            
            # Adaptive resolution based on potential file size
            if page_num > 0 and page_num % 5 == 0:  # Check every 5 pages
                current_size_mb = enhanced_pdf_path.stat().st_size / (1024 * 1024) if enhanced_pdf_path.exists() else 0
                estimated_final_size = current_size_mb * (len(pdf_doc) / page_num)
                
                if estimated_final_size > 500:  # Reduce quality if approaching limit
                    matrix_scale = 2.0
                    dpi = 200
                    print(f"     🔄 Reducing resolution to avoid file size limit")
            
            # Apply selective enhancement
            if needs_enhancement:
                # Higher resolution for problematic pages
                enhancement_matrix_scale = 6.0
                enhancement_dpi = 600
                print(f"   🎯 Applying selective enhancement to page {page_num + 1}...")
            else:
                enhancement_matrix_scale = matrix_scale
                enhancement_dpi = dpi
            
            # Convert with appropriate resolution
            mat = fitz.Matrix(enhancement_matrix_scale, enhancement_matrix_scale)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Apply text darkening
            import io
            pil_img = Image.open(io.BytesIO(img_data))
            
            if needs_enhancement:
                enhanced_img = self._apply_aggressive_table_enhancement(pil_img)
            else:
                enhanced_img = self._apply_text_darkening(pil_img)
            
            # Save to PDF with appropriate DPI
            img_bytes = io.BytesIO()
            enhanced_img.save(img_bytes, format='PNG', dpi=(enhancement_dpi, enhancement_dpi))
            img_bytes.seek(0)
            
            img_rect = fitz.Rect(0, 0, enhanced_img.width, enhanced_img.height)
            new_page = enhanced_pdf.new_page(width=img_rect.width, height=img_rect.height)
            new_page.insert_image(img_rect, stream=img_bytes.getvalue())
        
        enhanced_pdf.save(enhanced_pdf_path)
        enhanced_pdf.close()
        pdf_doc.close()
        
        # Check final file size
        final_size_mb = enhanced_pdf_path.stat().st_size / (1024 * 1024)
        print(f"✅ Enhanced PDF created ({final_size_mb:.1f}MB)")
        
        if selective_pages:
            print(f"🎯 Selective enhancement applied to pages: {selective_pages}")
        
        if final_size_mb > 600:
            print(f"⚠️  Enhanced PDF is {final_size_mb:.1f}MB (exceeds 600MB limit)")
        
        return enhanced_pdf_path
    
    def _apply_aggressive_table_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply conservative enhancement specifically for table text recognition."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print("     🎯 Applying conservative table enhancement...")
        
        # Stage 1: Conservative contrast and brightness adjustment
        enhancer = ImageEnhance.Contrast(image)
        moderate_contrast = enhancer.enhance(1.8)  # Reduced from 4.0 to 1.8
        
        brightness_enhancer = ImageEnhance.Brightness(moderate_contrast)
        adjusted = brightness_enhancer.enhance(0.85)  # Increased from 0.4 to 0.85
        
        sharpener = ImageEnhance.Sharpness(adjusted)
        sharp = sharpener.enhance(1.5)  # Reduced from 3.5 to 1.5
        
        # Stage 2: Conservative OpenCV processing
        cv_image = cv2.cvtColor(np.array(sharp, copy=None), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Stage 3: Simple noise reduction without over-processing
        # Use gentle Gaussian blur instead of bilateral filter
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Stage 4: Conservative thresholding
        # Use only adaptive thresholding with larger block size for stability
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        
        # Stage 5: Ensure text is black on white background
        if np.mean(adaptive_thresh) > 127:
            adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
        
        # Stage 6: Minimal morphological operations
        # Only apply very light cleaning operations
        clean_kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, clean_kernel, iterations=1)
        
        # Convert back to PIL
        final_image = Image.fromarray(cleaned, mode='L')
        return final_image.convert('RGB')

    def _apply_text_darkening(self, image: Image.Image) -> Image.Image:
        """Apply standard text darkening techniques."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Standard enhancement pipeline
        enhancer = ImageEnhance.Contrast(image)
        high_contrast = enhancer.enhance(2.5)
        
        brightness_enhancer = ImageEnhance.Brightness(high_contrast)
        darker = brightness_enhancer.enhance(0.7)
        
        sharpener = ImageEnhance.Sharpness(darker)
        sharper = sharpener.enhance(2.0)
        
        # OpenCV processing
        cv_image = cv2.cvtColor(np.array(sharper, copy=None), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Morphological operations
        kernel = np.ones((2,2), np.uint8)
        thick_text = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            thick_text, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Ensure text is black
        if np.mean(adaptive) > 127:
            adaptive = cv2.bitwise_not(adaptive)
        
        # Make text bolder
        dilate_kernel = np.ones((1,1), np.uint8)
        bold_text = cv2.dilate(adaptive, dilate_kernel, iterations=1)
        
        # Convert back to PIL
        final_image = Image.fromarray(bold_text, mode='L')
        return final_image.convert('RGB')
    
    def _check_file_size(self, pdf_path: Path) -> bool:
        """Check if file size is within API limits (600MB)."""
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        return file_size_mb <= 600
    
    def _split_pdf_into_chunks(self, pdf_path: Path) -> list[Path]:
        """Split PDF into manageable chunks for processing."""
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        
        # Calculate optimal chunk size based on file size
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 800:
            pages_per_chunk = max(1, total_pages // 6)  # 6 chunks for very large files
        elif file_size_mb > 400:
            pages_per_chunk = max(1, total_pages // 4)  # 4 chunks for large files
        else:
            pages_per_chunk = max(1, total_pages // 2)  # 2 chunks for medium files
        
        split_files = []
        temp_dir = pdf_path.parent / f"chunks_{pdf_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        
        for i in range(0, total_pages, pages_per_chunk):
            end_page = min(i + pages_per_chunk - 1, total_pages - 1)
            
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(pdf_doc, from_page=i, to_page=end_page)
            
            chunk_path = temp_dir / f"{pdf_path.stem}_chunk_{len(split_files) + 1}.pdf"
            chunk_doc.save(chunk_path)
            chunk_doc.close()
            
            split_files.append(chunk_path)
            print(f"   📄 Created chunk {len(split_files)}: pages {i+1}-{end_page+1} ({chunk_path.stat().st_size / (1024*1024):.1f}MB)")
        
        pdf_doc.close()
        return split_files
    
    def _get_pages_per_chunk(self, pdf_path: Path) -> int:
        """Get the number of pages per chunk for a given PDF."""
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        pdf_doc.close()
        
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 800:
            return max(1, total_pages // 6)
        elif file_size_mb > 400:
            return max(1, total_pages // 4)
        else:
            return max(1, total_pages // 2)
    
    
    def _split_pdf_if_needed(self, pdf_path: Path) -> list[Path]:
        """Legacy method - replaced by _split_pdf_into_chunks."""
        return self._split_pdf_into_chunks(pdf_path)
    
    def _run_ocr(self, pdf_path: Path):
        """Run OCR on enhanced PDF, handling file size limits."""
        try:
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": pdf_path.stem,
                    "content": pdf_path.read_bytes(),
                },
                purpose="ocr",
            )
            
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            
            response = self.client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model=self.ocr_model,
                include_image_base64=True
            )
            
            return response
            
        except Exception as e:
            if "too large" in str(e).lower() or "600" in str(e):
                print(f"❌ File size error: {e}")
                raise Exception("File too large for API (600MB limit). Consider reducing image resolution or splitting the PDF manually.")
            else:
                raise e
    
    def _extract_tables_from_enhanced_ocr(self, ocr_response) -> dict:
        """Extract tables from enhanced OCR results with enhanced parsing for complex structures."""
        page_tables = {}
        
        for page_num, page in enumerate(ocr_response.pages, 1):
            print(f"   Processing page {page_num}...")
            
            # Try multiple extraction methods
            all_tables = []
            
            # Method 1: Direct markdown table extraction
            markdown_tables = self._extract_markdown_tables(page.markdown)
            if markdown_tables:
                all_tables.extend(markdown_tables)
            
            # Method 2: Enhanced table reconstruction for financial documents
            reconstructed_tables = self._reconstruct_financial_tables(page.markdown)
            if reconstructed_tables:
                all_tables.extend(reconstructed_tables)
            
            if all_tables:
                # Check if tables have meaningful content
                meaningful_tables = []
                for table in all_tables:
                    has_content = any(
                        any(cell.strip() and cell.strip() not in ['---', '2022 General Requirements Invoice Backup'] 
                            for cell in row) 
                        for row in table
                    )
                    if has_content or len(table) > 1:  # Include if has content OR is multi-row
                        meaningful_tables.append(table)
                
                if meaningful_tables:
                    page_tables[page_num] = meaningful_tables
                    print(f"     Found {len(meaningful_tables)} table(s)")
                else:
                    print(f"     No meaningful tables found")
            else:
                print(f"     No tables detected")
        
        return page_tables
    
    def _reconstruct_financial_tables(self, page_text: str):
        """Reconstruct financial tables from fragmented OCR output."""
        tables = []
        lines = page_text.split('\n')
        
        # Look for financial statement patterns
        for i, line in enumerate(lines):
            # Look for invoice/transaction table headers
            if 'Invoice No' in line and 'Type' in line and 'Reference' in line:
                print(f"     🔍 Found invoice table pattern at line {i}")
                table = self._extract_invoice_table_from_text(lines, i)
                if table and len(table) > 1:  # Must have header + data
                    tables.append(table)
                    print(f"     ✅ Reconstructed invoice table with {len(table)} rows")
        
        # Additional pattern: Look for scattered data that could form tables
        # This handles cases where table structure is lost but data elements are present
        scattered_data_table = self._extract_scattered_data_patterns(lines)
        if scattered_data_table and len(scattered_data_table) > 1:
            tables.append(scattered_data_table)
            print(f"     ✅ Reconstructed table from scattered data with {len(scattered_data_table)} rows")
        
        return tables
    
    def _extract_invoice_table_from_text(self, lines, start_idx):
        """Extract invoice table from text lines starting at given index."""
        import re
        
        # Define expected column headers
        expected_headers = ['Invoice No', 'Type', 'Reference', 'Invoice Date', 'Due Date', 'Debit', 'Credit']
        
        # Find the header line and parse it
        header_line = lines[start_idx]
        print(f"     📋 Header line: {header_line}")
        
        # Create header row
        table = [expected_headers]
        
        # Look for data lines after the header
        data_lines = []
        i = start_idx + 1
        
        # Collect potential data lines
        while i < len(lines) and i < start_idx + 50:  # Look ahead up to 50 lines
            line = lines[i].strip()
            
            # Stop at next section or table
            if line.startswith('#') or line.startswith('|') or 'STRIVING TO EARN' in line:
                break
                
            # Look for data patterns
            if line and any(char.isdigit() for char in line):
                # Check if this looks like structured data
                if self._looks_like_data_line(line):
                    data_lines.append(line)
            
            i += 1
        
        print(f"     📝 Found {len(data_lines)} potential data lines")
        
        # Now try to parse the data lines into structured rows
        # Look for patterns like: "220426 I *PO#: Tom9172024"
        invoice_data = {}
        
        # First pass: collect all invoice-related data
        for line in data_lines:
            # Split by common separators and look for patterns
            parts = line.replace('*PO#:', '').replace('*CK#:', '').split()
            
            if len(parts) >= 2:
                # Try to identify invoice number (first numeric part)
                invoice_num = None
                for part in parts:
                    if part.isdigit() and len(part) >= 5:  # Invoice numbers are typically 5+ digits
                        invoice_num = part
                        break
                
                if invoice_num:
                    if invoice_num not in invoice_data:
                        invoice_data[invoice_num] = {
                            'Invoice No': invoice_num,
                            'Type': '',
                            'Reference': '',
                            'Invoice Date': '',
                            'Due Date': '',
                            'Debit': '',
                            'Credit': ''
                        }
                    
                    # Look for type indicators
                    if ' I ' in line or line.strip().endswith(' I'):
                        invoice_data[invoice_num]['Type'] = 'I'
                    elif ' P ' in line or line.strip().endswith(' P'):
                        invoice_data[invoice_num]['Type'] = 'P'
                    elif ' D ' in line or line.strip().endswith(' D'):
                        invoice_data[invoice_num]['Type'] = 'D'
                    
                    # Extract reference (PO# numbers, etc.)
                    po_match = re.search(r'(\w+\d+)', line)
                    if po_match:
                        invoice_data[invoice_num]['Reference'] = po_match.group(1)
        
        # Second pass: look for dates and amounts in nearby table structures
        # Check for date patterns in markdown tables
        for i, line in enumerate(lines[start_idx:start_idx + 50]):
            if '|' in line and any(char.isdigit() for char in line):
                # Parse potential date/amount table
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(cells) >= 2:
                    # Look for date patterns (MM/DD/YYYY)
                    date_pattern = re.compile(r'\d{2}/\d{2}/\d{4}')
                    amount_pattern = re.compile(r'[\d,]+\.\d{2}')
                    
                    dates_in_line = date_pattern.findall(line)
                    amounts_in_line = amount_pattern.findall(line)
                    
                    if len(dates_in_line) >= 2:  # Invoice Date and Due Date
                        # Try to match with invoice data
                        invoice_nums = list(invoice_data.keys())
                        if len(invoice_nums) > 0:
                            # Assign dates to invoices in order
                            for j, inv_num in enumerate(invoice_nums):
                                if j < len(dates_in_line) // 2:
                                    invoice_data[inv_num]['Invoice Date'] = dates_in_line[j * 2]
                                    if j * 2 + 1 < len(dates_in_line):
                                        invoice_data[inv_num]['Due Date'] = dates_in_line[j * 2 + 1]
                    
                    if amounts_in_line:
                        # Try to assign amounts to invoices
                        invoice_nums = list(invoice_data.keys())
                        for j, amount in enumerate(amounts_in_line):
                            if j < len(invoice_nums):
                                # Determine if it's debit or credit based on context
                                invoice_data[invoice_nums[j]]['Debit'] = amount
        
        # Convert invoice_data to table rows
        for inv_num in sorted(invoice_data.keys()):
            row_data = invoice_data[inv_num]
            table.append([
                row_data['Invoice No'],
                row_data['Type'],
                row_data['Reference'],
                row_data['Invoice Date'],
                row_data['Due Date'],
                row_data['Debit'],
                row_data['Credit']
            ])
        
        return table if len(table) > 1 else None
    
    def _looks_like_data_line(self, line):
        """Check if a line looks like structured data rather than headers or narrative text."""
        import re
        
        line = line.strip()
        if not line:
            return False
        
        # Count different types of content
        digit_count = sum(c.isdigit() for c in line)
        alpha_count = sum(c.isalpha() for c in line)
        special_char_count = sum(c in '.,#*:/-()$' for c in line)
        
        # Calculate ratios
        total_chars = len(line)
        if total_chars == 0:
            return False
            
        digit_ratio = digit_count / total_chars
        special_ratio = special_char_count / total_chars
        
        # Data lines typically have:
        # - Higher ratio of digits (dates, amounts, IDs)
        # - Presence of special characters (formatting, references)
        # - Not purely alphabetic (which suggests narrative text)
        
        has_digits = digit_ratio > 0.1  # At least 10% digits
        has_structure = special_ratio > 0.05  # At least 5% special chars
        not_pure_text = digit_ratio > 0 or special_ratio > 0.15
        
        # Additional checks for data patterns
        has_date_pattern = bool(re.search(r'\d+/\d+/\d+', line))
        has_amount_pattern = bool(re.search(r'\d+\.\d{2}', line))
        has_id_pattern = bool(re.search(r'\b\d{3,}\b', line))
        
        # Line is likely data if it has structural elements
        return (has_digits and has_structure and not_pure_text) or \
               has_date_pattern or has_amount_pattern or has_id_pattern
    
    def _extract_scattered_data_patterns(self, lines):
        """Extract data from lines where table structure is completely lost."""
        import re
        
        # Generic patterns for common data types
        data_patterns = {
            'numbers': r'\b\d{3,}\b',                    # Numbers 3+ digits (IDs, amounts without decimals)
            'amounts': r'\b\d{1,3}(?:,\d{3})*\.\d{2}\b', # Monetary amounts
            'negative_amounts': r'-\d{1,3}(?:,\d{3})*\.\d{2}\b',  # Negative amounts
            'dates': r'\b\d{1,2}/\d{1,2}/\d{4}\b',       # Dates
            'references': r'\*[A-Z]+#:?\s*[\w\d#\-]+',   # Reference patterns
            'codes': r'\b[A-Z]{1,3}\d{2,6}[A-Z]?\b',     # Alphanumeric codes
        }
        
        # Collect all data elements with their line positions
        all_data_elements = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('**'):
                continue
                
            # Extract all types of data from this line
            line_data = {'line_num': i, 'original_text': line, 'elements': []}
            
            for pattern_name, pattern in data_patterns.items():
                matches = re.findall(pattern, line)
                for match in matches:
                    line_data['elements'].append({
                        'type': pattern_name,
                        'value': match,
                        'position': line.find(match)
                    })
            
            if line_data['elements']:
                all_data_elements.append(line_data)
        
        if len(all_data_elements) < 2:
            return None
        
        # Try to identify potential table structure from scattered data
        return self._reconstruct_table_from_scattered_data(all_data_elements)
    
    def _reconstruct_table_from_scattered_data(self, data_elements):
        """Attempt to reconstruct a table from scattered data elements."""
        
        # Group data elements that appear to be related (close line numbers)
        grouped_data = []
        current_group = []
        
        for i, data_line in enumerate(data_elements):
            if not current_group:
                current_group = [data_line]
            else:
                # If lines are close together (within 2 lines), group them
                if data_line['line_num'] - current_group[-1]['line_num'] <= 2:
                    current_group.append(data_line)
                else:
                    if len(current_group) >= 2:  # Only keep groups with multiple elements
                        grouped_data.append(current_group)
                    current_group = [data_line]
        
        # Don't forget the last group
        if len(current_group) >= 2:
            grouped_data.append(current_group)
        
        if len(grouped_data) < 2:
            return None
        
        # Analyze the structure to determine columns
        all_element_types = set()
        for group in grouped_data:
            for data_line in group:
                for element in data_line['elements']:
                    all_element_types.add(element['type'])
        
        # Create dynamic column headers based on found data types
        column_headers = []
        type_mapping = {
            'numbers': 'Number',
            'amounts': 'Amount',
            'negative_amounts': 'Credit',
            'dates': 'Date',
            'references': 'Reference',
            'codes': 'Code'
        }
        
        for data_type in sorted(all_element_types):
            column_headers.append(type_mapping.get(data_type, data_type.title()))
        
        # If we have both amounts and negative_amounts, reorganize as Debit/Credit
        if 'amounts' in all_element_types and 'negative_amounts' in all_element_types:
            column_headers = [h for h in column_headers if h not in ['Amount', 'Credit']]
            column_headers.extend(['Debit', 'Credit'])
        
        # Build the table
        table = [column_headers]
        
        for group in grouped_data:
            row = [''] * len(column_headers)
            
            # Collect all elements from this group
            group_elements = []
            for data_line in group:
                group_elements.extend(data_line['elements'])
            
            # Fill in the row based on element types
            for element in group_elements:
                element_type = element['type']
                element_value = element['value']
                
                # Map element to appropriate column
                if element_type == 'numbers':
                    col_idx = self._find_column_index(column_headers, 'Number')
                elif element_type == 'amounts':
                    if 'Debit' in column_headers:
                        col_idx = self._find_column_index(column_headers, 'Debit')
                    else:
                        col_idx = self._find_column_index(column_headers, 'Amount')
                elif element_type == 'negative_amounts':
                    col_idx = self._find_column_index(column_headers, 'Credit')
                    # Ensure negative sign is preserved
                    if not element_value.startswith('-'):
                        element_value = '-' + element_value
                elif element_type == 'dates':
                    col_idx = self._find_column_index(column_headers, 'Date')
                elif element_type == 'references':
                    col_idx = self._find_column_index(column_headers, 'Reference')
                elif element_type == 'codes':
                    col_idx = self._find_column_index(column_headers, 'Code')
                else:
                    continue
                
                if col_idx is not None and col_idx < len(row):
                    # If cell is empty, fill it; if not, append with separator
                    if row[col_idx] == '':
                        row[col_idx] = element_value
                    else:
                        row[col_idx] += ' | ' + element_value
            
            # Only add row if it has meaningful data
            if any(cell.strip() for cell in row):
                table.append(row)
        
        return table if len(table) > 1 else None
    
    def _find_column_index(self, headers, target_name):
        """Find the index of a column header, case-insensitive."""
        for i, header in enumerate(headers):
            if header.lower() == target_name.lower():
                return i
        return None
    
    def _extract_markdown_tables(self, page_text: str):
        """Extract tables from markdown format."""
        tables = []
        lines = page_text.split('\n')
        current_table = []
        in_table = False
        
        for line in lines:
            if '|' in line and line.count('|') >= 2:
                # Skip separator lines
                if not line.strip().replace('|', '').replace('-', '').replace(' ', ''):
                    continue
                
                # Extract cells
                cells = [cell.strip() for cell in line.split('|')]
                if cells and cells[0] == '':
                    cells = cells[1:]
                if cells and cells[-1] == '':
                    cells = cells[:-1]
                
                if cells:
                    if not in_table:
                        current_table = []
                        in_table = True
                    current_table.append(cells)
            else:
                if in_table and current_table:
                    tables.append(current_table)
                    current_table = []
                    in_table = False
        
        # Don't forget last table
        if in_table and current_table:
            tables.append(current_table)
        
        return tables
    
    def _save_tables_to_csv(self, page_tables: dict, pdf_name: str, output_path: Path):
        """Save tables to CSV files - one CSV per page with all tables concatenated."""
        csv_files = []
        
        for page_num, tables in page_tables.items():
            csv_filename = f"{pdf_name}_page_{page_num}.csv"
            csv_path = output_path / csv_filename
            
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Concatenate all tables from this page
                    for table_idx, table_data in enumerate(tables):
                        # Add a separator comment between tables if there are multiple tables
                        if table_idx > 0:
                            # Add empty row as separator
                            writer.writerow([])
                            # Add table identifier comment
                            writer.writerow([f"--- Table {table_idx + 1} ---"])
                        
                        # Write table data
                        writer.writerows(table_data)
                
                csv_files.append(str(csv_path))
                print(f"     Saved: {csv_filename} (concatenated {len(tables)} table(s))")
                
            except Exception as e:
                print(f"     Error saving {csv_filename}: {e}")
        
        return csv_files
    
    def _save_ocr_results(self, response, pdf_name: str, output_path: Path):
        """Save OCR results."""
        # JSON
        json_path = output_path / f"{pdf_name}_enhanced_ocr.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json.loads(response.model_dump_json()), f, indent=2, ensure_ascii=False)
        
        # Markdown
        markdown_path = output_path / f"{pdf_name}_enhanced_ocr.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# Enhanced OCR Results\n\n")
            for i, page in enumerate(response.pages):
                f.write(f"## Page {i + 1}\n\n")
                f.write(page.markdown)
                f.write("\n\n---\n\n")
        
        print(f"📁 OCR results saved: {json_path.name}, {markdown_path.name}")

def main():
    # Load environment variables from .env file
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    parser = argparse.ArgumentParser(description="Unified Enhanced PDF Processor")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output-dir", default="unified_enhanced_results", help="Output directory")
    parser.add_argument("--api-key", help="Mistral API key (can also be set via MISTRAL_API_KEY environment variable)")
    
    args = parser.parse_args()
    
    # Get API key from command line argument or environment variable
    api_key = args.api_key or os.getenv('MISTRAL_API_KEY')
    
    if not api_key:
        print("❌ Error: Mistral API key is required!")
        print("   Set it via:")
        print("   1. Command line: --api-key YOUR_KEY")
        print("   2. Environment variable: MISTRAL_API_KEY=YOUR_KEY")
        print("   3. .env file: MISTRAL_API_KEY=YOUR_KEY")
        return 1
    
    print("🚀 Unified Enhanced PDF Processor")
    print("=" * 50)
    print("Features: Text darkening + Direct table extraction from enhanced OCR")
    print()
    
    try:
        processor = UnifiedEnhancedExtractor(api_key=api_key)
        result = processor.process_pdf_unified(args.pdf, args.output_dir)
        
        print(f"\n✅ PROCESSING COMPLETE")
        print("=" * 30)
        
        if "error" not in result:
            print(f"PDF: {result['pdf_name']}")
            print(f"Pages: {result['pages_processed']}")
            print(f"Tables found: {result['tables_found']}")
            
            # Show processing method
            if result.get('chunks_processed', 1) > 1:
                print(f"Processing method: Chunked processing (transparent to user)")
            else:
                print(f"Processing method: Single file")
            
            if result.get('pages_with_tables'):
                print(f"Pages with tables: {result['pages_with_tables']}")
            
            print(f"CSV files: {len(result['csv_files'])}")
            
            # Check if tables were found
            if result.get('csv_files'):
                print("🎯 SUCCESS: Tables extracted to CSV files!")
            else:
                print("⚠️  No tables extracted to CSV")
                
            for csv_file in result['csv_files']:
                print(f"  📄 {Path(csv_file).name}")
        else:
            print(f"❌ Error: {result['error']}")
        
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())