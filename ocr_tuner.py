#!/usr/bin/env python3
"""
OCR Enhancement Parameter Tuner

Tunes the enhancement parameters in unified_enhanced_extractor.py to minimize 
OCR error rates on the FC-AMF-OCR dataset using grid search.
"""

import os
import json
import tempfile
import logging
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from itertools import product
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
from sklearn.model_selection import KFold

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import cv2
    from PIL import Image, ImageEnhance
    import fitz  # PyMuPDF
    PROCESSING_AVAILABLE = True
except ImportError:
    PROCESSING_AVAILABLE = False

try:
    from mistralai import Mistral, DocumentURLChunk
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

# Levenshtein distance for text comparison
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

@dataclass
class EnhancementParameters:
    """Parameters for image enhancement in OCR preprocessing"""
    # Basic enhancement parameters (table-specific)
    table_contrast_factor: float = 2.8
    table_brightness_factor: float = 0.65
    table_sharpness_factor: float = 2.2
    
    # OpenCV parameters
    bilateral_d: int = 11
    bilateral_sigma_color: float = 85.0
    bilateral_sigma_space: float = 85.0
    
    # Adaptive threshold parameters
    adaptive_max_value: int = 255
    adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    adaptive_thresh_type: int = cv2.THRESH_BINARY
    adaptive_block_size: int = 13
    adaptive_c: float = 1.5
    
    # Morphological operations (table-specific)
    morph_kernel_size: Tuple[int, int] = (2, 2)
    table_morph_close_iterations: int = 1
    table_morph_dilate_iterations: int = 1
    
    # Resolution/DPI settings (table-specific)
    matrix_scale: float = 4.5
    enhancement_dpi: int = 550

@dataclass
class OCRResult:
    """Results from OCR evaluation"""
    character_error_rate: float
    word_error_rate: float
    confidence_score: float
    processing_time: float
    parameters: EnhancementParameters

@dataclass
class CVResult:
    """Results from cross-validation"""
    fold_results: List[OCRResult]
    mean_cer: float
    std_cer: float
    mean_wer: float
    std_wer: float
    mean_confidence: float
    std_confidence: float
    parameters: EnhancementParameters
    fold_count: int

class OCRTuner:
    def __init__(self, api_key: str, use_cv: bool = False, n_folds: int = 5):
        """Initialize the OCR tuner with Mistral API key"""
        if not DATASETS_AVAILABLE:
            raise ImportError("Install datasets: pip install datasets")
        if not PROCESSING_AVAILABLE:
            raise ImportError("Install processing libraries: pip install opencv-python pillow PyMuPDF")
        if not MISTRAL_AVAILABLE:
            raise ImportError("Install mistralai: pip install mistralai")
        
        self.client = Mistral(api_key=api_key)
        self.ocr_model = "mistral-ocr-latest"
        self.logger = self._setup_logging()
        
        # Cross-validation settings
        self.use_cv = use_cv
        self.n_folds = n_folds
        
        # Load table datasets
        self.dataset = None
        self.dataset_samples = []
        self._load_dataset()
        
        # Track experiment results
        self.results = []
        self.cv_results = []
        self.best_result = None
        self.best_cv_result = None
        
        # Load existing optimized parameters as baseline
        self._load_optimized_parameters()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the tuning process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ocr_tuning.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_dataset(self):
        """Load table-focused datasets for OCR tuning"""
        try:
            from table_dataset_loader import TableDatasetLoader
            self.logger.info("Loading table-focused datasets...")
            
            self.table_loader = TableDatasetLoader()
            
            if self.use_cv:
                # Pre-load samples for CV splitting
                self._preload_dataset_samples()
            else:
                # Use mixed dataset iterator for simple evaluation
                self.dataset_iter = self.table_loader.load_mixed_dataset(
                    pubtables_samples=10000,  # PubTables-1M samples (increased from 1000)
                    pubtabnet_samples=10000,  # PubTabNet 2.0 samples (increased from 1000)
                    streaming=True
                )
            
            self.logger.info("Table datasets loaded successfully")
            
        except ImportError:
            # Fallback to original FC-AMF-OCR if table loader not available
            self.logger.warning("Table dataset loader not available, falling back to FC-AMF-OCR")
            self.dataset = load_dataset('lightonai/fc-amf-ocr', streaming=True)
            if self.use_cv:
                self.logger.error("Cross-validation requires table dataset loader")
                raise ImportError("Cross-validation requires table_dataset_loader.py")
            else:
                self.dataset_iter = iter(self.dataset['train'])
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            raise
    
    def _load_optimized_parameters(self):
        """Load existing optimized parameters as baseline"""
        try:
            from optimized_enhancement_parameters import OPTIMIZED_PARAMETERS
            self.logger.info("Loaded existing optimized parameters as baseline")
            self.baseline_params = OPTIMIZED_PARAMETERS
        except ImportError:
            self.logger.warning("optimized_enhancement_parameters.py not found, using defaults")
            self.baseline_params = {}
    
    def _preload_dataset_samples(self, max_samples: int = 5000):
        """Pre-load dataset samples for cross-validation splitting"""
        self.logger.info(f"Pre-loading dataset samples for CV (max {max_samples})...")
        
        dataset_iter = self.table_loader.load_mixed_dataset(
            pubtables_samples=max_samples//2,  # Will be 2500 each for CV
            pubtabnet_samples=max_samples//2,  # Will be 2500 each for CV
            streaming=True
        )
        
        count = 0
        for sample in dataset_iter:
            try:
                # Validate sample has required data
                if (hasattr(sample, 'ground_truth_text') and 
                    sample.ground_truth_text and 
                    sample.ground_truth_text.strip() and
                    hasattr(sample, 'image') and 
                    sample.image is not None):
                    
                    self.dataset_samples.append(sample)
                    count += 1
                    
                    if count >= max_samples:
                        break
                    
                    if count % 100 == 0:
                        self.logger.info(f"Pre-loaded {count} valid samples")
                        
            except Exception as e:
                self.logger.warning(f"Error validating sample {count}: {e}")
                continue
        
        self.logger.info(f"Pre-loaded {len(self.dataset_samples)} valid samples for CV")
        
        if len(self.dataset_samples) < self.n_folds * 2:
            raise ValueError(f"Insufficient samples ({len(self.dataset_samples)}) for {self.n_folds}-fold CV")
    
    def generate_parameter_combinations(self, 
                                       num_samples: int = 50,
                                       focused_search: bool = True) -> List[EnhancementParameters]:
        """Generate parameter combinations for grid search"""
        
        if focused_search:
            # Focused search around table-optimized parameters
            base_contrast = self.baseline_params.get('table_contrast_factor', 2.8)
            base_brightness = self.baseline_params.get('table_brightness_factor', 0.65)
            base_sharpness = self.baseline_params.get('table_sharpness_factor', 2.2)
            
            param_ranges = {
                'table_contrast_factor': np.linspace(base_contrast - 0.5, base_contrast + 0.5, 5),
                'table_brightness_factor': np.linspace(base_brightness - 0.15, base_brightness + 0.15, 5),
                'table_sharpness_factor': np.linspace(base_sharpness - 0.3, base_sharpness + 0.3, 5),
                'bilateral_d': [9, 11, 13],
                'bilateral_sigma_color': [75, 85, 95],
                'bilateral_sigma_space': [75, 85, 95],
                'adaptive_block_size': [11, 13, 15],
                'adaptive_c': np.linspace(1.0, 2.0, 4),
                'morph_kernel_size': [(1,1), (2,2), (3,3)],
                'table_morph_close_iterations': [1, 2],
                'table_morph_dilate_iterations': [1, 2],
                'matrix_scale': [4.0, 4.5, 5.0, 5.5],
                'enhancement_dpi': [500, 550, 600]
            }
        else:
            # Broader search for table parameters exploration
            param_ranges = {
                'table_contrast_factor': np.linspace(2.0, 3.5, 8),
                'table_brightness_factor': np.linspace(0.5, 0.8, 8),
                'table_sharpness_factor': np.linspace(1.5, 3.0, 8),
                'bilateral_d': [7, 9, 11, 13, 15],
                'bilateral_sigma_color': [50, 75, 85, 100, 125],
                'bilateral_sigma_space': [50, 75, 85, 100, 125],
                'adaptive_block_size': [9, 11, 13, 15, 17],
                'adaptive_c': np.linspace(0.8, 2.5, 6),
                'morph_kernel_size': [(1,1), (2,2), (3,3), (4,4)],
                'table_morph_close_iterations': [1, 2, 3],
                'table_morph_dilate_iterations': [1, 2, 3],
                'matrix_scale': [3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                'enhancement_dpi': [400, 500, 550, 600, 650]
            }
        
        # Generate combinations using smart sampling
        total_combinations = 1
        for param_values in param_ranges.values():
            total_combinations *= len(param_values)
        
        if num_samples <= total_combinations and total_combinations <= 1000:
            # Full grid search if feasible
            combinations = list(product(*param_ranges.values()))
            # Randomly sample if too many combinations
            if len(combinations) > num_samples:
                np.random.seed(42)  # For reproducibility
                indices = np.random.choice(len(combinations), num_samples, replace=False)
                combinations = [combinations[i] for i in indices]
        else:
            # Random sampling for large parameter spaces
            combinations = []
            np.random.seed(42)
            
            for _ in range(num_samples):
                params = {}
                for param_name, param_values in param_ranges.items():
                    if isinstance(param_values[0], tuple):
                        # Handle tuple parameters like morph_kernel_size
                        params[param_name] = param_values[np.random.randint(len(param_values))]
                    else:
                        # Handle scalar parameters
                        params[param_name] = np.random.choice(param_values)
                combinations.append(tuple(params.values()))
        
        # Convert to EnhancementParameters objects
        param_objects = []
        param_names = list(param_ranges.keys())
        
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            params = EnhancementParameters(**param_dict)
            param_objects.append(params)
        
        self.logger.info(f"Generated {len(param_objects)} parameter combinations")
        return param_objects
    
    def evaluate_parameters(self, 
                          parameters: EnhancementParameters,
                          num_samples: int = 10) -> OCRResult:
        """Evaluate a specific set of enhancement parameters"""
        
        self.logger.info(f"Evaluating parameters: contrast={parameters.contrast_factor:.2f}, "
                        f"brightness={parameters.brightness_factor:.2f}, "
                        f"sharpness={parameters.sharpness_factor:.2f}")
        
        total_cer = 0.0
        total_wer = 0.0
        total_confidence = 0.0
        total_time = 0.0
        successful_evaluations = 0
        
        # Process samples from the table dataset
        successful_samples = 0
        
        for i, table_sample in enumerate(self.dataset_iter):
            if successful_samples >= num_samples:
                break
            
            try:
                # Process table sample
                cer, wer, confidence, processing_time = self._evaluate_table_sample(
                    table_sample, parameters
                )
                
                if cer is not None:  # Successful evaluation
                    total_cer += cer
                    total_wer += wer
                    total_confidence += confidence
                    total_time += processing_time
                    successful_evaluations += 1
                    successful_samples += 1
                    
                    self.logger.debug(f"Sample {successful_samples}: CER={cer:.3f}, WER={wer:.3f}, "
                                    f"Confidence={confidence:.3f} ({table_sample.source_dataset})")
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate sample {i+1}: {e}")
                continue
        
        if successful_evaluations == 0:
            self.logger.error("No successful evaluations!")
            return OCRResult(
                character_error_rate=1.0,
                word_error_rate=1.0, 
                confidence_score=0.0,
                processing_time=float('inf'),
                parameters=parameters
            )
        
        # Calculate averages
        avg_cer = total_cer / successful_evaluations
        avg_wer = total_wer / successful_evaluations
        avg_confidence = total_confidence / successful_evaluations
        avg_time = total_time / successful_evaluations
        
        result = OCRResult(
            character_error_rate=avg_cer,
            word_error_rate=avg_wer,
            confidence_score=avg_confidence,
            processing_time=avg_time,
            parameters=parameters
        )
        
        self.logger.info(f"Average results: CER={avg_cer:.3f}, WER={avg_wer:.3f}, "
                        f"Confidence={avg_confidence:.3f}, Time={avg_time:.2f}s")
        
        return result
    
    def _evaluate_table_sample(self, 
                              table_sample,
                              parameters: EnhancementParameters) -> Tuple[Optional[float], Optional[float], float, float]:
        """Evaluate OCR performance on a table sample"""
        
        start_time = time.time()
        
        try:
            # Get ground truth text
            ground_truth = table_sample.ground_truth_text
            if not ground_truth or not ground_truth.strip():
                return None, None, 0.0, 0.0
            
            # Apply enhancement to image and convert to PDF
            enhanced_pdf_bytes = self._apply_table_image_enhancement(table_sample.image, parameters)
            if not enhanced_pdf_bytes:
                return None, None, 0.0, 0.0
            
            # Run OCR
            ocr_result = self._run_ocr_on_bytes(enhanced_pdf_bytes)
            if not ocr_result:
                return None, None, 0.0, 0.0
            
            # Extract text and confidence
            extracted_text, avg_confidence = self._extract_text_and_confidence(ocr_result)
            
            # Calculate error rates
            cer = self._calculate_character_error_rate(ground_truth, extracted_text)
            wer = self._calculate_word_error_rate(ground_truth, extracted_text)
            
            processing_time = time.time() - start_time
            
            return cer, wer, avg_confidence, processing_time
            
        except Exception as e:
            self.logger.error(f"Error in table sample evaluation: {e}")
            return None, None, 0.0, time.time() - start_time
    
    def _evaluate_single_sample(self, 
                               pdf_bytes: bytes,
                               ground_truth_compressed: bytes,
                               parameters: EnhancementParameters) -> Tuple[Optional[float], Optional[float], float, float]:
        """Evaluate OCR performance on a single sample (legacy FC-AMF-OCR format)"""
        
        start_time = time.time()
        
        try:
            # Extract ground truth
            ground_truth = self._extract_ground_truth(ground_truth_compressed)
            if not ground_truth:
                return None, None, 0.0, 0.0
            
            # Create enhanced PDF
            enhanced_pdf_bytes = self._apply_enhancement(pdf_bytes, parameters)
            if not enhanced_pdf_bytes:
                return None, None, 0.0, 0.0
            
            # Run OCR
            ocr_result = self._run_ocr_on_bytes(enhanced_pdf_bytes)
            if not ocr_result:
                return None, None, 0.0, 0.0
            
            # Extract text and confidence
            extracted_text, avg_confidence = self._extract_text_and_confidence(ocr_result)
            
            # Calculate error rates
            cer = self._calculate_character_error_rate(ground_truth, extracted_text)
            wer = self._calculate_word_error_rate(ground_truth, extracted_text)
            
            processing_time = time.time() - start_time
            
            return cer, wer, avg_confidence, processing_time
            
        except Exception as e:
            self.logger.error(f"Error in single sample evaluation: {e}")
            return None, None, 0.0, time.time() - start_time
    
    def _extract_ground_truth(self, compressed_data: bytes) -> str:
        """Extract ground truth text from compressed OCR annotations"""
        try:
            # Decompress the JSON data
            decompressed = gzip.decompress(compressed_data)
            ocr_data = json.loads(decompressed.decode('utf-8'))
            
            # Extract text from OCR structure
            text_parts = []
            
            for page in ocr_data.get('pages', []):
                for block in page.get('blocks', []):
                    for line in block.get('lines', []):
                        for word in line.get('words', []):
                            if 'value' in word:
                                text_parts.append(word['value'])
                        text_parts.append(' ')  # Add space after each line
                    text_parts.append('\n')  # Add newline after each block
            
            return ''.join(text_parts).strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting ground truth: {e}")
            return ""
    
    def _apply_enhancement(self, pdf_bytes: bytes, parameters: EnhancementParameters) -> Optional[bytes]:
        """Apply enhancement parameters to PDF"""
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_input:
                temp_input.write(pdf_bytes)
                temp_input_path = temp_input.name
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # Process the PDF with given parameters
                self._enhance_pdf_with_parameters(temp_input_path, temp_output_path, parameters)
                
                # Read enhanced PDF
                with open(temp_output_path, 'rb') as f:
                    enhanced_bytes = f.read()
                
                return enhanced_bytes
                
            finally:
                # Clean up temp files
                os.unlink(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
                
        except Exception as e:
            self.logger.error(f"Error applying enhancement: {e}")
            return None
    
    def _enhance_pdf_with_parameters(self, input_path: str, output_path: str, 
                                   parameters: EnhancementParameters):
        """Apply specific enhancement parameters to a PDF"""
        
        pdf_doc = fitz.open(input_path)
        enhanced_pdf = fitz.open()
        
        try:
            for page_num in range(min(len(pdf_doc), 3)):  # Limit to first 3 pages for speed
                page = pdf_doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(parameters.matrix_scale, parameters.matrix_scale)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Apply enhancement
                import io
                pil_img = Image.open(io.BytesIO(img_data))
                enhanced_img = self._apply_enhancement_pipeline(pil_img, parameters)
                
                # Save to PDF
                img_bytes = io.BytesIO()
                enhanced_img.save(img_bytes, format='PNG', 
                                dpi=(parameters.enhancement_dpi, parameters.enhancement_dpi))
                img_bytes.seek(0)
                
                img_rect = fitz.Rect(0, 0, enhanced_img.width, enhanced_img.height)
                new_page = enhanced_pdf.new_page(width=img_rect.width, height=img_rect.height)
                new_page.insert_image(img_rect, stream=img_bytes.getvalue())
            
            enhanced_pdf.save(output_path)
            
        finally:
            enhanced_pdf.close()
            pdf_doc.close()
    
    def _apply_table_image_enhancement(self, image: Image.Image, parameters: EnhancementParameters) -> Optional[bytes]:
        """Apply enhancement to table image and convert to PDF bytes"""
        
        try:
            # Apply enhancement pipeline
            enhanced_image = self._apply_enhancement_pipeline(image, parameters)
            
            # Convert enhanced image to PDF bytes
            import io
            pdf_bytes = io.BytesIO()
            enhanced_image.save(pdf_bytes, format='PDF', 
                              dpi=(parameters.enhancement_dpi, parameters.enhancement_dpi))
            pdf_bytes.seek(0)
            
            return pdf_bytes.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error applying table image enhancement: {e}")
            return None
    
    def _apply_enhancement_pipeline(self, image: Image.Image, 
                                  parameters: EnhancementParameters) -> Image.Image:
        """Apply the enhancement pipeline with given parameters"""
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Stage 1: Basic PIL enhancements (table-optimized)
        enhancer = ImageEnhance.Contrast(image)
        enhanced_contrast = enhancer.enhance(parameters.table_contrast_factor)
        
        brightness_enhancer = ImageEnhance.Brightness(enhanced_contrast)
        adjusted_brightness = brightness_enhancer.enhance(parameters.table_brightness_factor)
        
        sharpener = ImageEnhance.Sharpness(adjusted_brightness)
        sharpened = sharpener.enhance(parameters.table_sharpness_factor)
        
        # Stage 2: OpenCV processing
        cv_image = cv2.cvtColor(np.array(sharpened), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Stage 3: Noise reduction
        denoised = cv2.bilateralFilter(
            gray, 
            parameters.bilateral_d,
            parameters.bilateral_sigma_color,
            parameters.bilateral_sigma_space
        )
        
        # Stage 4: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised,
            parameters.adaptive_max_value,
            parameters.adaptive_method,
            parameters.adaptive_thresh_type,
            parameters.adaptive_block_size,
            parameters.adaptive_c
        )
        
        # Stage 5: Ensure proper contrast
        if np.mean(adaptive_thresh) > 127:
            adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
        
        # Stage 6: Morphological operations (table-optimized)
        kernel = np.ones(parameters.morph_kernel_size, np.uint8)
        
        closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, 
                                iterations=parameters.table_morph_close_iterations)
        dilated = cv2.dilate(closed, kernel, iterations=parameters.table_morph_dilate_iterations)
        
        # Convert back to PIL
        final_image = Image.fromarray(dilated, mode='L')
        return final_image.convert('RGB')
    
    def _run_ocr_on_bytes(self, pdf_bytes: bytes):
        """Run OCR on enhanced PDF bytes"""
        try:
            # Upload to Mistral
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": "enhanced_sample.pdf",
                    "content": pdf_bytes,
                },
                purpose="ocr",
            )
            
            # Get signed URL
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            
            # Run OCR
            response = self.client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model=self.ocr_model,
                include_image_base64=False  # Don't need images for evaluation
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            return None
    
    def _extract_text_and_confidence(self, ocr_result) -> Tuple[str, float]:
        """Extract text and average confidence from OCR result"""
        
        text_parts = []
        confidences = []
        
        try:
            for page in ocr_result.pages:
                # Extract text from markdown
                text_parts.append(page.markdown)
                
                # Note: Mistral OCR doesn't provide word-level confidence scores
                # in the same way as other OCR engines, so we'll use a placeholder
                confidences.append(0.95)  # Default high confidence
            
            full_text = '\n'.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return full_text, avg_confidence
            
        except Exception as e:
            self.logger.error(f"Error extracting text and confidence: {e}")
            return "", 0.0
    
    def _calculate_character_error_rate(self, ground_truth: str, predicted: str) -> float:
        """Calculate Character Error Rate (CER)"""
        
        if not ground_truth and not predicted:
            return 0.0
        if not ground_truth:
            return 1.0
        
        if LEVENSHTEIN_AVAILABLE:
            distance = Levenshtein.distance(ground_truth, predicted)
            return distance / len(ground_truth)
        else:
            # Simple character-level comparison fallback
            return self._simple_character_error_rate(ground_truth, predicted)
    
    def _calculate_word_error_rate(self, ground_truth: str, predicted: str) -> float:
        """Calculate Word Error Rate (WER)"""
        
        gt_words = ground_truth.split()
        pred_words = predicted.split()
        
        if not gt_words and not pred_words:
            return 0.0
        if not gt_words:
            return 1.0
        
        if LEVENSHTEIN_AVAILABLE:
            distance = Levenshtein.distance(gt_words, pred_words)
            return distance / len(gt_words)
        else:
            # Simple word-level comparison fallback
            return self._simple_word_error_rate(gt_words, pred_words)
    
    def _simple_character_error_rate(self, ground_truth: str, predicted: str) -> float:
        """Simple CER calculation without external dependencies"""
        if len(ground_truth) == 0:
            return 1.0 if len(predicted) > 0 else 0.0
        
        correct_chars = sum(1 for a, b in zip(ground_truth, predicted) if a == b)
        return 1.0 - (correct_chars / len(ground_truth))
    
    def _simple_word_error_rate(self, gt_words: List[str], pred_words: List[str]) -> float:
        """Simple WER calculation without external dependencies"""
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) > 0 else 0.0
        
        correct_words = sum(1 for a, b in zip(gt_words, pred_words) if a == b)
        return 1.0 - (correct_words / len(gt_words))
    
    def create_cv_folds(self) -> List[Tuple[List, List]]:
        """Create cross-validation folds"""
        self.logger.info(f"Creating {self.n_folds}-fold cross-validation splits...")
        
        # Shuffle samples for random distribution
        random.seed(42)  # For reproducibility
        shuffled_samples = self.dataset_samples.copy()
        random.shuffle(shuffled_samples)
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        folds = []
        
        sample_indices = list(range(len(shuffled_samples)))
        
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(sample_indices)):
            train_samples = [shuffled_samples[i] for i in train_indices]
            val_samples = [shuffled_samples[i] for i in val_indices]
            
            folds.append((train_samples, val_samples))
            
            self.logger.info(f"Fold {fold_idx + 1}: {len(train_samples)} train, {len(val_samples)} validation")
        
        return folds
    
    def evaluate_parameters_cv(self, 
                              parameters: EnhancementParameters,
                              samples_per_fold: int = 5) -> CVResult:
        """Evaluate parameters using cross-validation"""
        
        self.logger.info(f"Evaluating parameters with {self.n_folds}-fold CV:")
        self.logger.info(f"  table_contrast={parameters.table_contrast_factor:.2f}, "
                        f"table_brightness={parameters.table_brightness_factor:.2f}, "
                        f"table_sharpness={parameters.table_sharpness_factor:.2f}")
        
        # Create CV folds
        folds = self.create_cv_folds()
        fold_results = []
        
        for fold_idx, (train_samples, val_samples) in enumerate(folds):
            self.logger.info(f"\n--- Fold {fold_idx + 1}/{self.n_folds} ---")
            
            # Use validation samples for this fold (limit to samples_per_fold)
            fold_samples = val_samples[:samples_per_fold]
            
            fold_cer_total = 0.0
            fold_wer_total = 0.0
            fold_confidence_total = 0.0
            fold_time_total = 0.0
            successful_evaluations = 0
            
            for sample_idx, sample in enumerate(fold_samples):
                try:
                    # Evaluate this sample
                    cer, wer, confidence, processing_time = self._evaluate_table_sample(
                        sample, parameters
                    )
                    
                    if cer is not None:  # Successful evaluation
                        fold_cer_total += cer
                        fold_wer_total += wer
                        fold_confidence_total += confidence
                        fold_time_total += processing_time
                        successful_evaluations += 1
                        
                        self.logger.debug(f"  Sample {sample_idx + 1}: CER={cer:.3f}, "
                                        f"WER={wer:.3f}, Conf={confidence:.3f}")
                    
                except Exception as e:
                    self.logger.warning(f"  Failed to evaluate sample {sample_idx + 1}: {e}")
                    continue
            
            if successful_evaluations == 0:
                self.logger.error(f"No successful evaluations in fold {fold_idx + 1}")
                fold_result = OCRResult(
                    character_error_rate=1.0,
                    word_error_rate=1.0,
                    confidence_score=0.0,
                    processing_time=float('inf'),
                    parameters=parameters
                )
            else:
                # Calculate fold averages
                fold_result = OCRResult(
                    character_error_rate=fold_cer_total / successful_evaluations,
                    word_error_rate=fold_wer_total / successful_evaluations,
                    confidence_score=fold_confidence_total / successful_evaluations,
                    processing_time=fold_time_total / successful_evaluations,
                    parameters=parameters
                )
            
            fold_results.append(fold_result)
            
            self.logger.info(f"Fold {fold_idx + 1} results: CER={fold_result.character_error_rate:.3f}, "
                           f"WER={fold_result.word_error_rate:.3f}, "
                           f"Confidence={fold_result.confidence_score:.3f}")
        
        # Calculate cross-validation statistics
        cers = [r.character_error_rate for r in fold_results]
        wers = [r.word_error_rate for r in fold_results]
        confidences = [r.confidence_score for r in fold_results]
        
        cv_result = CVResult(
            fold_results=fold_results,
            mean_cer=np.mean(cers),
            std_cer=np.std(cers),
            mean_wer=np.mean(wers),
            std_wer=np.std(wers),
            mean_confidence=np.mean(confidences),
            std_confidence=np.std(confidences),
            parameters=parameters,
            fold_count=len(fold_results)
        )
        
        self.logger.info(f"CV Results: CER={cv_result.mean_cer:.3f}±{cv_result.std_cer:.3f}, "
                        f"WER={cv_result.mean_wer:.3f}±{cv_result.std_wer:.3f}")
        
        return cv_result
    
    def run_grid_search(self, 
                       num_parameter_combinations: int = 20,
                       samples_per_combination: int = 5,
                       focused_search: bool = True) -> Dict:
        """Run grid search to find optimal parameters (with optional CV)"""
        
        if self.use_cv:
            return self.run_cv_grid_search(
                num_parameter_combinations=num_parameter_combinations,
                samples_per_fold=samples_per_combination,
                focused_search=focused_search
            )
        
        self.logger.info(f"Starting grid search with {num_parameter_combinations} "
                        f"parameter combinations, {samples_per_combination} samples each")
        
        # Generate parameter combinations
        parameter_combinations = self.generate_parameter_combinations(
            num_samples=num_parameter_combinations,
            focused_search=focused_search
        )
        
        self.results = []
        best_cer = float('inf')
        
        for i, parameters in enumerate(parameter_combinations):
            self.logger.info(f"\n--- Combination {i+1}/{len(parameter_combinations)} ---")
            
            # Evaluate this parameter set
            result = self.evaluate_parameters(parameters, samples_per_combination)
            self.results.append(result)
            
            # Track best result
            if result.character_error_rate < best_cer:
                best_cer = result.character_error_rate
                self.best_result = result
                self.logger.info(f"🎯 NEW BEST: CER={best_cer:.4f}")
            
            # Save intermediate results
            self._save_intermediate_results()
        
        # Save final results
        final_results = self._save_final_results()
        
        self.logger.info("\n" + "="*50)
        self.logger.info("GRID SEARCH COMPLETE")
        self.logger.info("="*50)
        
        if self.best_result:
            self.logger.info(f"Best CER: {self.best_result.character_error_rate:.4f}")
            self.logger.info(f"Best WER: {self.best_result.word_error_rate:.4f}")
            self.logger.info(f"Best parameters saved to: {final_results['output_file']}")
        
        return final_results
    
    def run_cv_grid_search(self, 
                          num_parameter_combinations: int = 15,
                          samples_per_fold: int = 3,
                          focused_search: bool = True) -> Dict:
        """Run grid search with cross-validation"""
        
        self.logger.info(f"Starting CV grid search:")
        self.logger.info(f"  Parameter combinations: {num_parameter_combinations}")
        self.logger.info(f"  Samples per fold: {samples_per_fold}")
        self.logger.info(f"  Cross-validation folds: {self.n_folds}")
        self.logger.info(f"  Total evaluations: {num_parameter_combinations * self.n_folds * samples_per_fold}")
        
        # Generate parameter combinations
        parameter_combinations = self.generate_parameter_combinations(
            num_samples=num_parameter_combinations,
            focused_search=focused_search
        )
        
        self.cv_results = []
        best_mean_cer = float('inf')
        
        for i, parameters in enumerate(parameter_combinations):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Parameter Combination {i+1}/{len(parameter_combinations)}")
            self.logger.info(f"{'='*60}")
            
            # Evaluate with cross-validation
            cv_result = self.evaluate_parameters_cv(parameters, samples_per_fold)
            self.cv_results.append(cv_result)
            
            # Track best result (using mean CER for robust selection)
            if cv_result.mean_cer < best_mean_cer:
                best_mean_cer = cv_result.mean_cer
                self.best_cv_result = cv_result
                self.logger.info(f"🎯 NEW BEST: Mean CER={cv_result.mean_cer:.4f}±{cv_result.std_cer:.4f}")
            
            # Save intermediate results
            self._save_cv_intermediate_results()
        
        # Save final results
        final_results = self._save_cv_final_results()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("CROSS-VALIDATION GRID SEARCH COMPLETE")
        self.logger.info(f"{'='*60}")
        
        if self.best_cv_result:
            self.logger.info(f"Best Mean CER: {self.best_cv_result.mean_cer:.4f} ± {self.best_cv_result.std_cer:.4f}")
            self.logger.info(f"Best Mean WER: {self.best_cv_result.mean_wer:.4f} ± {self.best_cv_result.std_wer:.4f}")
            self.logger.info(f"Best Mean Confidence: {self.best_cv_result.mean_confidence:.4f} ± {self.best_cv_result.std_confidence:.4f}")
            self.logger.info(f"Results saved to: {final_results['output_file']}")
            
            # Create optimal parameters file
            if final_results.get('optimal_params_file'):
                self.logger.info(f"Optimal parameters: {final_results['optimal_params_file']}")
        
        return final_results
    
    def _save_intermediate_results(self):
        """Save intermediate results during grid search"""
        results_data = {
            'timestamp': time.time(),
            'results': [
                {
                    'character_error_rate': r.character_error_rate,
                    'word_error_rate': r.word_error_rate,
                    'confidence_score': r.confidence_score,
                    'processing_time': r.processing_time,
                    'parameters': r.parameters.__dict__
                }
                for r in self.results
            ]
        }
        
        with open('ocr_tuning_intermediate.json', 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def _save_cv_intermediate_results(self):
        """Save intermediate CV results"""
        results_data = {
            'timestamp': time.time(),
            'n_folds': self.n_folds,
            'total_combinations_tested': len(self.cv_results),
            'cv_results': []
        }
        
        for cv_result in self.cv_results:
            result_dict = {
                'parameters': {k: (int(v) if isinstance(v, (np.int64, np.int32)) else 
                                 float(v) if isinstance(v, (np.float64, np.float32)) else v) 
                              for k, v in asdict(cv_result.parameters).items()},
                'mean_cer': float(cv_result.mean_cer),
                'std_cer': float(cv_result.std_cer),
                'mean_wer': float(cv_result.mean_wer),
                'std_wer': float(cv_result.std_wer),
                'mean_confidence': float(cv_result.mean_confidence),
                'std_confidence': float(cv_result.std_confidence),
                'fold_count': int(cv_result.fold_count),
                'fold_results': [
                    {
                        'character_error_rate': float(fold.character_error_rate),
                        'word_error_rate': float(fold.word_error_rate),
                        'confidence_score': float(fold.confidence_score),
                        'processing_time': float(fold.processing_time)
                    }
                    for fold in cv_result.fold_results
                ]
            }
            results_data['cv_results'].append(result_dict)
        
        with open('ocr_cv_tuning_intermediate.json', 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def _save_final_results(self) -> Dict:
        """Save final results and return summary"""
        
        timestamp = int(time.time())
        output_file = f'ocr_tuning_results_{timestamp}.json'
        
        # Prepare results data
        results_data = {
            'summary': {
                'total_combinations_tested': len(self.results),
                'best_character_error_rate': self.best_result.character_error_rate if self.best_result else None,
                'best_word_error_rate': self.best_result.word_error_rate if self.best_result else None,
                'best_parameters': self.best_result.parameters.__dict__ if self.best_result else None,
                'timestamp': timestamp
            },
            'all_results': [
                {
                    'character_error_rate': r.character_error_rate,
                    'word_error_rate': r.word_error_rate,
                    'confidence_score': r.confidence_score,
                    'processing_time': r.processing_time,
                    'parameters': r.parameters.__dict__
                }
                for r in self.results
            ]
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create optimal parameters file for easy import
        if self.best_result:
            optimal_params_file = f'optimal_parameters_{timestamp}.py'
            with open(optimal_params_file, 'w') as f:
                f.write("# Optimal enhancement parameters found by grid search\n")
                f.write("# Generated by OCR Tuner\n\n")
                f.write("OPTIMAL_PARAMETERS = {\n")
                for key, value in self.best_result.parameters.__dict__.items():
                    f.write(f"    '{key}': {repr(value)},\n")
                f.write("}\n\n")
                f.write(f"# Performance metrics:\n")
                f.write(f"# Character Error Rate: {self.best_result.character_error_rate:.4f}\n")
                f.write(f"# Word Error Rate: {self.best_result.word_error_rate:.4f}\n")
                f.write(f"# Confidence Score: {self.best_result.confidence_score:.4f}\n")
        
        return {
            'output_file': output_file,
            'optimal_params_file': optimal_params_file if self.best_result else None,
            'best_result': self.best_result
        }
    
    def _save_cv_final_results(self) -> Dict:
        """Save final CV results and return summary"""
        
        timestamp = int(time.time())
        output_file = f'ocr_cv_tuning_results_{timestamp}.json'
        
        # Prepare results data
        results_data = {
            'summary': {
                'method': 'cross_validation',
                'n_folds': int(self.n_folds),
                'total_combinations_tested': len(self.cv_results),
                'best_mean_cer': float(self.best_cv_result.mean_cer) if self.best_cv_result else None,
                'best_std_cer': float(self.best_cv_result.std_cer) if self.best_cv_result else None,
                'best_mean_wer': float(self.best_cv_result.mean_wer) if self.best_cv_result else None,
                'best_std_wer': float(self.best_cv_result.std_wer) if self.best_cv_result else None,
                'best_parameters': {k: (int(v) if isinstance(v, (np.int64, np.int32)) else 
                                      float(v) if isinstance(v, (np.float64, np.float32)) else v) 
                                   for k, v in asdict(self.best_cv_result.parameters).items()} if self.best_cv_result else None,
                'timestamp': timestamp
            },
            'all_cv_results': []
        }
        
        # Add all CV results
        for cv_result in self.cv_results:
            result_dict = {
                'parameters': {k: (int(v) if isinstance(v, (np.int64, np.int32)) else 
                                 float(v) if isinstance(v, (np.float64, np.float32)) else v) 
                              for k, v in asdict(cv_result.parameters).items()},
                'mean_cer': float(cv_result.mean_cer),
                'std_cer': float(cv_result.std_cer),
                'mean_wer': float(cv_result.mean_wer),
                'std_wer': float(cv_result.std_wer),
                'mean_confidence': float(cv_result.mean_confidence),
                'std_confidence': float(cv_result.std_confidence),
                'fold_count': int(cv_result.fold_count),
                'fold_results': [
                    {
                        'character_error_rate': float(fold.character_error_rate),
                        'word_error_rate': float(fold.word_error_rate),
                        'confidence_score': float(fold.confidence_score),
                        'processing_time': float(fold.processing_time)
                    }
                    for fold in cv_result.fold_results
                ]
            }
            results_data['all_cv_results'].append(result_dict)
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create optimal parameters file for easy import
        optimal_params_file = None
        if self.best_cv_result:
            optimal_params_file = f'optimal_cv_parameters_{timestamp}.py'
            with open(optimal_params_file, 'w') as f:
                f.write("# Optimal enhancement parameters found by 5-fold cross-validation\n")
                f.write("# Generated by OCR Tuner with table-focused datasets\n\n")
                f.write("OPTIMAL_PARAMETERS = {\n")
                for key, value in asdict(self.best_cv_result.parameters).items():
                    f.write(f"    '{key}': {repr(value)},\n")
                f.write("}\n\n")
                f.write(f"# Cross-validation performance metrics:\n")
                f.write(f"# Mean Character Error Rate: {self.best_cv_result.mean_cer:.4f} ± {self.best_cv_result.std_cer:.4f}\n")
                f.write(f"# Mean Word Error Rate: {self.best_cv_result.mean_wer:.4f} ± {self.best_cv_result.std_wer:.4f}\n")
                f.write(f"# Mean Confidence Score: {self.best_cv_result.mean_confidence:.4f} ± {self.best_cv_result.std_confidence:.4f}\n")
                f.write(f"# Number of CV folds: {self.best_cv_result.fold_count}\n")
                f.write(f"# Dataset: Mixed table datasets (PubTables-1M + PubTabNet 2.0)\n")
        
        return {
            'output_file': output_file,
            'optimal_params_file': optimal_params_file,
            'best_cv_result': self.best_cv_result
        }

def main():
    """Main function to run OCR parameter tuning"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR Enhancement Parameter Tuner with Table Datasets")
    parser.add_argument("--api-key", help="Mistral API key")
    parser.add_argument("--combinations", type=int, default=15, 
                       help="Number of parameter combinations to test")
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of samples per combination/fold")
    parser.add_argument("--focused", action="store_true", default=True,
                       help="Use focused search around table-optimized parameters")
    parser.add_argument("--cv", action="store_true", default=True,
                       help="Use 5-fold cross-validation (recommended)")
    parser.add_argument("--folds", type=int, default=5,
                       help="Number of cross-validation folds")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print("❌ Error: Mistral API key required!")
        print("   Set via --api-key or MISTRAL_API_KEY environment variable")
        return 1
    
    try:
        # Initialize tuner with CV support
        tuner = OCRTuner(api_key=api_key, use_cv=args.cv, n_folds=args.folds)
        
        print(f"🔧 OCR Parameter Tuning Configuration:")
        print(f"   Method: {'5-fold Cross-Validation' if args.cv else 'Simple Grid Search'}")
        print(f"   Dataset: Mixed Table Datasets (PubTables-1M + PubTabNet 2.0)")
        print(f"   Parameter combinations: {args.combinations}")
        print(f"   Samples per {'fold' if args.cv else 'combination'}: {args.samples}")
        print(f"   Search type: {'Focused (around optimized params)' if args.focused else 'Broad'}")
        
        # Run grid search
        results = tuner.run_grid_search(
            num_parameter_combinations=args.combinations,
            samples_per_combination=args.samples,
            focused_search=args.focused
        )
        
        print(f"\n🎯 {'Cross-validation' if args.cv else 'Grid search'} tuning complete!")
        print(f"Results saved to: {results['output_file']}")
        
        if results.get('optimal_params_file'):
            print(f"Optimal parameters: {results['optimal_params_file']}")
            
        # Show best results
        if args.cv and results.get('best_cv_result'):
            best = results['best_cv_result']
            print(f"\n📊 Best CV Results:")
            print(f"   Mean CER: {best.mean_cer:.4f} ± {best.std_cer:.4f}")
            print(f"   Mean WER: {best.mean_wer:.4f} ± {best.std_wer:.4f}")
            print(f"   Mean Confidence: {best.mean_confidence:.4f} ± {best.std_confidence:.4f}")
        elif results.get('best_result'):
            best = results['best_result']
            print(f"\n📊 Best Results:")
            print(f"   CER: {best.character_error_rate:.4f}")
            print(f"   WER: {best.word_error_rate:.4f}")
            print(f"   Confidence: {best.confidence_score:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during tuning: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())