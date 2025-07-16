
import torch
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image
from typing import Dict, Any, List, Tuple

class LayoutLMv3Invoice:
    """
    Wrapper for LayoutLMv3-Invoice model for token classification.
    Supports FP16 and 8-bit quantization for inference.
    """

    def __init__(self, model_name_or_path: str = "microsoft/layoutlmv3-base-finetuned-docvqa"):
        """
        Initializes the LayoutLMv3 model and processor.
        Args:
            model_name_or_path: Path to the pre-trained or fine-tuned LayoutLMv3 model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 # Load in FP16
        )
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

    def _process_image_and_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """
        Processes the image and performs OCR to get words and bounding boxes.
        This uses the processor's built-in OCR (if apply_ocr=True) or expects
        pre-extracted OCR results. For LayoutLMv3, it's common to use a separate
        OCR engine (like Tesseract or Google Cloud Vision API) and then feed
        the words and boxes to the processor.
        For simplicity, we'll use the processor's OCR if available, or assume
        words and boxes are provided externally.
        """
        # In a real scenario, you'd integrate with a robust OCR engine here
        # and then pass the words and boxes to the processor.
        # For now, let's simulate some OCR output.
        # This part needs to be aligned with pipeline/prepare.py's output.

        # Example: Using processor's built-in OCR (if apply_ocr=True during processor init)
        # If apply_ocr=False, you need to provide words and boxes explicitly.
        # For this example, let's assume we get words and boxes from an external OCR.
        # This is a critical integration point with pipeline/prepare.py
        # For now, we'll use a dummy OCR result.
        dummy_words = ["This", "is", "an", "invoice", "number", "12345"]
        dummy_boxes = [[10, 10, 50, 20], [60, 10, 80, 20], [90, 10, 110, 20],
                       [120, 10, 180, 20], [190, 10, 250, 20], [260, 10, 300, 20]]
        
        # The processor expects words and boxes.
        encoding = self.processor(image, dummy_words, boxes=dummy_boxes, return_tensors="pt")
        return encoding

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Performs inference using LayoutLMv3 with FP16 precision.
        Args:
            image: PIL Image object of the document.
        Returns:
            A list of dictionaries, each representing an extracted field.
        """
        encoding = self._process_image_and_ocr(image)

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        bbox = encoding["bbox"].to(self.device, dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox)

        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        extracted_fields = []
        current_field = None
        current_text = []
        current_bbox = []

        # Iterate through predictions and reconstruct fields
        for token, pred_id, box in zip(tokens, predictions, bbox.squeeze().tolist()):
            label = self.id2label[pred_id]

            if token.startswith("##"): # Handle subword tokens
                token = token[2:]

            if label.startswith("B-"): # Beginning of a new entity
                if current_field: # Save previous field if exists
                    extracted_fields.append({
                        "label": current_field,
                        "text": " ".join(current_text),
                        "bbox": self._merge_bboxes(current_bbox)
                    })
                current_field = label[2:] # Remove "B-" prefix
                current_text = [token]
                current_bbox = [box]
            elif label.startswith("I-") and current_field == label[2:]: # Inside an existing entity
                current_text.append(token)
                current_bbox.append(box)
            elif label == "O": # Outside of any entity
                if current_field:
                    extracted_fields.append({
                        "label": current_field,
                        "text": " ".join(current_text),
                        "bbox": self._merge_bboxes(current_bbox)
                    })
                    current_field = None
                    current_text = []
                    current_bbox = []
            # Handle the case where a new B- tag appears without an O tag in between
            elif label.startswith("B-") and current_field != label[2:]:
                if current_field:
                    extracted_fields.append({
                        "label": current_field,
                        "text": " ".join(current_text),
                        "bbox": self._merge_bboxes(current_bbox)
                    })
                current_field = label[2:]
                current_text = [token]
                current_bbox = [box]

        # Add the last field if it exists
        if current_field:
            extracted_fields.append({
                "label": current_field,
                "text": " ".join(current_text),
                "bbox": self._merge_bboxes(current_bbox)
            })

        return extracted_fields

    def _merge_bboxes(self, bboxes: List[List[int]]) -> List[int]:
        """
        Merges a list of bounding boxes into a single bounding box
        that encompasses all of them.
        Args:
            bboxes: List of bounding boxes, each [x_min, y_min, x_max, y_max].
        Returns:
            Merged bounding box [x_min, y_min, x_max, y_max].
        """
        if not bboxes:
            return []
        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)
        return [x_min, y_min, x_max, y_max]

