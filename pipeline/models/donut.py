import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from typing import Dict, Any, List
import json

class DonutModel:
    """
    Wrapper for Donut (Document Understanding Transformer) model.
    Supports FP16 and 8-bit quantization for inference.
    """

    def __init__(self, model_name_or_path: str = "naver-clova-ix/donut-base-finetuned-docvqa"):
        """
        Initializes the Donut model and processor.
        Args:
            model_name_or_path: Path to the pre-trained or fine-tuned Donut model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

    def _predict(self, image: Image.Image, precision: str) -> Dict[str, Any]:
        """
        Internal prediction method with precision handling.
        Args:
            image: PIL Image object of the document.
            precision: 'fp16' or 'int8'.
        Returns:
            A dictionary containing the extracted data.
        """
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            if precision == "fp16":
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(pixel_values)
            elif precision == "int8":
                # For 8-bit, you'd typically quantize the model once during init
                # and then use the quantized model here.
                # transformers library supports `load_in_8bit=True` during from_pretrained.
                # For this example, we'll just run regular inference if not explicitly quantized.
                # A proper 8-bit path would involve `bitsandbytes` integration.
                outputs = self.model.generate(pixel_values)
            else:
                raise ValueError(f"Unsupported precision: {precision}")

        # Decode the generated tokens
        decoded_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Donut output is typically a JSON string. Parse it.
        try:
            extracted_data = json.loads(decoded_output)
        except json.JSONDecodeError:
            extracted_data = {"raw_output": decoded_output, "error": "Failed to parse JSON from Donut output"}

        return extracted_data

    def predict_fp16(self, image: Image.Image) -> Dict[str, Any]:
        """
        Performs inference using Donut with FP16 precision.
        Args:
            image: PIL Image object of the document.
        Returns:
            Extracted data.
        """
        return self._predict(image, "fp16")

    def predict_int8(self, image: Image.Image) -> Dict[str, Any]:
        """
        Performs inference using Donut with 8-bit quantization.
        Note: For true 8-bit inference, the model should ideally be loaded
        with `load_in_8bit=True` during initialization, which requires `bitsandbytes`.
        This method assumes the model is already quantized or handles it internally.
        Args:
            image: PIL Image object of the document.
        Returns:
            Extracted data.
        """
        # A more robust 8-bit path would involve:
        # from transformers import BitsAndBytesConfig
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # self.model = AutoModelForVision2Seq.from_pretrained(
        #     model_name_or_path, quantization_config=quantization_config
        # )
        # For now, it will run standard inference if not explicitly quantized.
        return self._predict(image, "int8")