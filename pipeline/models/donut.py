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
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 # Load in FP16
        )
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Performs inference using Donut with FP16 precision.
        Args:
            image: PIL Image object of the document.
        Returns:
            A dictionary containing the extracted data.
        """
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(pixel_values)

        # Decode the generated tokens
        decoded_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Donut output is typically a JSON string. Parse it.
        try:
            # The output can sometimes be a list of JSON objects, so we need to handle that
            # by finding the start and end of the JSON string.
            json_start = decoded_output.find('{')
            json_end = decoded_output.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = decoded_output[json_start:json_end]
                extracted_data = json.loads(json_str)
            else:
                extracted_data = {"raw_output": decoded_output, "error": "No JSON object found in Donut output"}

        except json.JSONDecodeError:
            extracted_data = {"raw_output": decoded_output, "error": "Failed to parse JSON from Donut output"}

        return extracted_data