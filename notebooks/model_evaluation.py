import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image

from datasets import Dataset, Features, Value, Sequence
from transformers import AutoProcessor, AutoModelForTokenClassification
import torch

# Assuming these are available from your pipeline.models
from pipeline.models.nanonets_ocr import NanonetsOCR
from pipeline.models.layoutlm_invoice import LayoutLMv3Invoice

class ModelEvaluator:
    """
    Evaluates the performance of different document extraction models
    (Nanonets-OCR-s, LayoutLMv3-Invoice) on a given dataset.
    """

    def __init__(self, nanonets_api_key: str = None, layoutlm_model_path: str = "microsoft/layoutlmv3-base-finetuned-docvqa"):
        self.nanonets_client = NanonetsOCR(api_key=nanonets_api_key)
        self.layoutlm_client = LayoutLMv3Invoice(model_name_or_path=layoutlm_model_path)

        # Define the target fields for evaluation
        self.target_fields = [
            "vendor_name", "invoice_number", "invoice_date", "due_date",
            "subtotal", "tax_total", "grand_total", "currency",
            "HS_code", "country_of_origin", "shipment_weight"
        ]
        # Line items will require special handling for evaluation

    def _load_dataset(self, dataset_path: Path) -> Dataset:
        """
        Loads the Hugging Face dataset.
        Args:
            dataset_path: Path to the Hugging Face dataset (e.g., from cvat_to_hf_dataset.py).
        Returns:
            A Hugging Face Dataset object.
        """
        # This is a placeholder. You'd load your actual dataset here.
        # For demonstration, let's create a dummy dataset.
        features = Features({
            "id": Value("string"),
            "image_path": Value("string"),
            "document_type": Value("string"),
            "annotations": Sequence({
                "field_name": Value("string"),
                "text_content": Value("string"),
                "bbox": Sequence(Value("int")), # [x_min, y_min, x_max, y_max]
            })
        })
        dummy_data = [
            {
                "id": "doc_001",
                "image_path": "dummy_invoice_1.png", # Replace with actual image paths
                "document_type": "invoice",
                "annotations": [
                    {"field_name": "vendor_name", "text_content": "Acme Corp", "bbox": [100, 50, 300, 80]},
                    {"field_name": "invoice_number", "text_content": "INV-2023-001", "bbox": [500, 50, 700, 80]},
                    {"field_name": "grand_total", "text_content": "123.45", "bbox": [700, 800, 800, 820]},
                    {"field_name": "currency", "text_content": "USD", "bbox": [650, 800, 690, 820]},
                ]
            },
            {
                "id": "doc_002",
                "image_path": "dummy_invoice_2.png", # Replace with actual image paths
                "document_type": "invoice",
                "annotations": [
                    {"field_name": "vendor_name", "text_content": "Beta Ltd", "bbox": [120, 60, 320, 90]},
                    {"field_name": "invoice_number", "text_content": "INV-2023-002", "bbox": [520, 60, 720, 90]},
                    {"field_name": "grand_total", "text_content": "99.99", "bbox": [720, 810, 820, 830]},
                    {"field_name": "currency", "text_content": "EUR", "bbox": [670, 810, 710, 830]},
                ]
            }
        ]
        # Create dummy image files for testing
        for data in dummy_data:
            img = Image.new('RGB', (1000, 1000), color = 'white')
            img.save(data["image_path"])
            print(f"Created dummy image: {data['image_path']}")

        return Dataset.from_list(dummy_data, features=features)

    def evaluate_model(self, model_client: Any, dataset: Dataset, precision: str) -> Dict[str, Any]:
        """
        Evaluates a given model client on the dataset.
        Args:
            model_client: An instance of NanonetsOCR or LayoutLMv3Invoice.
            dataset: The Hugging Face Dataset to evaluate on.
            precision: 'fp16' or 'int8'.
        Returns:
            A dictionary of evaluation metrics (e.g., F1 score per field).
        """
        predictions = []
        ground_truths = []

        for item in dataset:
            image_path = item["image_path"]
            image = Image.open(image_path).convert("RGB")

            if isinstance(model_client, NanonetsOCR):
                # Nanonets expects image path
                if precision == "fp16":
                    model_output = model_client.predict_fp16(image_path, model_id="your_nanonets_model_id")
                else:
                    model_output = model_client.predict_int8(image_path, model_id="your_nanonets_model_id")
                # Convert Nanonets output to a consistent format
                predicted_fields = {p["label"]: p["text"] for p in model_output.get("prediction", [])}
            elif isinstance(model_client, LayoutLMv3Invoice):
                # LayoutLMv3 expects PIL Image
                if precision == "fp16":
                    model_output = model_client.predict_fp16(image)
                else:
                    model_output = model_client.predict_int8(image)
                # Convert LayoutLMv3 output to a consistent format
                predicted_fields = {p["label"]: p["text"] for p in model_output}
            else:
                raise ValueError("Unsupported model client type.")

            # Prepare ground truth
            gt_fields = {ann["field_name"]: ann["text_content"] for ann in item["annotations"]}

            predictions.append(predicted_fields)
            ground_truths.append(gt_fields)

        # Calculate F1 score (simplified for demonstration)
        # A proper F1 calculation would involve matching predicted fields to ground truth
        # considering partial matches, bounding box overlap, etc.
        # For now, we'll do exact text match for simplicity.
        field_f1_scores: Dict[str, float] = {}
        for field in self.target_fields:
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for pred, gt in zip(predictions, ground_truths):
                pred_value = pred.get(field)
                gt_value = gt.get(field)

                if gt_value is not None and pred_value is not None:
                    if pred_value.strip().lower() == gt_value.strip().lower():
                        true_positives += 1
                    else:
                        false_positives += 1 # Predicted but incorrect
                        false_negatives += 1 # Ground truth existed but not correctly predicted
                elif gt_value is not None and pred_value is None:
                    false_negatives += 1
                elif gt_value is None and pred_value is not None:
                    false_positives += 1

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            field_f1_scores[field] = f1

        return {"field_f1_scores": field_f1_scores}

    def run_evaluation(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Runs the full evaluation process for all models and precisions.
        Args:
            dataset_path: Path to the Hugging Face dataset.
        Returns:
            A decision matrix with evaluation results.
        """
        dataset = self._load_dataset(dataset_path)

        results = {}

        print("\n--- Evaluating Nanonets-OCR-s (FP16) ---")
        nanonets_fp16_metrics = self.evaluate_model(self.nanonets_client, dataset, "fp16")
        results["nanonets_fp16"] = nanonets_fp16_metrics

        print("\n--- Evaluating Nanonets-OCR-s (INT8) ---")
        nanonets_int8_metrics = self.evaluate_model(self.nanonets_client, dataset, "int8")
        results["nanonets_int8"] = nanonets_int8_metrics

        print("\n--- Evaluating LayoutLMv3-Invoice (FP16) ---")
        layoutlm_fp16_metrics = self.evaluate_model(self.layoutlm_client, dataset, "fp16")
        results["layoutlmv3_fp16"] = layoutlm_fp16_metrics

        print("\n--- Evaluating LayoutLMv3-Invoice (INT8) ---")
        layoutlm_int8_metrics = self.evaluate_model(self.layoutlm_client, dataset, "int8")
        results["layoutlmv3_int8"] = layoutlm_int8_metrics

        # Clean up dummy image files
        for item in dataset:
            Path(item["image_path"]).unlink(missing_ok=True)

        return self._generate_decision_matrix(results)

    def _generate_decision_matrix(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a decision matrix based on evaluation results.
        This is a simplified example. A real decision matrix would consider
        latency, throughput, VRAM usage, and accuracy.
        Args:
            results: Dictionary of evaluation results from different models/precisions.
        Returns:
            A dictionary representing the decision matrix.
        """
        decision_matrix = {
            "summary": "Model Performance Comparison",
            "models": []
        }

        for model_name, metrics in results.items():
            avg_f1 = sum(metrics["field_f1_scores"].values()) / len(metrics["field_f1_scores"])
            decision_matrix["models"].append({
                "name": model_name,
                "average_f1_score": f"{avg_f1:.4f}",
                "field_f1_scores": {k: f"{v:.4f}" for k, v in metrics["field_f1_scores"].items()},
                "notes": "Latency and throughput metrics would be added here from load tests."
            })
        return decision_matrix

if __name__ == "__main__":
    # Example usage:
    # Ensure you have a dummy dataset or integrate with your actual dataset path
    # For Nanonets, you'll need a valid API key.
    # For LayoutLMv3, ensure the model is downloaded or accessible.

    # Create a dummy dataset path (this will be replaced by your actual dataset)
    dummy_dataset_path = Path("./dummy_hf_dataset") # This path is not actually used for loading in this dummy example

    evaluator = ModelEvaluator(nanonets_api_key="YOUR_NANONETS_API_KEY") # Replace with your actual API key or env var
    decision_matrix = evaluator.run_evaluation(dummy_dataset_path)

    print("\n--- Decision Matrix ---")
    print(json.dumps(decision_matrix, indent=2))
