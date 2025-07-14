
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import Dataset, Features, Value, Sequence
from transformers import AutoProcessor, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig # For 8-bit quantization
from PIL import Image
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

# Assume prepare.py is available for OCR and image processing
from pipeline.prepare import ocr_image

class LayoutLMv3Trainer:
    """
    Handles training and evaluation of LayoutLMv3-Invoice model.
    """

    def __init__(self, model_name_or_path: str = "microsoft/layoutlmv3-base",
                 output_dir: str = "./results",
                 quantize_8bit: bool = False):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.quantize_8bit = quantize_8bit

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, apply_ocr=False)

        # Placeholder for labels - these should come from your dataset
        # For now, using dummy labels based on the schema
        self.labels = [
            "vendor_name", "invoice_number", "invoice_date", "due_date",
            "subtotal", "tax_total", "grand_total", "currency",
            "HS_code", "country_of_origin", "shipment_weight"
        ]
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        # Load model
        if quantize_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path, quantization_config=bnb_config, num_labels=len(self.labels), id2label=self.id2label, label2id=self.label2id
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path, num_labels=len(self.labels), id2label=self.id2label, label2id=self.label2id
            )
        self.model.to(self.device)

    def _normalize_bbox(self, bbox: List[int], width: int, height: int) -> List[int]:
        """
        Normalizes bounding box coordinates to be between 0 and 1000.
        """
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]

    def _process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single example for LayoutLMv3 training.
        This involves OCR, tokenization, and aligning labels.
        """
        image_path = example["image_path"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Perform OCR using the prepare module
        ocr_results = ocr_image(image)
        words = ocr_results["words"]
        boxes = [self._normalize_bbox(b, width, height) for b in ocr_results["boxes"]]

        # Tokenize and align labels
        tokenized_inputs = self.processor(image, words, boxes=boxes, return_offsets_mapping=True, truncation=True)

        labels = []
        offset_mapping = tokenized_inputs["offset_mapping"][0]
        word_ids = tokenized_inputs.word_ids(batch_index=0)

        previous_word_idx = None
        sequence_ids = tokenized_inputs.sequence_ids(batch_index=0)

        for idx, word_idx in enumerate(word_ids):
            if sequence_ids[idx] == 1: # Only label tokens from the first sequence (the document)
                if word_idx is None or word_idx == previous_word_idx:
                    labels.append(-100) # Special token or subword of previous word
                else:
                    # Find the corresponding annotation for this word
                    found_label = "O"
                    for annotation in example["annotations"]:
                        # This is a simplified check. A more robust approach would involve
                        # checking bounding box overlap or more sophisticated text matching.
                        if annotation["text_content"] in words[word_idx]:
                            found_label = annotation["field_name"]
                            break
                    labels.append(self.label2id[found_label])
                previous_word_idx = word_idx
            else:
                labels.append(-100) # Special tokens

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for the DataLoader.
        """
        # This is a simplified collate function. For LayoutLMv3, you need to pad
        # input_ids, attention_mask, token_type_ids, bbox, and labels.
        # The processor's tokenizer can handle this.
        return self.processor.tokenizer.pad(batch, return_tensors="pt")

    def compute_metrics(self, p: Any) -> Dict[str, float]:
        """
        Computes metrics for token classification.
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_labels = [[self.id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Ensure lengths match for seqeval
        min_len = min(len(true_labels), len(true_predictions))
        true_labels = true_labels[:min_len]
        true_predictions = true_predictions[:min_len]

        results = {
            "f1": f1_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
        }
        return results

    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        """
        Trains the LayoutLMv3 model.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3, # Adjust as needed
            per_device_train_batch_size=2, # Adjust based on VRAM
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1, # Adjust for larger effective batch size
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            learning_rate=5e-5,
            fp16=torch.cuda.is_available(), # Enable FP16 if CUDA is available
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self._collate_fn,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(f"{self.output_dir}/final_model")

    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluates the trained model.
        """
        eval_results = self.trainer.evaluate(eval_dataset)
        print(f"Evaluation results: {eval_results}")
        return eval_results


if __name__ == "__main__":
    # Dummy dataset creation for demonstration
    # In a real scenario, you would load your dataset using cvat_to_hf_dataset.py
    # and split it into train and validation sets.

    # Create dummy image files for testing
    dummy_image_paths = [f"dummy_doc_{i}.png" for i in range(5)]
    for img_path in dummy_image_paths:
        img = Image.new('RGB', (1000, 1000), color = 'white')
        img.save(img_path)
        print(f"Created dummy image: {img_path}")

    dummy_data = [
        {
            "id": f"doc_{i}",
            "image_path": dummy_image_paths[i],
            "document_type": "invoice",
            "annotations": [
                {"field_name": "vendor_name", "text_content": "Vendor A", "bbox": [100, 50, 300, 80]},
                {"field_name": "invoice_number", "text_content": f"INV-00{i}", "bbox": [500, 50, 700, 80]},
                {"field_name": "grand_total", "text_content": f"{100.0 + i}", "bbox": [700, 800, 800, 820]},
                {"field_name": "currency", "text_content": "USD", "bbox": [650, 800, 690, 820]},
            ]
        } for i in range(5)
    ]

    # Split into train and eval (very small for demonstration)
    train_data = dummy_data[:4]
    eval_data = dummy_data[4:]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # Initialize and train the model
    trainer = LayoutLMv3Trainer(output_dir="./layoutlmv3_results", quantize_8bit=False)

    # Apply processing to datasets
    train_dataset = train_dataset.map(trainer._process_example, batched=False)
    eval_dataset = eval_dataset.map(trainer._process_example, batched=False)

    # Remove columns that are not needed by the model
    train_dataset = train_dataset.remove_columns(["id", "image_path", "document_type", "annotations", "offset_mapping", "word_ids"])
    eval_dataset = eval_dataset.remove_columns(["id", "image_path", "document_type", "annotations", "offset_mapping", "word_ids"])

    trainer.train(train_dataset, eval_dataset)

    # Clean up dummy image files
    for img_path in dummy_image_paths:
        Path(img_path).unlink(missing_ok=True)

