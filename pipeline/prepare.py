
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_path
from typing import List, Tuple, Dict, Any
import pytesseract
import os

# Set Tesseract path if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # Adjust path as needed for your system

def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Converts a PDF file into a list of PIL Image objects.
    Args:
        pdf_path: Path to the PDF file.
        dpi: DPI for rendering the PDF pages.
    Returns:
        A list of PIL Image objects, one for each page.
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    return images

def deskew_image(image: Image.Image) -> Image.Image:
    """
    Deskews an image using OpenCV.
    Args:
        image: PIL Image object.
    Returns:
        Deskewed PIL Image object.
    """
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_np, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

def ocr_image(image: Image.Image) -> Dict[str, Any]:
    """
    Performs OCR on a PIL Image and returns words, bounding boxes, and text.
    This uses pytesseract for simplicity. For production, consider more robust OCR solutions.
    Args:
        image: PIL Image object.
    Returns:
        A dictionary containing 'words', 'boxes', and 'full_text'.
        'words': List of detected words.
        'boxes': List of bounding boxes for each word [x_min, y_min, x_max, y_max].
        'full_text': The full extracted text.
    """
    # Get OCR data including bounding box information
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    boxes = []
    full_text_lines = []

    n_boxes = len(data['level'])
    for i in range(n_boxes):
        # Filter out empty words or low confidence detections
        if int(data['conf'][i]) > 60 and data['text'][i].strip():
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            words.append(data['text'][i])
            boxes.append([x, y, x + w, y + h])
            full_text_lines.append(data['text'][i])

    return {
        "words": words,
        "boxes": boxes,
        "full_text": " ".join(full_text_lines)
    }

