# preprocess.py

import os
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Any
import fitz  # PyMuPDF
from PIL import Image, ImageStat
import numpy as np
import cv2
from pdf2image import convert_from_path
import tempfile
import json
from path import EMBEDDED_IMAGE_PATH, PAGES


from describe_image import image_description  # Ensure this is correctly implemented

def preprocess(file_path: str, poppler_path: str = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Preprocess a PDF file to extract text, images, and image descriptions.

    Args:
        file_path (str): Path to the PDF file.
        poppler_path (str, optional): Path to the Poppler 'bin' directory (required for Windows).

    Returns:
        Tuple containing lists of text data, image data, and image descriptions.
    """
    text_data = []
    img_data = []
    image_descriptions = []

    try:
        # Define output directories
        extracted_images_dir = Path(EMBEDDED_IMAGE_PATH)
        page_images_dir = Path(PAGES)
        page_images_dir.mkdir(parents=True, exist_ok=True)
        extracted_images_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing file: {file_path}")

        # Open PDF with PyMuPDF
        with fitz.open(file_path) as pdf_file:
            for page_number in range(len(pdf_file)):
                page = pdf_file[page_number]

                # Extract and clean text
                text = page.get_text().strip()
                cleaned_text = clean_text(text)
                if cleaned_text:
                    text_data.append({
                        "text": cleaned_text,
                        "file_path": file_path,
                        "page_number": page_number + 1
                    })
                    print(f"Extracted text from page {page_number + 1}")

                # Extract embedded images
                images = page.get_images(full=True)
                for image_index, img in enumerate(images, start=1):
                    try:
                        xref = img[0]
                        base_image = pdf_file.extract_image(xref)

                        if is_valid_image(base_image):
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]

                            # Define image path
                            image_filename = f"embedded_{page_number + 1}_{image_index}.{image_ext}"
                            image_path = extracted_images_dir / image_filename

                            # Save embedded image
                            with open(image_path, "wb") as image_file:
                                image_file.write(image_bytes)
                            print(f"Saved embedded image: {image_path}")

                            # Check if image is blank
                            if is_blank_image(str(image_path)):
                                print(f"Embedded image {image_path} is blank. Skipping description.")
                                continue

                            # Generate description for embedded image
                            try:
                                embedded_desc = image_description(str(image_path))
                                if embedded_desc:
                                    img_data.append({
                                        "image_path": str(image_path),
                                        "file_path": file_path,
                                        "page_number": page_number + 1
                                    })
                                    image_descriptions.append({
                                        "text": embedded_desc,
                                        "file_path": str(image_path),
                                        "page_number": page_number + 1
                                    })
                                    print(f"Generated description for embedded image: {image_path}")
                            except Exception as e:
                                print(f"Error generating description for embedded image {image_path}: {e}")

                    except Exception as e:
                        print(f"Error processing embedded image {image_index} on page {page_number + 1}: {e}")
                        continue

        return text_data, img_data, image_descriptions

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [], [], []


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing unnecessary symbols and normalizing whitespace.

    Args:
        text (str): The extracted text.

    Returns:
        str: Cleaned text.
    """
    try:
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)

        # Replace non-ASCII characters with closest ASCII equivalent
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Remove special characters but keep basic punctuation
        cleaned = re.sub(r'[^\w\s.,!?-]', ' ', text)

        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\.{2,}', '.', cleaned)
        cleaned = cleaned.strip()

        # Remove any lines that are just numbers or single characters
        lines = cleaned.split('\n')
        cleaned_lines = [
            line.strip()
            for line in lines
            if len(line.strip()) > 1 and not line.strip().isdigit()
        ]

        return '\n'.join(cleaned_lines)

    except Exception as e:
        print(f"Error cleaning text: {e}")
        # Return original text if cleaning fails
        return text.encode('ascii', 'ignore').decode('ascii')


def is_valid_image(base_image: Dict[str, Any]) -> bool:
    """
    Validate if the extracted image meets the criteria.

    Args:
        base_image (Dict[str, Any]): The image data extracted from PDF.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        # Implement actual validation logic
        # Example: Check image dimensions
        width = base_image.get("width", 0)
        height = base_image.get("height", 0)
        if width < 50 or height < 50:
            print(f"Image too small: {width}x{height}")
            return False

        return True
    except Exception as e:
        print(f"Error validating image: {e}")
        return False


def is_blank_image(image_path: str) -> bool:
    """
    Check if the image is mostly white or black.

    Args:
        image_path (str): Path to the image file.

    Returns:
        bool: True if the image is blank, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale
            img = img.convert('L')
            stat = ImageStat.Stat(img)
            mean_value = stat.mean[0]

            # Define thresholds
            if mean_value < 10 or mean_value > 245:
                print(f"Image {image_path} is blank (mean: {mean_value})")
                return True
            return False

    except Exception as e:
        print(f"Error checking if image is blank {image_path}: {e}")
        return False


def load_documents(data_path: str, poppler_path: str = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load and process all PDF documents in the specified directory.

    Args:
        data_path (str): Path to the directory containing PDF files.
        poppler_path (str, optional): Path to the Poppler 'bin' directory (required for Windows).

    Returns:
        Tuple containing lists of all text data, image data, and image descriptions.
    """
    all_text_data = []
    all_img_data = []
    all_image_descriptions = []

    if not os.path.exists(data_path):
        print(f"Directory not found: {data_path}")
        return [], [], []

    for file in os.listdir(data_path):
        if file.lower().endswith('.pdf'):
            file_path = os.path.join(data_path, file)
            print(f"\nProcessing file: {file}")
            text, images, descriptions = preprocess(file_path, poppler_path=poppler_path)
            all_text_data.extend(text)
            all_img_data.extend(images)
            all_image_descriptions.extend(descriptions)

    return all_text_data, all_img_data, all_image_descriptions


def convert_pdf_to_images(pdf_path: str, output_dir: Path, dpi: int = 300, poppler_path: str = None) -> List[Path]:
    """
    Convert PDF pages to high-resolution images optimized for OCR and visual clarity.

    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (Path): Directory to save the converted images.
        dpi (int, optional): Resolution for the output images. Defaults to 300.
        poppler_path (str, optional): Path to the Poppler 'bin' directory (required for Windows).

    Returns:
        List[Path]: List of paths to the generated images.
    """
    image_paths = []

    try:
        # Convert PDF pages to images
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            output_folder=str(tempfile.gettempdir()),
            fmt="png",
            thread_count=4,
            grayscale=False,
            size=None,  # Maintain original size
            use_pdftocairo=True,  # Better quality for text
            poppler_path=poppler_path  # Specify Poppler path if needed
        )

        pdf_name = Path(pdf_path).stem

        for i, image in enumerate(images, start=1):
            try:
                # Enhance image quality
                enhanced_image = enhance_image_quality(image)

                # Define output image path
                output_filename = f"{pdf_name}_page_{i}.png"
                output_path = output_dir / output_filename

                # Save the enhanced image
                enhanced_image.save(
                    output_path,
                    "PNG",
                    dpi=(dpi, dpi),
                    quality=95
                )
                image_paths.append(output_path)
                print(f"Converted page {i} to image: {output_path}")

            except Exception as e:
                print(f"Error enhancing/saving image for page {i}: {e}")
                continue

        return image_paths

    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return image_paths


def enhance_image_quality(image: Image.Image) -> Image.Image:
    """
    Enhance image quality for better OCR and visibility.

    Args:
        image (PIL.Image.Image): PIL Image object.

    Returns:
        PIL.Image.Image: Enhanced PIL Image object.
    """
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)

        # Convert RGB to BGR
        if img_array.ndim == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Denoise the image
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

        # Convert to LAB color space for contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge channels back
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Sharpen the image
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Convert back to RGB for PIL
        final_image = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

        return Image.fromarray(final_image)

    except Exception as e:
        print(f"Error enhancing image quality: {e}")
        return image  # Return original image if enhancement fails


def main():
    """
    Main execution function.
    """
    try:
        poppler_path = None
        if os.name == 'nt':  # If Windows
            poppler_path = r"C:\path\to\poppler\bin"  # Update this path accordingly

        data_path = "data1"  # Update as per your directory structure

        # Load and process all documents
        all_text_data, all_img_data, all_image_descriptions = load_documents(data_path, poppler_path=poppler_path)

        # Define output directory and file
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "output_data.json"

        # Prepare data for JSON serialization
        output_data = {
            "all_text_data": all_text_data,
            "all_image_data": all_img_data,
            "all_image_descriptions": all_image_descriptions
        }

        # Write results to JSON file
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(output_data, file, ensure_ascii=False, indent=4)

        print(f"✅ Data has been written to {output_file}")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")


if __name__ == "__main__":
    main()
