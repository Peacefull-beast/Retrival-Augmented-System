from langchain.schema.document import Document
from PIL import Image


def structured_data(text_data, image_description, image_data):
    """Converts data into a structured format."""
    text_documents = []
    image_documents = []
    
    # Process text data
    for text in text_data:
        # Check if the text is using 'text' or 'response' key
        content = text.get('text') or text.get('response')
        file_path = text.get('file_path') or text.get('name')
        page_number = text.get('page_number')
        
        if content and file_path:
            text_documents.append(Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "tag": "text",
                    "page_number": page_number
                }
            ))

    # Process image descriptions
    for image in image_description:
        content = image.get('text') or image.get('response')
        file_path = image.get('file_path') or image.get('name')
        page_number = image.get('page_number')
        if content and file_path:
            text_documents.append(Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "tag": "image",
                    "page_number": page_number
                }
            ))

    # Process image data
    for image in image_data:
        image_path = image.get('image_path') or image.get('path')
        file_path = image.get('file_path') or image.get('name')
        
        if image_path and file_path:
            try:
                image_documents.append(Document(
                    page_content=image_path,
                    metadata={
                        "source": file_path,
                        "tag": "image",
                        "page_number": page_number
                    }
                ))
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue

    return text_documents, image_documents
