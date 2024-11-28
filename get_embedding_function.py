import torch
import clip
from PIL import Image
from langchain_ollama import OllamaEmbeddings
import numpy as np

def get_text_embedding():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings



class CLIPImageEmbedding:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    
    def embed_documents(self, image_paths):
        """Embed a list of image paths"""
        embeddings = []
        for image_path in image_paths:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                # Convert to numpy and normalize
                embedding = image_features.cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, image_path):
        """Embed a single image path"""
        return self.embed_documents([image_path])[0]

def get_image_embedding():
    """Initialize and return CLIP embedding model for images"""
    return CLIPImageEmbedding()


