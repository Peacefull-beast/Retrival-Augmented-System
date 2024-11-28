import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from get_embedding_function import get_image_embedding
from query_exp import rewrite_query
from pprint import pprint
from path import CHROMA_PATH_IMAGE
from describe_image import image_description
from PIL import Image
import io
from query_text import query_text

def retrieve_image(query_path): #check if the image is in the correct format
    # Step 2: Prepare the context
    embedding_function = get_image_embedding()
    db = Chroma(persist_directory=CHROMA_PATH_IMAGE, embedding_function=embedding_function)

    query_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    query_docs = query_retriever.invoke(query_path)
    
    return query_docs



def query_image(image_path):
    #Step 1: generate description of the image
    image_des = image_description(image_path)
    print(image_des)

    #Step 2 : retrive documents based on the description of the image from db1
    image_description_docs = query_text(image_des)

    #Step 3: retrieve documents based on image 
    image = Image.open(image_path)      #check if the image is in the correct format
    image_docs = retrieve_image(image_path)

    return image_description_docs, image_docs
