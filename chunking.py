from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from typing import List
from langchain.schema import Document

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of Document objects to split
        
    Returns:
        List of split Document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: List[Document], db) -> None:
    """Add new document chunks to Chroma DB if they don't already exist.
    
    Args:
        chunks: List of Document chunks to potentially add
        db: Chroma database instance
    """
    if not chunks:
        print("No chunks provided")
        return
        
    chunks_with_ids = calculate_chunk_ids(chunks)
    try:
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])

        new_chunks = [
            chunk for chunk in chunks_with_ids 
            if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new documents to add")
    except Exception as e:
        print(f"Error adding documents to database: {str(e)}")

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """Add unique IDs to document chunks' metadata.
    
    Args:
        chunks: List of Document chunks
        
    Returns:
        List of Document chunks with added IDs in metadata
    """
    for chunk in chunks:
        chunk.metadata["id"] = str(uuid.uuid4())
    return chunks
   
