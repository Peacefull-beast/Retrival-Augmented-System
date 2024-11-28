import os
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_text_embedding, get_image_embedding
from preprocess_file import load_documents
from chunking import split_documents, add_to_chroma
from path import DATA_PATH, CHROMA_PATH_TEXT, CHROMA_PATH_IMAGE
from make_data import structured_data

def clear_database():
    """Clear existing databases if they exist"""
    try:
        if os.path.exists(CHROMA_PATH_TEXT):
            print("Clearing existing text database...")
            Chroma(persist_directory=CHROMA_PATH_TEXT).delete_collection()
        if os.path.exists(CHROMA_PATH_IMAGE):
            print("Clearing existing image database...")
            Chroma(persist_directory=CHROMA_PATH_IMAGE).delete_collection()
    except Exception as e:
        print(f"Error clearing databases: {str(e)}")

def main():
    try:
        clear_database()
        print("Processing documents...")
        text_data, image_data, image_description = load_documents(DATA_PATH)
        print("Documents processed")

        # Initialize Chroma databases with appropriate embedding functions
        dbv1 = Chroma(
            persist_directory=CHROMA_PATH_TEXT, 
            embedding_function=get_text_embedding()
        )

        dbv2 = Chroma(
            persist_directory=CHROMA_PATH_IMAGE, 
            embedding_function=get_image_embedding()
        )

        # Structure the data
        text_docs, image_docs = structured_data(text_data, image_description, image_data)
        print(text_docs[0].metadata)
        print("Data structured")
        text_chunks = split_documents(text_docs)
        image_chunks = split_documents(image_docs)

        # print(text_chunks)
        # Add chunks to respective Chroma DBs
        print("Adding text chunks to database...")
        add_to_chroma(text_chunks, dbv1)
        
        print("Adding image chunks to database...")
        add_to_chroma(image_chunks, dbv2)
        
        print("✅ Database population completed successfully")
        
    except Exception as e:
        print(f"❌ Error during database population: {str(e)}")
        raise

if __name__ == "__main__":
    main()


#data cleaning to be done
#promt engineering
#clean image extraction
#ollama models context length moderation
#meta id : replace random strings with logical ids

