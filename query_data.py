import argparse
from langchain.prompts import ChatPromptTemplate
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
#from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from query_exp import rewrite_query
from prompt import FEW_SHOT_PROMPT_TEMPLATE
from pprint import pprint
from path import CHROMA_PATH
from get_embedding_function import get_text_embedding



def main():
    query_text = "When did the defect occur?"
    print("Query text:", query_text)
    data = query_rag(query_text)
    return data

def query_rag(query_text: str):
    # Step 1: Expand the query using the Ollama model
    expanded_query = rewrite_query(query_text)

    # Step 2: Prepare the context
    embedding_function = get_text_embedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Step 3: Retrieve documents based on the expanded query (base retrieval)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    context_docs = retriever.invoke(expanded_query)
    
    # Step 4: Contextual compression with embeddings filter
    embeddings_filter = EmbeddingsFilter(embeddings=embedding_function, similarity_threshold=0.5)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    
    # Retrieve relevant documents based on the compressed context
    compressed_docs = compression_retriever.invoke(expanded_query)
    
    # Step 5: Retrieve documents based on the original query (direct retrieval)
    query_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    query_docs = query_retriever.invoke(query_text)
    
    # Step 6: Intersect the retrieved document IDs
    context_doc_ids = {doc.metadata["id"] for doc in context_docs}
    query_doc_ids = {doc.metadata["id"] for doc in query_docs}
    intersected_doc_ids = context_doc_ids.intersection(query_doc_ids)
    
    # Step 7: Re-rank the relevant documents based on the query
    final_retrieved_docs = [
        doc for doc in compressed_docs if doc.metadata["id"] in intersected_doc_ids
    ]
    
    # Step 8: Prepare the context from the retrieved and compressed documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc in final_retrieved_docs])
    
    # Dynamic few-shot prompting with context
    prompt_template = ChatPromptTemplate.from_template(FEW_SHOT_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Step 9: Use Ollama model for generating a response
    model = OllamaLLM(model="phi3.5")
    response_text = model.invoke(prompt)
    
    # Prepare the sources for output
    sources = [doc.metadata.get("id", None) for doc in final_retrieved_docs]
    data = {
        "query_text": query_text,
        "sources": sources,
        "Response": response_text.strip()
    }
    return data


if __name__ == "__main__":
    pprint(main())