from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from get_embedding_function import get_text_embedding
from query_exp import rewrite_query
from path import CHROMA_PATH_TEXT




def query_text(query_text: str):
    # Step 1: Expand the query using the Ollama model if text is there
    if query_text:
        expanded_query = rewrite_query(query_text)
    

    # Step 2: Prepare the context
    embedding_function_text = get_text_embedding()
    db = Chroma(persist_directory=CHROMA_PATH_TEXT, embedding_function=embedding_function_text)
    
    # Step 3: Retrieve documents based on the expanded query (base retrieval)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    context_docs = retriever.invoke(expanded_query)
    
    # Step 4: Contextual compression with embeddings filter
    embeddings_filter = EmbeddingsFilter(embeddings=embedding_function_text, similarity_threshold=0.5)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    
    # Retrieve relevant documents based on the compressed context
    compressed_docs = compression_retriever.invoke(expanded_query)
    
    # Step 5: Retrieve documents based on the original query (direct retrieval)
    query_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})#ADJUST K
    query_docs = query_retriever.invoke(query_text)
    
    # Step 6: Intersect the retrieved document IDs
    context_doc_ids = {doc.metadata["id"] for doc in context_docs}
    query_doc_ids = {doc.metadata["id"] for doc in query_docs}
    intersected_doc_ids = context_doc_ids.union(query_doc_ids)  #replaced intersection with union
    
    # Step 7: Re-rank the relevant documents based on the query
    final_retrieved_docs = [
        doc for doc in compressed_docs if doc.metadata["id"] in intersected_doc_ids
    ]
    
    return final_retrieved_docs