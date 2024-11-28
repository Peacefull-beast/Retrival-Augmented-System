from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


re_write_llm = Ollama(model="mistral")

# Create a prompt template for query rewriting
query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=query_rewrite_template
)

# Create an LLMChain for query rewriting
query_rewriter = query_rewrite_prompt | re_write_llm

def rewrite_query(original_query):
    response = query_rewriter.invoke(original_query)
    return response


