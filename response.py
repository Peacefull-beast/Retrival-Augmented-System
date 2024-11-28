from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from query_image import query_image
from query_text import query_text
from prompt import prompt_template

model_name ="mistral" #coming from gui
model = OllamaLLM(model=model_name)

def generate_response(querytext, query_image_path):

    #Step 1: retrive text docs based on text to text retrival from database 1
    if querytext:
        text_docs = query_text(querytext)
    else: 
        text_docs = []

    if query_image_path:
        image_description_docs, image_docs = query_image(query_image_path)

    else:
        image_description_docs = []
        image_docs = []

    #Step 2: combine all the docs
    all_docs = []
    print(len(text_docs), len(image_docs), len(image_description_docs))
    all_docs.extend(text_docs)
    all_docs.extend(image_docs)
    all_docs.extend(image_description_docs)

    #Step 3: prepare the context
    context_text = "\n".join([doc.page_content for doc in all_docs])
    context_sources = []


    #Step 4: prepare the context sources
    for doc in all_docs:
        file_path = doc.metadata["source"]
        page_number = doc.metadata["page_number"]
        context_sources.append(f"{file_path} - Page {page_number}")
    
    
    #Step 5: prepare the prompt
    chat_prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = chat_prompt.format(context=context_text, query=querytext)
    
    #Step 6: generate the response
    response_text = model.invoke(prompt)
    
    #Step 7: prepare the output
    sources = [doc.metadata["id"] for doc in all_docs]
    output = {
        "query_text": query_text,
        "sources": sources,
        "response": response_text.strip()
    }
    
    return output["response"],context_sources


def main():
    """
    Example usage of the generate_response function.
    Replace the `query_question` and `query_image_path` with actual inputs as needed.
    """
    query_question = "Are muslims terrorists?" # Text from GUI or user input
    query_image_path = ""  # Image path from GUI or user input (if any)
    response, context_sources = generate_response(query_question, query_image_path)
    
    print(response)
    print(context_sources)


if __name__ == "__main__":
    main()
