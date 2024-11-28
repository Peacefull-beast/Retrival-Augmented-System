from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from PIL import Image
def image_description(image_path):
    
    # Generate description for image
    from prompt import image_description_prompt
    prompt_template = ChatPromptTemplate.from_template(image_description_prompt)
    image = Image.open(image_path)
    prompt = prompt_template.format(image=image)
    model = OllamaLLM(model="llava")
    response = model.invoke(prompt)
    return response

def main():
    print(image_description("extracted_images\\20231002_135907.jpg"))

if __name__ == "__main__":
    main()
