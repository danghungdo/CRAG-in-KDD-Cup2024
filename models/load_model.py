from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from dotenv import load_dotenv
import os 

def load_model(model_name="gpt-4o", api_key=None, base_url=None, **kwargs):
    if api_key is None and base_url is None:
        model = ChatOpenAI(model_name=model_name, **kwargs)
    elif api_key is not None and base_url is None:
        model = ChatOpenAI(model_name=model_name, api_key=api_key, **kwargs)
    elif api_key is None and base_url is not None:
        model = ChatOpenAI(model_name=model_name, base_url=base_url, **kwargs)
    else:
        model = ChatOpenAI(model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)
    return model

def load_model_ollama(model_name="llama3", **kwargs):
    model = Ollama(model=model_name, **kwargs)
    return model

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("INTERWEB_APIKEY")
    # base_url = "<your-base-url>"
    base_url = "https://interweb.l3s.uni-hannover.de"
    # base_url = "http://gpunode04.kbs:11434/v1/"
    model_name = "llama3.3:70b"
    # model_name= "gpt-4o"
    chat_model = load_model(model_name=model_name, api_key=api_key, base_url=base_url, temperature=0)
    messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
    ai_msg = chat_model.invoke(messages)
    print(ai_msg.content)