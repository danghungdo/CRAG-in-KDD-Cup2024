import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from openai import OpenAI

class SequenceClassificationRouter:
    def __init__(self, model_path, classes, device_map="auto", peft_path=None, use_bits_and_bytes=False, use_peft=False):
        if use_bits_and_bytes:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                device_map=device_map,
                num_labels=len(classes),
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                device_map=device_map,
                num_labels=len(classes),
                torch_dtype=torch.bfloat16,
            )
        if use_peft:
            assert peft_path is not None, "PEFT path is required."
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_path
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.classes = classes
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax().item()
        predicted_class = self.classes[predicted_class_idx]
        return predicted_class
    
    
class OpenAISequenceClassifier:
    def __init__(self, openai_api_key, classes, model="gpt-4"):
        """
        Initialize the OpenAISequenceClassifier.

        Args:
            openai_api_key (str): Your OpenAI API key.
            classes (list): List of target classes for classification.
            model (str): OpenAI model to use (default is "gpt-4").
        """
        self.api_key = openai_api_key
        self.llm = OpenAI(
            base_url="https://interweb.l3s.uni-hannover.de",
            api_key=self.api_key,
        )
        self.classes = classes
        self.model = model

    def __call__(self, text):
        """
        Perform classification using the OpenAI API.

        Args:
            text (str): The input text to classify.

        Returns:
            str: Predicted class label.
        """
        # print(text)
        prompt = (
            f"You are a classification assistant. Your task is to classify the input text "
            f"into one of the following domains: {self.classes}. The output should strictly "
            f"match one of these domains, with no additional explanation or formatting.\n\n"
            f"Here is the text:\n"
            f"\"{text}\"\n"
            f"Domain:"
        )
        # try:
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Deterministic output
        )
        predicted_class = response.choices[0].message.content
        return predicted_class
        #     print(f"Predicted class: {predicted_class}")
        #     if predicted_class in self.classes:
        #         print(f"Predicted class: {predicted_class}")
        #         return predicted_class
        #     else:
        #         return "Unknown"  # Handle unexpected outputs gracefully
        # except Exception as e:
        #     return f"Error: {e}"


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    # Define the classes for classification
    classes = ["open", "movie", "music", "sports", "finance"]

    # Initialize the SequenceClassificationRoute
    api_key = os.getenv("INTERWEB_APIKEY")
    # Initialize the OpenAISequenceClassifier
    domain_router = OpenAISequenceClassifier(api_key, ["finance", "music", "movie", "sports", "open"], model="llama3.1:8b-instruct-q8_0")
    
    text = "name three celebrities who have been involved in successful collaborations with beauty brands."
    
    print(domain_router(text))