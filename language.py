import torch
from transformers import pipeline

class LanguageModel:
    def __init__(self, model_name=""):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline("text-generation", model=self.model_name, torch_dtype=torch.bfloat16, device_map="auto" ,use_auth_token=True)

    def generate_chat(self, data):
        new_data = []
        for index, item in enumerate(data):
            new_data.append({
                "role": "user",
                "content": f"Paragraph {index}: {item}"
            })
        
        chat = [{
            "role": "system",
            "content": """You are a topic model that groups paragraphs into topics based on their content. Separate the following paragraphs into 10 topics and return the topics in JSON format.

            Format example:
            {
                "topic_1": {
                    "top_5_words": "top 5 words of topic 1",
                    "number_of_paragraphs": "number of paragraphs in topic 1",
                    "index": "index of paragraphs of topic 1"
                },
                "topic_2": {
                    "top_5_words": "top 5 words of topic 2", 
                    "number_of_paragraphs": "number of paragraphs in topic 2",
                    "index": "index of paragraphs of topic 2"
                }
            }"""
        }] + new_data
    
        return chat
    
    def generate_text(self, max_length=100, data=None):
        prompt = self.generate_chat(data)
        
        return self.pipe(prompt, max_new_tokens=max_length)[0]['generated_text'][-1]["content"]