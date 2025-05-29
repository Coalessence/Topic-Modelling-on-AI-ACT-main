import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LanguageModel:
    def __init__(self, model_name=""):
        self.model_name = model_name
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            
        print(f"Using device: {torch.device}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name,  torch_dtype=torch.bfloat16,use_auth_token=True)
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left") 
        
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}{% endif %}{% endfor %}Assistant:"
        
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
        message = self.generate_chat(data)
        
        tokenized_chat = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(tokenized_chat, max_new_tokens=512) 
        print(self.tokenizer.decode(outputs[0]))