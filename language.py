import json
import requests
from typing import List, Dict, Any

class LanguageModel:
    def __init__(self, model_name: str = "llama3.3:70-128k", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def create_topic_modeling_prompt(self, texts: List[str]) -> str:
        numbered_texts = "\n".join([f"{i}: {text}" for i, text in enumerate(texts)])
        
        prompt = f"""
        Analyze the following texts and perform topic modeling. Group similar texts into topics and return the results in JSON format.

        Texts to analyze:
        {numbered_texts}

        Please identify the main 7 topics in these texts and return the results in the following JSON format:
        {{
            "topic_1": {{
                "top_5_words": "comma-separated list of top 5 words for this topic",
                "number_of_paragraphs": number_of_texts_in_this_topic,
                "index": [list_of_text_indices_for_this_topic]
            }},
            "topic_2": {{
                "top_5_words": "comma-separated list of top 5 words for this topic",
                "number_of_paragraphs": number_of_texts_in_this_topic,
                "index": [list_of_text_indices_for_this_topic]
            }}
        }}

        Rules:
        - Group texts with similar themes/topics together
        - Identify the most representative words for each topic
        - Include the indices of texts that belong to each topic
        - Return only valid JSON, no additional explanation
        """
        return prompt
    
    def query_ollama(self, prompt: str) -> str:

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "seed": 42,
                "num_ctx": 65535
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error querying Ollama: {e}")
    
    def parse_topic_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response from the model
        
        Args:
            response: Raw response string from the model
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Clean the response - sometimes models add extra text
            response = response.strip()
            
            # Find JSON content between braces
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
                
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {e}\nResponse: {response}")
    
    def perform_topic_modeling(self, texts: List[str]) -> Dict[str, Any]:
        """
        Perform complete topic modeling process
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary containing topic modeling results
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        print(f"Analyzing {len(texts)} texts for topic modeling...")
        
        # Create prompt
        prompt = self.create_topic_modeling_prompt(texts)
        
        # Query model
        print("Querying Ollama model...")
        response = self.query_ollama(prompt)
        
        # Parse response
        print("Parsing response...")
        topics = self.parse_topic_response(response)
        
        return topics
    
    def print_results(self, topics: Dict[str, Any]):
        """
        Pretty print the topic modeling results
        
        Args:
            topics: Dictionary containing topic results
        """
        print("\n" + "="*50)
        print("TOPIC MODELING RESULTS")
        print("="*50)
        
        for topic_name, topic_data in topics.items():
            print(f"\n{topic_name.upper().replace('_', ' ')}:")
            print(f"  Top 5 words: {topic_data.get('top_5_words', 'N/A')}")
            print(f"  Number of paragraphs: {topic_data.get('number_of_paragraphs', 'N/A')}")
            print(f"  Paragraph indices: {topic_data.get('index', 'N/A')}")