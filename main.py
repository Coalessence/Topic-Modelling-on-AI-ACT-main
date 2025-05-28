import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import html
import re
import csv
from bertEx import BertEx
from LDA import LDA
from language import LanguageModel
from huggingface_hub import login
from dotenv import load_dotenv

def get_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def get_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return ""
    
def get_paragraph_from_html(html_content):
    print("Extracting paragraphs from HTML content...")
    total=[]
    shortsummer=[]
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all("p", {"class": "oj-normal"})
    
    for div in paragraphs:
        # Get plain text, strip extra whitespace
        text = div.get_text(strip=True)
        # Clean up multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        if len(text) > 5:
            """ if text[0] == text[0].lower() and not text[0].isdigit():
                shortsummer.append(text)
            else:
                if(shortsummer):
                    total.append(" ".join(shortsummer))
                    print("Short summary added:", " ".join(shortsummer))
                    shortsummer = [] """
            total.append(text)
    
    return total
        
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=False, quotechar='"', quoting=csv.QUOTE_ALL)
    print(f"Data saved to {filename}")
    
def load_data():
    ds = pd.read_csv('data.csv', header=0)
    data = []
    for i in range(len(ds.values)):
        data.append(ds.values[i][0])
        
    return data

def get_data():
    if os.path.exists("data.csv"):
        print("Loading data from CSV file...")
        return load_data()
    else:
        print("Fetching data from URL...")
        data = get_paragraph_from_html(get_html("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"))
        save_to_csv(data, "data.csv")
        return data
    
def bert_ex(data):
    bert= BertEx(data)
    print("BERTopic model fitted.")
    bert.get_tab_data().to_csv("bert_topics.csv", index=False)
    plot=bert.get_graph_data(data)
    plot.write_html("bert_plot.html")
        
def ldaf(data):
    lda= LDA(data)
    lda.plot_top_words("LDA Topics")

def language_model(data):
    lm = LanguageModel(model_name="meta-llama/Llama-3.3-70B-Instruct")
    print("Chat generated for language model.")
    text = lm.generate_text(max_length=512, data=data)
    print("Generated text:", text)
    
def main():
    load_dotenv()
    login(token=os.getenv("HuggingFace_API_KEY"))
    data = get_data()
    
    
    
    
    
    
if __name__ == "__main__":
    main()