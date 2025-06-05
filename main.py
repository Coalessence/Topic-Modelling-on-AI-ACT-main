import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import html
import json
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

def smart_extend(target_array, item):
    all_arrays = all(isinstance(element, (list, tuple)) for element in item)
    
    if all_arrays and len(item) > 1:
        target_array.extend(item)
    else:
        target_array.append(item)
    
    return target_array

def eli_subdivision_articles(tag):
    if tag.has_attr('class') and 'eli-subdivision' in tag['class']:
        if tag.has_attr('id'):
            if tag['id'].startswith('art'):
                return True
    return False

def div_cpt(tag):
    if tag.name == 'div':
        if tag.has_attr('id'):
            if tag['id'].startswith('cpt'):
                return True
    return False

def div_sec(tag):
    if tag.name == 'div':
        if tag.has_attr('id'):
            if re.fullmatch(r'^cpt_[A-Za-z]+\.sct_\d$', tag['id']):
                return True
    return False

def eli_subdivision_enacting(tag):
    if tag.has_attr('class') and 'eli-subdivision' in tag['class']:
        if tag.has_attr('id'):
            if tag['id'].startswith('enc'):
                return True
    return False

def eli_subdivision_recitals(tag):
    if tag.has_attr('class') and 'eli-subdivision' in tag['class']:
        if tag.has_attr('id'):
            if tag['id'].startswith('rct'):
                return True
    return False

def eli_subdivision_annexes(tag):
    if tag.has_attr('class') and 'eli-container' in tag['class']:
        if tag.has_attr('id'):
            if tag['id'].startswith('anx'):
                return True
    return False

def par_normal(tag):
    if tag.name == 'p':
        if tag.has_attr('class'):
            if 'oj-normal' in tag['class']:
                return True
    return False

def process_element_annex(element, level=1, l1=0, l2=0, l3=0, chapter=1, section=1, article=1, part=3):
    
    total=[]
    
    if element.name == 'p' and element.has_attr('style'):
        text= element.get_text(separator=" ", strip=True)
        if len(text) > 4:
            cleantext = text.strip('\xa0')
            return [part, chapter, section, article, l1, l2, l3, cleantext]
    
    #easiest case like article 4
    if element.name == 'p' and element.has_attr('class'):
        if 'oj-normal' in element.get('class'):
            text= element.get_text(separator=" ", strip=True)
            if len(text) > 4:
                cleantext = text.strip('\xa0')
                return [part, chapter, section, article, l1, l2, l3, cleantext]
        
    if element.name == 'span':
        text = element.get_text(strip=True)
        if len(text) > 5:
            cleantext = text.strip('\xa0')
            return [part, chapter, section, article, l1, l2, l3, cleantext]
    
    if element.name == 'div' and not element.has_attr('id'):
        for pis in element.find_all(['p']):
            temp=process_element_annex(pis, level, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
            if len(temp) > 0:
                total.append(temp)
            l1+=1
            
        
    elif element.name == 'div' and re.match(r'^\d\d\d\.\d\d\d$', element['id']):
        temp=element['id']
        l1= int(temp.split('.')[1])
        
        for pis in element.find_all(['p'], recursive=False):
            temp=process_element_annex(pis, level, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
            if len(temp) > 0:
                total.append(temp)
        
        for tab in element.find_all(['table'], recursive=False):
            match level:
                    case 0:
                        l1+= 1
                    case 1:
                        l2+= 1
                    case 2:
                        l3+= 1
            temp=process_element_annex(tab, level+1, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
            for t in temp:
                if len(t) > 0:
                    total.append(t)
                
            
    if element.name == 'table':
        for td in element.tbody.tr.find_all('td', recursive=False):
            for sub in td.find_all(['p'], recursive=False):
                temp=process_element_annex(sub, level, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
                if len(temp) > 0:
                    total.append(temp)
            
            for sub in td.find_all(['table'], recursive=False):
                
                match level:
                    case 0:
                        l1+= 1
                    case 1:
                        l2+= 1
                    case 2:
                        l3+= 1
                temp=process_element_annex(sub, level+1, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
                for t in temp:
                    if len(t) > 0:
                        total.append(t)
            
            for div in td.find_all(['div'], recursive=False):
                for p in div.find_all(['p'], recursive=False):
                    temp=process_element_annex(p, level, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
                    if len(temp) > 0:
                        total.append(temp)

            for span in td.find_all(['span'], recursive=False):
                temp=process_element_annex(span, level, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
                if len(temp) > 0:
                    total.append(temp)
    
    return total

def process_element(element, level=1, l1=0, l2=0, l3=0, chapter=1, section=1, article=1, part=1):
    
    total=[]
    
    if element.name == 'p' and 'oj-ti-art' in element['class']:
        current_article = element.get_text(strip=True)
    
    #easiest case like article 4
    if element.name == 'p' and 'oj-normal' in element['class']:
        text= element.get_text(separator=" ", strip=True)
        if len(text) > 5:
            cleantext = text.strip('\xa0')
            return [part, chapter, section, article, l1, l2, l3, cleantext]
    
    if element.name == 'div' and not element.has_attr('id'):
        for pis in element.find_all(['p']):
            temp=process_element(pis, level, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
            if len(temp) > 0:
                total.append(temp)
            l1+=1
            
        
    elif element.name == 'div' and re.match(r'^\d\d\d\.\d\d\d$', element['id']):
        temp=element['id']
        l1= int(temp.split('.')[1])
        
        for pis in element.find_all(['p'], recursive=False):
            temp=process_element(pis, level, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
            if len(temp) > 0:
                total.append(temp)
        
        for tab in element.find_all(['table'], recursive=False):
            match level:
                    case 0:
                        l1+= 1
                    case 1:
                        l2+= 1
                    case 2:
                        l3+= 1
            temp=process_element(tab, level+1, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
            for t in temp:
                if len(t) > 0:
                    total.append(t)
                
            
    if element.name == 'table':
        for td in element.tbody.tr.find_all('td', recursive=False):
            for sub in td.find_all(['p'], recursive=False):
                temp=process_element(sub, level, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
                if len(temp) > 0:
                    total.append(temp)
            
            for sub in td.find_all(['table'], recursive=False):
                
                match level:
                    case 0:
                        l1+= 1
                    case 1:
                        l2+= 1
                    case 2:
                        l3+= 1
                temp=process_element(sub, level+1, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
                for t in temp:
                    if len(t) > 0:
                        total.append(t)
            
            for div in td.find_all(['div'], recursive=False):
                for p in div.find_all(['p'], recursive=False):
                    temp=process_element(p, level, l1=l1, l2=l2, l3=l3, chapter=chapter, section=section, article=article, part=part)
                    if len(temp) > 0:
                        total.append(temp)
    
    return total
        

def get_data_from_html(html_content):
    print("Extracting paragraphs from HTML content...")
    total=[]
    soup = BeautifulSoup(html_content, 'html.parser')

    #handle enacting
    artn=0
    chapters = soup.find(eli_subdivision_enacting).find_all(div_cpt, recursive=False)
    cptn=0
    for chapter in chapters:

        cptn+=1
        secn=0
        sections=chapter.find_all(div_sec, recursive=False)
        
        if len(sections) == 0:
            sections = [chapter]
        
        for section in sections:
            secn+=1
            divs= section.find_all(eli_subdivision_articles, recursive=False)
            for div in divs:
                artn+=1
                l1=0
                for element in div.find_all(['p', 'div', 'table'], recursive=False):
                    if element.get('class'):
                        if 'oj-ti-art' in element['class'] or 'eli-title' in element['class']:
                            #print("Skipping title element:", element.get_text(strip=True))
                            continue
                    
                    l1+=1
                    temp=process_element(element, 1, l1,chapter=cptn, section=secn, article=artn, part=2)
                    if isinstance(temp[0], list):
                        for t in temp:
                            total.append(t)
                    else:
                        total.append(temp)
                                        

    
    #handle recitals
    recn=0
    recitals = soup.find_all(eli_subdivision_recitals)
    for recital in recitals:
        divs = recital.find_all('table', recursive=False)
        recn+=1
        for div in divs:
            total.extend(process_element(div, 0, l1=1, l2=0, l3=0, chapter=0, section=0, article=recn, part=1))
        
    #handle annexes
    annn=0
    annexes = soup.find_all(eli_subdivision_annexes)
    for annex in annexes:
        annn+=1
        l1=0
        for element in annex.find_all(['p', 'div', 'table'], recursive=False):
            if element.get('class'):
                if 'oj-doc-ti' in element['class'] or 'oj-ti-grseq-1' in element['class']:
                    #print("Skipping title element:", element.get_text(strip=True))
                    continue
            
            l1+=1
            temp=process_element_annex(element, 1, l1=l1, l2=0, l3=0, chapter=0, section=0, article=annn, part=3)
            if isinstance(temp[0], list):
                for t in temp:
                    total.append(t)
            else:
                total.append(temp)
        

    return total

        
def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['part', 'chapter', 'section', 'article', 'l1', 'l2', 'l3', 'text'])
    df[['text']]= df[['text']].astype(str)
    df.to_csv(filename, index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Data saved to {filename}")
    return df
    
def load_data():
    ds = pd.read_csv('data.csv')
    return ds

def get_text(data):
    return data['text'].tolist()
    
def get_part(data, partn):
    if partn not in [1, 2, 3]:
        raise ValueError("Part number must be 1, 2, or 3")
    return data[data['part'] == partn].reset_index(drop=True)

def get_l1_grouped(data):
    """
    Group the data by 'l1' and return a DataFrame with 'l1' as index and 'text' as values.
    """
    grouped = data.groupby(['part','article','l1'])['text'].apply(lambda x: ' '.join(x)).reset_index()
    grouped.rename(columns={'text': 'text'}, inplace=True)
    return grouped

def get_article_grouped(data):
    """
    Group the data by 'article' and return a DataFrame with 'article' as index and 'text' as values.
    """
    grouped = data.groupby(['part','article'])['text'].apply(lambda x: ' '.join(x)).reset_index()
    grouped.rename(columns={'text': 'text'}, inplace=True)
    return grouped

def get_data():
    if os.path.exists("data.csv"):
        print("Loading data from CSV file...")
        return load_data()
    else:
        print("Fetching data from URL...")
        data = get_data_from_html(get_html("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"))
        return save_to_csv(data, "data.csv")
    
    
def bert_ex(data):
    bert= BertEx(data)
    print("BERTopic model fitted.")
    bert.get_tab_data().to_csv("bert_topics.csv", index=False)
    plot=bert.get_graph_data(data)
    plot.write_html("bert_plot.html")
        
def ldaf(data):
    lda= LDA(data)
    lda.plot_top_words("LDA Topics")

def language_model(data, model_name="gemma3:27b"):
    lm = LanguageModel(model_name)
    print("Chat generated for language model.")
    text = lm.perform_topic_modeling(data)
    with open("language_model_output.json", "w") as f:
        json.dump(text, f, indent=4)
    print("Generated text:", text)
    
def main():
    load_dotenv()
    login(token=os.getenv("HuggingFace_API_KEY"))

    data = get_data()
    textdata= get_text(data)
    
    language_model(textdata)
    
        
if __name__ == "__main__":
    main()
    
    
#include recitals
#improve the dataset and do statistic, anverage lenght of number phargraph