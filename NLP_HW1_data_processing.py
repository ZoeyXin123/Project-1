import io
from ratelimit import limits, sleep_and_retry
import pandas as pd
from bs4 import BeautifulSoup
import requests
import pprint
from tqdm import tqdm
import re
import json
from datetime import datetime, date
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

cik_df = pd.read_csv("/Users/maryw/Documents/mary/master/Courses/46924 NLP/HW1/sp500_w_addl_id_with_cik.csv").dropna()[["ticker", "cik", "start", "ending"]]

start = date(2017, 1, 1)
end = date(2021, 12, 31)

cik_lookup = {}
for idx,row in cik_df.iterrows():
    cik = str(int(row["cik"]))
    cik_lookup[row["ticker"]] = (10 - len(cik)) * "0" + cik

HEADER = {'Host': 'www.sec.gov', 'Connection': 'close',
          'Accept': 'application/json, text/javascript, */*; q=0.01', 
          'X-Requested-With': 'XMLHttpRequest',
          'User-Agent': 'xiaolanw@andrew.cmu.edu'
         }

class SecAPI(object):
    SEC_CALL_LIMIT = {'calls': 10, 'seconds': 1}

    @staticmethod
    @sleep_and_retry
    # Dividing the call limit by half to avoid coming close to the limit
    @limits(calls=SEC_CALL_LIMIT['calls'] / 2, period=SEC_CALL_LIMIT['seconds'])
    def _call_sec(url):
        return requests.get(url, headers=HEADER)

    def get(self, url):
        return self._call_sec(url).text

def get_sec_data(cik, doc_type, start=20220101, count=20):
    rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
              '&CIK={}&type={}&dateb={}&count={}&owner=exclude&output=atom' \
             .format(cik, doc_type, start, count)
    html = requests.get(rss_url, headers=HEADER).text
    feed = BeautifulSoup(html).feed 
    entries = [
        (
            entry.content.find('filing-href').getText(),
            entry.content.find('filing-type').getText(),
            entry.content.find('filing-date').getText())
        for entry in feed.find_all('entry', recursive=False)]
    return entries

def get_documents(text):
    extracted_docs = []

    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')   
    
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(text)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(text)]
    
    for doc_start_i, doc_end_i in zip(doc_start_is, doc_end_is):
            doc = text[doc_start_i:doc_end_i]
            if get_document_type(doc) == '10-k' or get_document_type(doc) =='10-q':
                extracted_docs.append(doc)
    
    return extracted_docs

def get_document_type(doc):
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    doc_type = type_pattern.findall(doc)[0][len('<TYPE>'):] 
    return doc_type.lower()

def remove_html_tags(text):
    bs = BeautifulSoup(text, 'html.parser')
    # Remove all images and table
    for tag in ["table", "img", "IMS-HEADER", "SEC-HEADER"]:
      for ele in bs.find_all(tag):
        ele.extract()
    text = bs.get_text()
    return text
    
def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    return text

def lemmatize_words(words):
    lemmatized_words = [WordNetLemmatizer().lemmatize(word, 'v') for word in words]
    return lemmatized_words

# ---------- MAIN ------------
nltk.download('wordnet')
nltk.download('ombw-1.4')
nltk.download('stopwords')
nltk.download('omw-1.4')

word_pattern = re.compile('\w+')
lemma_english_stopwords = lemmatize_words(stopwords.words('english'))
  
sec_api = SecAPI()
sec_data = {}
for ticker, cik in cik_lookup.items():
    curr = get_sec_data(cik, '10-K', 20220101, 10) + get_sec_data(cik, '10-Q', 20220101, 30)
    sec_data[ticker] = [item for item in curr if datetime.strptime(item[2], '%Y-%m-%d').date() >= start]

# Store lemmatized filing for each ticker into json files.
# Each line of the json file is a list of 5-year filings for an asset
sec_data_list = list(sec_data.items())
batch_size = 40
num_batch = int(len(sec_data_list) / batch_size) + 1
for i in range(num_batch):
    output_file = open("statements" + str(i)+ ".json", "w", encoding="utf-8")
    batch_start = i * batch_size
    batch_end = (i+1)* batch_size
    
    for ticker, data in sec_data_list[batch_start:batch_end]:
        ticker_doc_array = []
        for index_url, file_type, file_date in tqdm(data, desc='Processing {} Fillings'.format(ticker), unit='filling'):
            if (file_type == '10-Q' or file_type == '10-K'):
                # Download 10-K/10-Q reports
                file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')         
                raw_data = sec_api.get(file_url)
                documents = get_documents(raw_data)
                for document in documents: 
                    
                    # Clean and lemmatize words
                    try:
                      document = clean_text(document)
                    except:
                      print("File cannot be cleaned!")
                    
                    try:
                        document = lemmatize_words(word_pattern.findall(document))
                        document = [word for word in document if word not in lemma_english_stopwords]
                    except KeyError:
                        print('Key error')
                    ticker_doc_array.append({'cik': cik_lookup[ticker], 'file': document, 'file_date': file_date})

        json.dump(ticker_doc_array, output_file) 
        output_file.write("\n")