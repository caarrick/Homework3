from sec_api import QueryApi


from sec_api import ExtractorApi

import pandas as pd
import json

queryApi = QueryApi(api_key="3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")

query = {
  "query": { "query_string": { 
      "query": "ticker: KHC AND filedAt:{2017-01-01 TO 2021-12-31} AND formType:\"10-K\"" 
    } },
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
}

filings = queryApi.get_filings(query)

#print(filings)


df =pd.json_normalize(filings['filings'])
df.drop(df[df['formType'] != '10-K'].index, inplace=True)
# df.reset_index


for i in df.index:
    #print url
    url_10K = df['linkToFilingDetails'][i]
    print(url_10K)
    #print(i)


# %% 10-K extract item 7

extractorApi = ExtractorApi("3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")   
#filing_url_10q = https://www.sec.gov/ix?doc=/Archives/edgar/data/1637459/000163745922000018/khc-20211225.htm"
filing_url = df['linkToFilingDetails'][1]
print(filing_url)
section_10k_item7 = extractorApi.get_section(filing_url, "7", "text")

### Problem: Return "undefined"
print(section_10K_item7)

with open('C:\\Users\\caarrick\\Data\\10K\\KHC_10K_item7.txt', 'w') as f:
    f.write(section_10K_item7)
    
    def txt2sentence(txt):
     import nltk
     nltk.download('punkt')
     from nltk.tokenize import sent_tokenize
     sentences = sent_tokenize(txt, language= "english")
     df=pd.DataFrame(sentences)
     return df  
    
    # %%
    
    from sec_api import ExtractorApi

    import pandas as pd
    import json

    queryApi = QueryApi(api_key="3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")

    query = {
      "query": { "query_string": { 
          "query": "ticker: ULTA AND filedAt:{2017-01-01 TO 2021-12-31} AND formType:\"10-k\"" 
        } },
      "from": "0",
      "size": "10",
      "sort": [{ "filedAt": { "order": "desc" } }]
    }

    filings = queryApi.get_filings(query)

    #print(filings)


    df =pd.json_normalize(filings['filings'])
    df.drop(df[df['formType'] != '10-K'].index, inplace=True)
    # df.reset_index


    for i in df.index:
        #print url
        url_10K = df['linkToFilingDetails'][i]
        print(url_10K)
        #print(i)


    # %% 10-Q extract item 2 

extractorApi = ExtractorApi("3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")   
#filing_url_10q = "https://www.sec.gov/ix?doc=/Archives/edgar/data/1403568/000155837022004330/ulta-20220129x10k.htm"
filing_url = df['linkToFilingDetails'][1]
print(filing_url)
section_10k_item7 = extractorApi.get_section(filing_url, "7", "text")

### Problem: Return "undefined"
print(section_10k_item7)

with open('C:\\Users\\caarrick\\Data\\10K\\ULTA_10K_item7.txt', 'w') as f:
   f.write(section_10k_item7)
   
   def txt2sentence(txt):
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(txt, language= "english")
    df=pd.DataFrame(sentences)
    return df  
        
        # %%
        
    from sec_api import ExtractorApi

    import pandas as pd
    import json

    queryApi = QueryApi(api_key="3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")

    query = {
          "query": { "query_string": { 
              "query": "ticker: WMT AND filedAt:{2017-01-01 TO 2021-12-31} AND formType:\"10-k\"" 
            } },
          "from": "0",
          "size": "10",
          "sort": [{ "filedAt": { "order": "desc" } }]
        }

    filings = queryApi.get_filings(query)

        #print(filings)


df =pd.json_normalize(filings['filings'])
df.drop(df[df['formType'] != '10-K'].index, inplace=True)
        # df.reset_index


for i in df.index:
            #print url
            url_10K = df['linkToFilingDetails'][i]
            print(url_10K)
            #print(i)


        # %% 10-Q extract item 2 

extractorApi = ExtractorApi("3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")
#filing_url_10q = "https://www.sec.gov/ix?doc=/Archives/edgar/data/104169/000010416922000012/wmt-20220131.htm"
filing_url = df['linkToFilingDetails'][1]
print(filing_url)
section_10k_item7 = extractorApi.get_section(filing_url, "7", "text")
### Problem: Return "undefined"
print(section_10k_item7)

with open('C:\\Users\\caarrick\\Data\\10K\\WMT_10K_item7.txt', 'w') as f:
            f.write(section_10k_item7)
            
            def txt2sentence(txt):
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(txt, language= "english")
    df=pd.DataFrame(sentences)
    return df  
            
# %% 

from sec_api import ExtractorApi

import pandas as pd
import json

queryApi = QueryApi(api_key="3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")

query = {
  "query": { "query_string": { 
      "query": "ticker: TGT AND filedAt:{2017-01-01 TO 2021-12-31} AND formType:\"10-k\"" 
    } },
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
}

filings = queryApi.get_filings(query)

#print(filings)


df =pd.json_normalize(filings['filings'])
df.drop(df[df['formType'] != '10-K'].index, inplace=True)
# df.reset_index


for i in df.index:
    #print url
    url_10K = df['linkToFilingDetails'][i]
    print(url_10K)
    #print(i)


# %% 10-Q extract item 2 

extractorApi = ExtractorApi("3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")   
#filing_url_10q = "https://www.sec.gov/ix?doc=/Archives/edgar/data/27419/000002741922000007/tgt-20220129.htm"
filing_url = df['linkToFilingDetails'][1]
print(filing_url)
section_10k_item7 = extractorApi.get_section(filing_url, "7", "text")

### Problem: Return "undefined"
print(section_10k_item7)

with open('C:\\Users\\caarrick\\Data\\10K\\TGT_10K_item7.txt', 'w') as f:
    f.write(section_10k_item7)
    
    def txt2sentence(txt):
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(txt, language= "english")
    df=pd.DataFrame(sentences)
    return df  
    
# %%

from sec_api import ExtractorApi

import pandas as pd
import json

queryApi = QueryApi(api_key="3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")

query = {
  "query": { "query_string": { 
      "query": "ticker: GIS AND filedAt:{2017-01-01 TO 2021-12-31} AND formType:\"10-k\"" 
    } },
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
}

filings = queryApi.get_filings(query)

#print(filings)


df =pd.json_normalize(filings['filings'])
df.drop(df[df['formType'] != '10-K'].index, inplace=True)
# df.reset_index


for i in df.index:
    #print url
    url_10K = df['linkToFilingDetails'][i]
    print(url_10K)
    #print(i)


# %% 10-Q extract item 2 

extractorApi = ExtractorApi("3d6c989f663effa78060b723678597151fc092d3f88899f0435ce4b1f7c7f3e5")   
#filing_url_10q = "https://www.sec.gov/ix?doc=/Archives/edgar/data/40704/000119312521204830/d184854d10k.htm"
filing_url = df['linkToFilingDetails'][1]
print(filing_url)
section_10k_item7 = extractorApi.get_section(filing_url, "7", "text")

### Problem: Return "undefined"
print(section_10k_item7)

with open('C:\\Users\\caarrick\\Data\\10K\\GIS_10K_item7.txt', 'w') as f: 
        f.write(section_10k_item7)
        
def txt2sentence(txt):
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(txt, language= "english")
    df=pd.DataFrame(sentences)
    return df  

# %%
 
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

df_GIS=pd.DataFrame()
for index in range(len(Final_10k)):
    df_sentence = txt2sentence(Final_10k[index])
    
    results=[]
    for index, row in df_sentence.iterrows():
        try:
            results.append(nlp(row[0]))
        except:
            print('COULD NOT READ A LINE')
            
    neutral=0
    negative=0
    positive=0
    
    for i in results:
        if i[0]['label'] == 'neutral':
            neutral=neutral+1
        if i[0]['label'] == 'negative':
            negative=negative+1
        if i[0]['label'] == 'positive':
            positive=positive+1
    
    GIS = [{'Ticker':"CROX",'Year':'2021','Neutral':neutral,'Positive':positive,'Negative':negative}]
    df_GIS=df_GIS.append(pd.DataFrame(CROX))
            
# %% SENTIMENT ANALYSIS *need to fix year in df output*
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

df_GIS=pd.DataFrame()
for index in range(len(Final_10K)):
    df_sentence = txt2sentence(Final_10K[index])
    
    results=[]
    for index, row in df_sentence.iterrows():
        try:
            results.append(nlp(row[0]))
        except:
            print('COULD NOT READ A LINE')
            
    neutral=0
    negative=0
    positive=0
    
    for i in results:
        if i[0]['label'] == 'neutral':
            neutral=neutral+1
        if i[0]['label'] == 'negative':
            negative=negative+1
        if i[0]['label'] == 'positive':
            positive=positive+1
    
    GIS = [{'Ticker':"GIS",'Year':'2021','Neutral':neutral,'Positive':positive,'Negative':negative}]
    df_GIS=df_GIS.append(pd.DataFrame(CROX))
            
#%% *keep years straight before fixed
# 2021: neutral 376
# Final_10k[0]=2021
# Final_10k[5]=2016
# to use a sentiment score: results[0][0]['score']
#%% Stock Return Analysis for Comparison

def price2ret(prices,retType='simple'):
    if retType == 'simple':
        ret = (prices/prices.shift(1))-1
    else:
        ret = np.log(prices/prices.shift(1))
    return ret

import pandas_datareader.data as web
GIS_Price= web.DataReader('GIS', 'yahoo', start='2017-01-01', end='2021-04-01')
GIS_Price['Returns']= price2ret(CROX_Price[['Adj Close']])

import matplotlib.pyplot as plt

plt.figure()
plt.plot(GIS_Price['Adj Close'], color='Purple',)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('GIS Price!')

plt.figure()
plt.plot(GIS_Price['Returns'], color='Pink',)
plt.xlabel('Date')
plt.ylabel('% Returns')
plt.title('GIS Returns!')# %% SENTIMENT ANALYSIS *need to fix year in df output*
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

df_GIS=pd.DataFrame()
for index in range(len(Final_10k)):
    df_sentence = txt2sentence(Final_10k[index])
    
    results=[]
    for index, row in df_sentence.iterrows():
        try:
            results.append(nlp(row[0]))
        except:
            print('COULD NOT READ A LINE')
            
    neutral=0
    negative=0
    positive=0
    
    for i in results:
        if i[0]['label'] == 'neutral':
            neutral=neutral+1
        if i[0]['label'] == 'negative':
            negative=negative+1
        if i[0]['label'] == 'positive':
            positive=positive+1
    
    GIS = [{'Ticker':"GIS",'Year':'2021','Neutral':neutral,'Positive':positive,'Negative':negative}]
    df_GIS=df_GIS.append(pd.DataFrame(GIS))
            
