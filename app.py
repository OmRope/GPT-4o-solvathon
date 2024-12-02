#Importing the libraries
import os
import time
from bs4 import BeautifulSoup
import re
import requests
import base64
import json
import yfinance as yf
import langchain
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain import LLMChain, PromptTemplate
import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import util, SentenceTransformer

import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv

load_dotenv()

# Access the API key using the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_background(png_file):
#     bin_str = get_base64(png_file)
#     page_bg_img = '''
#     <style>
#     .stApp {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# set_background('download.jpg')
st.header('Stock Recommendation System')
#importing api key as environment variable

llm=ChatOpenAI(temperature=0,model_name='gpt-4o',openai_api_key=openai_api_key)

st.sidebar.write('This tool provides recommendation based on the RAG & ReAct Based Schemes:')

companies = {
    'Zomato': 'ZOMATO.NS',
    'TCS': 'TCS.NS',
    'Wipro': 'WIPRO.NS',
    'Infosys': 'INFY.NS',
    'HCL Technologies': 'HCLTECH.NS',
    'Tata Motors': 'TATAMOTORS.NS',
    'SBI': 'SBIN.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Vodafone Idea': 'IDEA.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Tech Mahindra': 'TECHM.NS',
    'Larsen & Toubro': 'LT.NS',
    'Adani Enterprises': 'ADANIENT.NS',
    'Adani Ports': 'ADANIPORTS.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'Nestle India': 'NESTLEIND.NS',
    'UltraTech Cement': 'ULTRACEMCO.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Tata Steel': 'TATASTEEL.NS',
    'JSW Steel': 'JSWSTEEL.NS',
    'Power Grid Corporation': 'POWERGRID.NS',
    'NTPC': 'NTPC.NS',
    'Coal India': 'COALINDIA.NS',
    'Tata Power': 'TATAPOWER.NS',
    'Havells India': 'HAVELLS.NS',
    'Britannia': 'BRITANNIA.NS',
    'Titan Company': 'TITAN.NS',
    'Godrej Consumer Products': 'GODREJCP.NS',
    'Piramal Enterprises': 'PEL.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Eicher Motors': 'EICHERMOT.NS',
    'Hero MotoCorp': 'HEROMOTOCO.NS',
    'Bajaj Auto': 'BAJAJ-AUTO.NS',
    'Sun Pharmaceutical': 'SUNPHARMA.NS',
    'Dr. Reddy’s Laboratories': 'DRREDDY.NS',
    'Cipla': 'CIPLA.NS',
    'Divi’s Laboratories': 'DIVISLAB.NS',
    'Biocon': 'BIOCON.NS',
    'Aurobindo Pharma': 'AUROPHARMA.NS',
    'Motherson Sumi Systems': 'MOTHERSUMI.NS',
    'Hindalco Industries': 'HINDALCO.NS',
    'Grasim Industries': 'GRASIM.NS',
    'Vedanta': 'VEDL.NS',
    'ONGC':'ONGC.NS',
}

com=[]
val=[]
for i,j in companies.items():
    com.append(i.lower())
    val.append(j)
companies=dict(zip(com,val))

#Get Historical Stock Closing Price for Last 1 Year
def get_stock_price(ticker):
    if "." in ticker:
        ticker = ticker.split(".")[0]
    stock = yf.Ticker(companies[ticker])
    df = stock.history(period="1y")
    df = df[["Close","Volume"]]
    df.index=[str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date",inplace=True)
    return df.to_string()

#Get News From Web Scraping
def google_query(search_term):
    if "news" not in search_term:
        search_term = search_term+" stock news"
    url = f"https://www.google.com/search?q={search_term}"
    url = re.sub(r"\s","+",url)
    return url

#Get Recent Stock News
def get_recent_stock_news(company_name):
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    g_query = google_query(company_name)
    res=requests.get(g_query,headers=headers).text
    soup = BeautifulSoup(res,"html.parser")
    news=[]
    for n in soup.find_all("div","n0jPhd ynAwRc tNxQIb nDgy9d"):
        news.append(n.text)
    for n in soup.find_all("div","IJl0Z"):
        news.append(n.text)

    if len(news) > 6:
        news = news[:4]
    else:
        news = news
    
    news_string=""
    for i,n in enumerate(news):
        news_string+=f"{i}. {n}\n"
    top5_news="Recent News:\n\n"+news_string
    
    return top5_news

#Get Financial Statements
def get_financial_statements(ticker):
    if "." in ticker:
        ticker = ticker.split(".")[0]
    else:
        ticker=ticker
    company = yf.Ticker(companies[ticker])
    balance_sheet = company.balance_sheet
    if balance_sheet.shape[1]>3:
        balance_sheet = balance_sheet.iloc[:,:3]
    balance_sheet = balance_sheet.dropna(how="any")
    balance_sheet = balance_sheet.to_string()
    return balance_sheet

#Initialize DuckDuckGo Search Engine
search=DuckDuckGoSearchRun()     
tools = [
Tool(
    name="Stock Ticker Search",
    func=search.run,
    description="Use only when you need to get stock ticker from internet, you can also get recent stock related news. Dont use it for any other analysis or task"
),
Tool(
    name = "Get Stock Historical Price",
    func = get_stock_price,
    description="Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it"
),
Tool(
    name="Get Recent News",
    func= get_recent_stock_news,
    description="Use this to fetch recent news about stocks"
),
Tool(
    name="Get Financial Statements",
    func=get_financial_statements,
    description="Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluated. You should input stock ticker to it"
)
]

    # zero_shot_agent=initialize_agent(
    #     llm=llm,
    #     agent="zero-shot-react-description",
    #     tools=tools,
    #     verbose=True,
    #     max_iteration=4,
    #     return_intermediate_steps=False,
    #     handle_parsing_errors=True
    # )

    #Adding predefine evaluation steps in the agent Prompt
stock_prompt="""You are a financial advisor. Give stock recommendations for given query.
    Everytime first you should identify the company name and get the stock ticker symbol for the stock.
    Answer the following questions as best you can. You have access to the following tools:

    Get Stock Historical Price: Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it 
    Stock Ticker Search: Use only when you need to get stock ticker from internet, you can also get recent stock related news. Dont use it for any other analysis or task
    Get Recent News: Use this to fetch recent news about stocks
    Get Financial Statements: Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluaated. You should input stock ticker to it

    steps- 
    Note- if you fail in satisfying any of the step below, Just move to next one
    1) Get the company name and search for the "company name + stock ticker" on internet. Dont hallucinate extract stock ticker as it is from the text. Output- stock ticker. If stock ticker is not found, stop the process and output this text: This stock does not exist
    2) Use "Get Stock Historical Price" tool to gather stock info. Output- Stock data
    3) Get company's historic financial data using "Get Financial Statements". Output- Financial statement
    4) Use this "Get Recent News" tool to search for latest stock related news. Output- Stock news
    5) Analyze the stock based on gathered data and give detailed analysis for investment choice. provide numbers and reasons to justify your answer. Output- Give a single answer if the user should buy,hold or sell. You should Start the answer with Either Buy, Hold, or Sell in Bold after that Justify.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do, Also try to follow steps mentioned above
    Action: the action to take, should be one of [Get Stock Historical Price, Stock Ticker Search, Get Recent News, Get Financial Statements]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times, if Thought is empty go to the next Thought and skip Action/Action Input and Observation)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

# Define the prompt template
template = "What is the stock price for {company}?"
prompt_template = PromptTemplate(
    input_variables=["company"],
    template=template
)

# Example of stock_prompt input
stock_prompt = {"company": companies['tcs']}

# Use LLMChain with your preferred LLM
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Run the chain with input
output = llm_chain.run(stock_prompt)

def get_all_stock_data(company_name):
  company_name=company_name.lower()
  histroical_price = get_stock_price(company_name)
  financial_statements = get_financial_statements(company_name)
  recent_news = get_recent_stock_news(company_name)
  return histroical_price, financial_statements, recent_news


# Define a function to analyze data and make a recommendation
def analyze_data(historical_price, financial_statements, recent_news):
    """
    Analyze data and determine a buy/hold/sell recommendation.
    Arguments:
        - historical_price: List or dataframe of stock prices.
        - financial_statements: Parsed financial statement data.
        - recent_news: Parsed recent news articles.
    Returns:
        - recommendation: Buy/Hold/Sell recommendation.
        - justification: Justification for the recommendation.
    """
    # Example placeholder analysis logic (replace with actual financial analysis)
    
    # Check if the stock price has been steadily increasing
    if historical_price and historical_price[-1] > historical_price[0]:
        recommendation = "Buy"
        justification = "Stock price has shown consistent growth over the past year."
    # Check if there is negative sentiment in recent news
    elif "scandal" in recent_news.lower():
        recommendation = "Sell"
        justification = "Recent news suggests negative sentiment due to potential scandals."
    else:
        recommendation = "Hold"
        justification = "More data and analysis is needed for a definitive recommendation."

    return recommendation, justification


historical_price, financial_statements, recent_news = get_all_stock_data('tata steel')

def analyze_stock(company_name):
    historical_price, financial_statements, recent_news = get_all_stock_data(company_name)
    
    # Construct the prompt for the LLM
    prompt = f"""Analyze the following stock data and provide a buy/sell/hold recommendation.

    **Historical Price:**
    ```
    {historical_price}
    ```

    **Financial Statements:**
    ```
    {financial_statements}
    ```

    **Recent News:**
    ```
    {recent_news}
    ```

    Provide a concise recommendation (Buy, Sell, or Hold) followed by a detailed justification.  Use numbers and specific details from the data to support your reasoning.
    """

    # Use the LLM to generate the analysis
    llm = ChatOpenAI(temperature=0, model_name='gpt-4o', openai_api_key=openai_api_key)  # Use a suitable model
    analysis = llm.predict(prompt)
    
    return analysis

# Create an input box for the user to enter a company name
company_name = st.text_input("Enter the company name (e.g., Tata Steel):")

# Create a button to trigger the analysis
if st.button("Analyze"):
    if company_name:
        # Call the analyze_stock function
        analysis_result = analyze_stock(company_name)
        
        # Display the results
        st.subheader(f"Analysis for {company_name}:")
        st.markdown(analysis_result)
    else:
        st.error("Please enter a valid company name.")


#-----------------------------------------------------------------------------------------------------
import torch
import streamlit as st
from sentence_transformers import util, SentenceTransformer
import openai
import numpy as np 
import pandas as pd
from time import perf_counter as timer
from langchain_community.chat_models import ChatOpenAI

embedding_model = SentenceTransformer(model_name_or_path='sentence-transformers/all-MiniLM-L6-v2',device='cpu')
llm=ChatOpenAI(temperature=0,model_name='gpt-4o',openai_api_key=openai_api_key)

text_chunks_and_embedding_df = pd.read_csv('text_chunks_and_embedding_technical.csv')
text_chunks_and_embedding_df['embedding'] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
embedding = torch.tensor(np.array(text_chunks_and_embedding_df['embedding'].tolist()),dtype=torch.float32).to('cpu')
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient='records')

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=False):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)

    return scores, indices

def prompt_formatter(query:str,
                    context_item: list[dict])->str:
    context =  "- "+"\n-".join([item['sentence_chunk'] for item in context_item])
    base_prompt = """ You are a financial Education Provider. As Many individual struggle with stock investment due to lack of finanical education and acess to the relevent resourses. 
    on the following context items, Answere the query.
    context item:
    {context}
    Query : {query}
    NOTE:
    Only give answere that is relevant to the user query
    Answere:
    """
    prompt = base_prompt.format(context=context,query=query) 
    return prompt

st.title("Ask Doubt")
text_input = st.text_area("Enter your text:", height=200)

if st.button("Answer"):
    query = (text_input)
    scores,indices = retrieve_relevant_resources(query=query,embeddings=embedding)
    context_item = [pages_and_chunks[i] for i in indices]
    prompt =prompt_formatter(query,context_item)
    prediction = llm.predict(prompt)
    st.write("Response:")
    st.write(prediction)


# # Example usage
# company_to_analyze = "tata steel"  # Replace with the desired company
# analysis_result = analyze_stock(company_to_analyze)
# print(f"Analysis for {company_to_analyze}:\n{analysis_result}")
