#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('pip', 'install -U langgraph langchain langchain-ollama requests beautifulsoup4')
# get_ipython().run_line_magic('pip', 'install streamlit')


# In[2]:


from typing import TypedDict
import requests
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# In[3]:


class State(TypedDict):
    url: str
    html: str
    text: str
    summary: str


# In[4]:


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# In[5]:


### from url to html
def fetch_url(state: State):
    response = requests.get(
        state["url"],
        timeout=15,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    return {
        "html": response.text
    }


# In[6]:


def extract_text(state: State):
    soup = BeautifulSoup(state["html"], "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return {
        "text": text
    }


# In[7]:


def summarize(state: State):
    text = state["text"]

    # 防止网页太长
    text = text[:12000]

    prompt = f"""
You are a research assistant.

Summarize the following webpage.

Requirements:
1. Identify the main topic.
2. Summarize the most important points.
3. Keep the response concise.
4. Do not make up information that is not in the webpage.

Webpage content:
{text}
"""

    response = llm.invoke(prompt)

    return {
        "summary": response.content
    }


# In[8]:


builder = StateGraph(State)

builder.add_node("fetch_url", fetch_url)
builder.add_node("extract_text", extract_text)
builder.add_node("summarize", summarize)

builder.add_edge(START, "fetch_url")
builder.add_edge("fetch_url", "extract_text")
builder.add_edge("extract_text", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()


# In[10]:


# url = input("Enter a URL: ")


# # In[11]:


# result = graph.invoke({
#     "url": url,
#     "html": "",
#     "text": "",
#     "summary": ""
# })


# # In[12]:


# print(result['summary'])


# In[9]:


import streamlit as st
url = st.text_input("Enter URL")

if st.button("Summarize"):
    result = graph.invoke({
    "url": url,
    "html": "",
    "text": "",
    "summary": ""
})
    st.write(result["summary"])


# In[ ]:




