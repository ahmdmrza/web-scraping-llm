import streamlit as st
import requests
import re
import json
from bs4 import BeautifulSoup
import trafilatura
from transformers import pipeline
from serpapi import GoogleSearch
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import time

# Cache the transformer pipeline
@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

ner_pipeline = load_ner_pipeline()

# SerpAPI search function
def search_links(query, api_key, location="Malaysia"):
    params = {
        "q": query,
        "location": location,
        "engine": "google",
        "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return [res["link"] for res in results.get("organic_results", []) if "link" in res]

# Download and extract text from a URL using trafilatura
def extract_text_from_url(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded)
    except Exception:
        pass
    return ""

# Regex for email and phone
def extract_contact_info(text):
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phones = re.findall(r"\+?6?01[0-46-9]-*[0-9]{7,8}", text)  # Malaysian mobile format
    return list(set(emails)), list(set(phones))

# Named Entity Recognition for names
def extract_names(text):
    entities = ner_pipeline(text[:1000])  # Limit for performance
    return list(set(ent['word'] for ent in entities if ent['entity_group'] == 'PER'))

# Process a single URL
def process_url(url):
    text = extract_text_from_url(url)
    if "negeri sembilan" not in text.lower():
        return None  # Skip irrelevant locations
    emails, phones = extract_contact_info(text)
    names = extract_names(text)
    return {
        "url": url,
        "emails": emails,
        "phones": phones,
        "names": names
    }

# Main app
st.title("🏢 Building Owner Contact Info Extractor (Negeri Sembilan)")

query = st.text_input("Search query", "building owner contact Negeri Sembilan")
api_key = st.text_input("SerpAPI Key", type="password")

if st.button("Search & Extract"):
    if not api_key:
        st.warning("Please enter your SerpAPI key.")
    else:
        with st.spinner("Searching and scraping..."):
            start = time.time()
            urls = search_links(query, api_key)
            results = []

            # Parallel processing for speed
            with ThreadPoolExecutor(max_workers=5) as executor:
                for data in executor.map(process_url, urls):
                    if data:
                        results.append(data)

            duration = time.time() - start
            st.success(f"✅ Extracted {len(results)} pages in {duration:.2f} seconds.")

            for res in results:
                st.subheader(f"🔗 {res['url']}")
                st.write(f"📧 Emails: {res['emails']}")
                st.write(f"📱 Phones: {res['phones']}")
                st.write(f"👤 Names: {res['names']}")
