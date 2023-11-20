import streamlit as st
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from api import main_api
import google.generativeai as palm

# Inject palm api
palm.configure(api_key=main_api)

# Function to get the model for summarization
def get_palm_model():
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    return models[0].name if models else None

# Function to summarize text using PALM
def summarize_text(text):
    model = get_palm_model()
    if model:
        prompt = f"summarize this text.\ntext = {text}"
        completion = palm.generate_text(
            model=model,
            prompt=prompt,
            temperature=0,
            max_output_tokens=800
        )
        return completion.result
    return "Summarization model not available."

# Define a function to load BERT model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

# Load BERT model and tokenizer (you can choose a specific pre-trained model)
model_name = "bert-base-uncased"
model, tokenizer = load_model_and_tokenizer(model_name)

# Load the English stopwords
stop_words = set(stopwords.words('english'))

# Preprocess query
def preprocess_query(query):
    query = query.lower()
    tokens = [token for token in query.split() if token not in stop_words]
    return ' '.join(set(tokens))

# Define a function to encode text using BERT
@st.cache_data
def encode_text(text):
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True)['input_ids']
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Define a function to calculate cosine similarity
@st.cache_data
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Load and preprocess paper data
@st.cache_resource
def preprocess_and_encode_papers(file):
    if file is not None:
        papers = pd.read_csv(file).sample(n=200, random_state=1)
        papers['Title_Encoded'] = papers['Title'].apply(encode_text)
        papers['Abstract_Encoded'] = papers['Abstract'].apply(encode_text)
        return papers
    return None

# Define the search function
def search_papers(query, papers_data, similarity_threshold, include_summarization=False):
    processed_query = preprocess_query(query)
    query_embedding = encode_text(processed_query)
    search_results = []

    for _, paper in papers_data.iterrows():
        title_similarity = cosine_similarity(query_embedding, paper['Title_Encoded'])
        abstract_similarity = cosine_similarity(query_embedding, paper['Abstract_Encoded'])

        overall_similarity = max(title_similarity, abstract_similarity)
        if overall_similarity > similarity_threshold:
            search_results.append({**paper, 'Similarity': overall_similarity})

    for result in search_results:
        if include_summarization:
            result['Summarization'] = summarize_text(result['Abstract'])
    return sorted(search_results, key=lambda x: x['Similarity'], reverse=True)


# Streamlit UI
st.title('Paper Search Engine with BERT')

# File uploader widget
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = preprocess_and_encode_papers(uploaded_file)

    query_input = "give me data related about " + st.sidebar.text_input("Enter your search query:")
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.7)
    # Add toggle for summarization
    include_summarization = st.sidebar.checkbox("Include Summarization")
    if st.sidebar.button('Search') and query_input and data is not None:
        with st.spinner('Searching for papers...'):
            matched_papers = search_papers(query_input, data, similarity_threshold, include_summarization)
        
        if include_summarization:
            if matched_papers:
                st.success(f"Found {len(matched_papers)} papers matching your query:")
                papers_df = pd.DataFrame(matched_papers)
                papers_df = papers_df[['Title', 'Authors', 'Summarization', 'Abstract', 'Similarity']]
                st.dataframe(papers_df)
            else:
                st.error("No matching papers found.")
        else:
            if matched_papers:
                st.success(f"Found {len(matched_papers)} papers matching your query:")
                papers_df = pd.DataFrame(matched_papers)
                papers_df = papers_df[['Title', 'Authors', 'Abstract', 'Similarity']]
                st.dataframe(papers_df)
            else:
                st.error("No matching papers found.")
else:
    st.sidebar.warning("Please upload a CSV file to proceed.")