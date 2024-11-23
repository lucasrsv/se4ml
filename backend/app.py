from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# Paths to pre-saved data
PATH_TO_EMBEDS = 'compressed_array.npz'
PATH_TO_DF = 'compressed_dataframe.csv.gz'

# Initialize models
model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Load embeddings and DataFrame
embeddings = np.load(PATH_TO_EMBEDS)['array_data']
df_data = pd.read_csv(PATH_TO_DF, compression='gzip')

# Initialize FAISS index
embed_length = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embed_length)
faiss_index.add(embeddings)

# FastAPI app
app = FastAPI()

# CORS Configuration
origins = [
    "http://localhost",  # Local development
    "http://localhost:3000",  # React app or similar running on port 3000
    "https://example.com",  # Add any other domains that should be allowed
]

# Add CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins to allow
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Request schema
class SearchRequest(BaseModel):
    query_text: str
    num_results_to_print: int = 10
    top_k: int = 10

# Response schema
class SearchResult(BaseModel):
    arxiv_id: str
    link_to_pdf: str
    cat_text: str
    title: str
    abstract: str

# Topic schema
class Topic(BaseModel):
    topic: str
    score: float


def run_faiss_search(query_text, top_k):
    query = [query_text]
    query_embedding = model.encode(query)
    scores, index_vals = faiss_index.search(query_embedding, top_k)
    return index_vals[0]  # Return the list of top_k indices


def run_rerank(index_vals_list, query_text):
    chunk_list = list(df_data['prepared_text'])
    pred_strings_list = [chunk_list[item] for item in index_vals_list]

    cross_input_list = [[query_text, item] for item in pred_strings_list]
    df = pd.DataFrame(cross_input_list, columns=['query_text', 'pred_text'])
    df['original_index'] = index_vals_list

    cross_scores = cross_encoder.predict(cross_input_list)
    df['cross_scores'] = cross_scores

    df_sorted = df.sort_values(by='cross_scores', ascending=False).reset_index(drop=True)

    pred_list = []
    for i in range(len(df_sorted)):
        original_index = df_sorted.loc[i, 'original_index']
        arxiv_id = str(df_data.loc[original_index, 'id'])  # Ensure arxiv_id is a string
        pred_list.append({
            "arxiv_id": arxiv_id,
            "link_to_pdf": f"https://arxiv.org/pdf/{arxiv_id}",
            "cat_text": df_data.loc[original_index, 'cat_text'],
            "title": df_data.loc[original_index, 'title'],
            "abstract": df_sorted.loc[i, 'pred_text']
        })
    return pred_list


def extract_topics(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=10)
    X = tfidf.fit_transform(df['prepared_text']) 

    feature_names = tfidf.get_feature_names_out()
    dense = X.todense()
    topic_scores = dense.sum(axis=0).A1
    topic_data = list(zip(feature_names, topic_scores))

    sorted_topics = sorted(topic_data, key=lambda x: x[1], reverse=True)

    return sorted_topics[:10]

@app.post("/search", response_model=List[SearchResult])
def search_arxiv(request: SearchRequest):
    try:
        index_vals_list = run_faiss_search(request.query_text, request.top_k)
        pred_list = run_rerank(index_vals_list, request.query_text)
        return pred_list[:request.num_results_to_print]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/topics", response_model=List[Topic])
def get_topics():
    try:
        topics = extract_topics(df_data)
        return [{"topic": topic, "score": score} for topic, score in topics]
    except Exception as e:
        return {"error": str(e)}