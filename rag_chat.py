import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from fastembed import TextEmbedding

CHAT_MODEL = "llama-3.1-8b-instant"   # check console.groq.com for current model names
TOP_K = 5

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"],
)


@st.cache_resource
def get_embedder():
    return TextEmbedding("BAAI/bge-small-en-v1.5")


def embed(texts):
    return list(get_embedder().embed(texts))


@st.cache_data
def load_reviews():
    df = pd.read_csv("reviews.csv")
    df["embedding"] = embed(df["comment"].tolist())
    return df


st.title("Chat with your Zomato Reviews")
st.caption(f"Searching 500 reviews, answering with {CHAT_MODEL} (via Groq)")


def find_similar_reviews(question, df):
    question_vector = np.array(embed([question])[0])
    review_matrix = np.vstack(df["embedding"].to_numpy())

    scores = review_matrix @ question_vector / (
        np.linalg.norm(review_matrix, axis=1) * np.linalg.norm(question_vector)
    )

    df = df.copy()
    df["score"] = scores
    return df.nlargest(TOP_K, "score")


def ask_llm(question, top_reviews):
    context = ""
    for _, row in top_reviews.iterrows():
        context += f"- ({row['city']}, {row['rating']} stars) {row['comment']}\n"

    system_prompt = (
        "Answer ONLY using the customer reviews provided. "
        "Be concise. If the reviews don't cover it, say so."
    )
    user_prompt = f"Question: {question}\n\nReviews:\n{context}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


review_df = load_reviews()

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder="e.g. What are the most common complaints about delivery?",
)

if question:
    with st.spinner("Thinking..."):
        top_reviews = find_similar_reviews(question, review_df)
        answer = ask_llm(question, top_reviews)

    st.markdown("**Answer:**")
    st.write(answer)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(top_reviews[["city", "rating", "comment"]], hide_index=True)
