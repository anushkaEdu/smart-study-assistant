import streamlit as st
from transformers import pipeline
import nltk
import os

nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.set_page_config(page_title="Smart Study Assistant", layout="wide")

# Sidebar
st.sidebar.title("🔧 Options")
st.sidebar.info(
    "Steps:\n"
    "1. Upload or paste notes\n"
    "2. Click Generate\n"
    "3. Review Summary & Flashcards"
)

st.title("Smart Study Assistant")
st.write("Paste your notes or upload a text file, and get **summaries + flashcards**.")
st.markdown("---")  

# Input
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
else:
    text = st.text_area("Or paste your notes here:")

# Generate outputs
if st.button("Generate Summary & Flashcards"):
    if text.strip():
        with st.spinner("Summarizing..."):
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

        st.subheader("Summary")
        st.write(summary)

        st.subheader(" Flashcards (Q&A)")
        sentences = nltk.sent_tokenize(summary)
        for i, sent in enumerate(sentences):
            st.markdown(f"**Q{i+1}:** What is the key idea here?")
            st.write(f"**A{i+1}:** {sent}")
    else:
        st.warning("Please paste or upload some text first!")
