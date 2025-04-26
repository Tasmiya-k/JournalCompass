import streamlit as st
from yake import KeywordExtractor
from sentence_transformers import SentenceTransformer

# Initialize models
kw_extractor = KeywordExtractor(lan="en", top=5)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Keyword extraction function
def extract_keyphrases(text):
    if not text:
        return []
    try:
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except:
        return []

# Streamlit UI
st.set_page_config(page_title="JournalCompass", layout="centered")

st.title("🧭 JournalCompass")
st.subheader("Find the soul of your research paper through keywords!")

# Input fields
title = st.text_input("Enter Research Paper Title:")
abstract = st.text_area("Enter Research Paper Abstract:")

if st.button("Analyze"):
    full_text = f"{title} {abstract}"
    keyphrases = extract_keyphrases(full_text)

    if keyphrases:
        st.success("🔑 Extracted Keyphrases:")
        for idx, phrase in enumerate(keyphrases, 1):
            st.write(f"{idx}. {phrase}")
    else:
        st.warning("⚠️ No keyphrases could be extracted. Please check the input.")

    # Optional: Display embedding
    embedding = embedder.encode(full_text)
    st.write("🧠 Sentence Embedding Vector (first 5 values):", embedding[:5])
