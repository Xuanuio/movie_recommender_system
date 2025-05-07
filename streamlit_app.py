import pandas as pd
import numpy as np
import re
import os
import streamlit as st

# --- Xử lý dữ liệu và lưu/đọc file trung gian ---
if os.path.exists("processed_movies.pkl") and os.path.exists("tfidf_vectors.npy") and os.path.exists("similarity_matrix.npy"):
    new_data = pd.read_pickle("processed_movies.pkl")
    tfidf_vectors = np.load("tfidf_vectors.npy")
    similarity_matrix = np.load("similarity_matrix.npy")
else:
    movies = pd.read_csv('dataset.csv')
    movies['tags'] = movies['overview'] + movies['genre']
    new_data = movies.drop(columns=['overview', 'genre'])

    stopwords = set("""
    a an the and or in on with to of at for from by about as is are was were be been being
    has have had do does did will would can could should this that these those so such
    """.split())

    def preprocess(text):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        return [word for word in words if word not in stopwords]

    new_data['tokens'] = new_data['tags'].apply(preprocess)

    vocab = sorted(set(word for tokens in new_data['tokens'] for word in tokens))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)

    def compute_tf(tokens):
        tf = np.zeros(vocab_size, dtype=np.float32)
        for word in tokens:
            if word in word_to_index:
                tf[word_to_index[word]] += 1
        tf /= len(tokens) if tokens else 1
        return tf

    tf_vectors = np.stack(new_data['tokens'].apply(compute_tf).values)

    df = np.zeros(vocab_size, dtype=np.float32)
    for tokens in new_data['tokens']:
        seen = set()
        for word in tokens:
            if word in word_to_index and word not in seen:
                df[word_to_index[word]] += 1
                seen.add(word)

    idf = np.log(len(new_data) / (df + 1))
    tfidf_vectors = tf_vectors * idf

    def compute_cosine_matrix(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_vectors = vectors / norms
        return np.dot(norm_vectors, norm_vectors.T)

    similarity_matrix = compute_cosine_matrix(tfidf_vectors)

    # Lưu lại dữ liệu đã xử lý
    new_data.to_pickle("processed_movies.pkl")
    np.save("tfidf_vectors.npy", tfidf_vectors)
    np.save("similarity_matrix.npy", similarity_matrix)

# --- Streamlit Giao diện ---
st.title("🎬 Hệ thống Gợi ý Phim")
movie_title = st.text_input("Nhập tên phim bạn thích:")

if movie_title:
    if movie_title not in new_data['title'].values:
        st.error("❌ Không tìm thấy phim trong dữ liệu.")
    else:
        index = new_data[new_data['title'] == movie_title].index[0]
        scores = list(enumerate(similarity_matrix[index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        st.subheader("📽️ Top 5 phim tương tự:")
        for i in range(1, 6):
            st.write(f"{i}. {new_data.iloc[scores[i][0]]['title']}")