# 🛍️ E-commerce Product Search using TF-IDF (WANDS Dataset)

This project demonstrates how to build a **simple product search engine from scratch** using **TF-IDF** and **cosine similarity** on the [WANDS dataset](https://paperswithcode.com/dataset/wands). The goal is to highlight the end-to-end development of a basic NLP-based retrieval system and evaluate its performance using **Mean Average Precision at K (MAP@K)**.

---

## 🔍 What the Program Does

1. **Loads and preprocesses product and query data** from the WANDS dataset:
   - Handles missing values.
   - Applies standard NLP preprocessing (lowercasing, stopword removal, lemmatization).

2. **Enriches product and query texts**:
   - Combines multiple product text fields (name, description, features, etc.).
   - Expands queries using WordNet synonyms to improve matching recall.

3. **Builds a TF-IDF representation** of the product catalog:
   - Uses unigrams and bigrams for richer representation.
   - Computes cosine similarity between each enriched query and the product embeddings.

4. **Retrieves top-K product matches** for each query using cosine similarity scores.

5. **Evaluates retrieval quality**:
   - Compares predicted products to labeled "Exact" matches from the WANDS dataset.
   - Calculates `MAP@10` for each query to quantify search quality.

---

## 📊 Purpose

- Serves as a baseline for product search engines.
- Illustrates basic **information retrieval (IR)** and **evaluation techniques**.
- Establishes a foundation for integrating **advanced embedding models** (e.g., BERT, Sentence Transformers) for improved performance in future iterations.

---

## 📁 Files Used from WANDS

- `query.csv`: Contains user search queries.
- `product.csv`: Metadata of product catalog.
- `label.csv`: Relevance labels between queries and products (used for MAP@K evaluation).

---

## 🔧 Dependencies

- Python
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`

To install required NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')