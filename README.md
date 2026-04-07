# 📚 A Preference-Based and Explainable Book Recommendation System

**Destiny Raburnel**
College of Computing and Software Engineering — Kennesaw State University

---

## Overview

This project is a personalized, explainable book recommendation system powered by semantic AI. It goes beyond traditional recommendation approaches by combining semantic embeddings, agentic book discovery, machine learning reranking, and a natural language explainability layer to recommend books that genuinely match what a user likes — and explain *why*.

The system is built around a real personal reading history, where books the user has rated serve as the training signal for building a preference profile, and books the user hasn't read yet are what the system recommends.

---

## Key Features

- **Semantic Embeddings** — Uses Sentence-BERT (`all-MiniLM-L6-v2`) to understand the *meaning* of book descriptions, genres, and subjects — not just keyword overlap
- **Cosine Similarity Scoring** — Measures how closely each candidate book aligns with the user's preference profile in vector space
- **Agentic Book Discovery** — Rather than scoring a fixed dataset, the system actively searches for new books using KeyBERT-extracted profile keywords, querying ISBNdb (with Google Books API as fallback)
- **Weighted User Profile** — Builds a profile vector from the user's rated books, with high-rated books contributing more and low-rated books actively steering the profile *away* from disliked content
- **ML Reranking Layer** — An XGBoost model reranks candidates using multiple signals: cosine similarity score, publish year, page count, community rating, and metadata coverage
- **Cold-Start Handling** — New users (or users exploring a new genre) can answer natural language prompts about their preferences instead of needing prior reading history
- **Explainability Layer** — Each recommendation comes with a human-readable explanation identifying which themes and features drove the suggestion
- **Hard Filters** — Candidates are filtered by publish year and metadata completeness before scoring, keeping results relevant and reliable

---

## How It Works

```
Your Rated Books (180 ISBNs)
        ↓
API Enrichment — ISBNdb → Google Books fallback
        ↓
You Assign Ratings (0–5 scale)
        ↓
Sentence-BERT Embedding of all books
        ↓
Weighted User Profile Vector
  (high ratings boosted, low ratings subtracted)
        ↓
KeyBERT Keyword Extraction from Profile
        ↓
Agent Queries ISBNdb/Google Books with Keywords
        ↓
Hard Filters — publish year floor, language, coverage score
        ↓
Sentence-BERT Embedding of Candidates
        ↓
Cosine Similarity Scoring
        ↓
XGBoost ML Reranking
  (cosine score + publish year + pages + community rating + coverage)
        ↓
Top N Recommendations + Explainability Layer
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Semantic Embedding | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Keyword Extraction | KeyBERT |
| Similarity Scoring | Cosine Similarity |
| ML Reranking | XGBoost |
| Primary Data API | ISBNdb API v2 |
| Fallback Data API | Google Books API |
| Development Environment | Python / Jupyter Notebook |

---

## Dataset

The dataset starts from a personal list of ~180 ISBNs. Each ISBN is enriched via API to collect the following fields:

- `isbn13` — unique identifier
- `title`, `authors` — basic metadata
- `synopsis` — primary text used for semantic embedding
- `subjects` — genre and theme tags
- `date_published` — used for publish year filtering and ML reranking
- `pages` — used as an ML reranking feature
- `language` — filtered to English
- `data_source` — tracks whether data came from ISBNdb or Google Books
- `coverage_score` — count of successfully populated key fields
- `user_rating` — manually assigned by the user (0–5); blank = unread

Books with a user rating form the **user profile set**. Books without a rating form the **candidate pool** used for recommendations.

---

## System Components

### Cold-Start Handling
Users with no reading history answer a few natural language questions:
- *"What genres or themes are you interested in?"*
- *"Describe the vibe or tone you're looking for."*

Responses are embedded with Sentence-BERT and used as a temporary profile vector. This also allows existing users to explore a new genre without their history working against them.

### User Profile Vector
A weighted average of the embeddings of all rated books:
- Ratings **4–5** → full/boosted weight
- Rating **3** → normal weight
- Ratings **1–2** → embedding subtracted at reduced weight (negative signal)

### Rating Prediction Task
A user can input any ISBN and the system will:
1. Query the API for the book's metadata
2. Embed the book using the rich constructed string
3. Compute cosine similarity against the user profile
4. Normalize the result into a predicted rating (0–5)

### Recommendation Task
1. KeyBERT extracts dominant keywords from the user profile
2. The agent queries ISBNdb/Google Books with those keywords
3. Results are filtered (publish year, language, coverage)
4. Candidates are embedded and scored via cosine similarity
5. XGBoost reranks results using multiple features
6. Top N books are returned with explanations

### Explainability Layer
Each recommendation includes a template-based explanation identifying the keywords and features that most influenced the result, e.g.:
> *"Recommended because your profile strongly reflects themes of found family and redemption, and this book was published recently with a high community rating."*

---

## Improvements Over Prior Work

| Limitation in Prior Research | How This Project Addresses It |
|---|---|
| No cold-start handling | Natural language prompting before any history exists |
| TF-IDF only (no semantic understanding) | Sentence-BERT semantic embeddings |
| Sparse metadata (title/author only) | API-enriched synopsis, subjects, and genres |
| Static candidate pools | Agentic live search via KeyBERT + ISBNdb |
| No explainability | KeyBERT + template-based explanations |
| All ratings weighted equally | Weighted profile vector by rating score |
| No penalty for disliked books | Low-rated embeddings subtracted from profile |
| Single similarity signal only | XGBoost reranking across multiple features |
| No recency awareness | Publish year as hard filter and ML feature |
| No metadata reliability signal | Coverage score flags low-confidence results |

---

## Project Structure

```
project/
│
├── data/
│   ├── isbns.txt                  # Raw input — 180 ISBNs
│   ├── books_enriched.csv         # API-enriched metadata
│   └── books_rated.csv            # Final dataset with user ratings
│
├── notebooks/
│   └── book_recommender.ipynb     # Main project notebook
│
├── README.md
└── requirements.txt
```

---

## Evaluation

Recommendations are evaluated using **Precision@k** — out of the top 10 recommendations returned, how many would the user actually want to read? The ML-reranked ordering is also compared against cosine-only ordering to demonstrate the reranking layer's contribution.

---

## References

1. Jia Liu. 2024. *Design of Book Recommendation System Based on Machine Learning in Smart Library.* IEEE AIARS 2024.
2. Adli Ihsan Hariadi and Dade Nurjanah. 2017. *Hybrid Attribute and Personality Based Recommender System for Book Recommendation.* IEEE ICoDSE 2017.
3. Anvi Vats, Yamini Agrawal and Neha Tyagi. 2025. *A Hybrid Book Recommendation System Using Collaborative Filtering and Content-Based Filtering with Neural Embeddings.* IEEE ICCCA 2025.
