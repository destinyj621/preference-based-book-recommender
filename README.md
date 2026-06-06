# Book Recommendation System

A preference-based book recommender that builds a personalized taste profile from your reading history and ranks candidates from the BookCrossing dataset using semantic similarity. Recommendations come with predicted star ratings and a "because you liked..." explanation for each pick.

---

## How It Works

1. **Load and clean the candidate pool** — pulls the [BookCrossing dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) from Kaggle, filters to books with at least 50 community ratings, and deduplicates editions by title.

2. **Enrich with ISBNdb** — fetches synopses and genre subjects for every book via the [ISBNdb API](https://isbndb.com/), caching results locally to avoid redundant calls.

3. **Generate embeddings** — encodes each book's title, author, synopsis, and subjects into a dense vector using [`all-MiniLM-L6-v2`](https://www.sbert.net/) (Sentence-BERT).

4. **Build a taste profile** — takes your rated books, converts scores to signed weights (high ratings push the profile toward a book; low ratings push it away), and computes a weighted average embedding as your personal taste vector.

5. **Rank and explain** — scores every unread candidate by cosine similarity to your profile, predicts a star rating using a weighted KNN approach, and identifies which of your highly-rated books each recommendation most resembles.

6. **Spot-check any ISBN** — a utility function lets you query any ISBN directly to see its similarity score, predicted rating, and closest matches in your reading history.

---

## Project Structure

```
.
├── book_recommender.ipynb   # Main notebook
└── data/
    ├── isbn_cache.json      # Cached ISBNdb responses (auto-generated)
    └── books_rated.csv      # Optional: your personal ratings CSV
```

---

## Setup

### Prerequisites

- Python 3.9+
- A free [ISBNdb API key](https://isbndb.com/isbn-database)
- A [Kaggle account](https://www.kaggle.com/) with `kagglehub` configured

### Install dependencies

```bash
pip install numpy pandas requests tqdm sentence-transformers scikit-learn kagglehub
```

### Configure

At the top of the notebook, set your ISBNdb key:

```python
isbndb_key = "your_api_key_here"
```

---

## Providing Your Ratings

There are two ways to supply your reading history:

**Option 1 — Inline dictionary (default)**

Edit the `my_ratings` dict in the *Build the Profile* cell directly:

```python
my_ratings = {
    "The Great Gatsby": 5,
    "Pride and Prejudice": 5,
    "Twilight": 1,
    ...
}
```

**Option 2 — CSV file**

Set `use_csv = True` and point `ratings_file` at a CSV with at minimum a title column and a rating column (0–5). Optional columns for author, synopsis, and subjects are used when a book isn't found in the Kaggle dataset.

```python
use_csv = True
ratings_file = "../data/books_rated.csv"
```

---

## Output

The final cell produces a ranked table of the top `top_n` recommendations (default: 10):

| Title | Author | Similarity | Predicted Rating | Explanation |
|---|---|---|---|---|
| ... | ... | 0.82 | 4.7 | Close to books liked: "The Great Gatsby" & "Anna Karenina" |

You can also query any ISBN directly using `check_isbn()`:

```python
check_isbn("9780439023528")
```

This prints the book's metadata, its similarity score to your profile, the predicted rating, and the books in your history it most resembles.

---

## Tech Stack

| Component | Library |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Similarity | `scikit-learn` cosine similarity |
| Book metadata | ISBNdb API |
| Candidate pool | BookCrossing dataset via `kagglehub` |
| Data handling | `pandas`, `numpy` |

---

## Notes

- ISBNdb API calls are cached in `data/isbn_cache.json`. The first run will be slow if you have a large candidate pool; subsequent runs skip cached ISBNs.
- Books rated 4–5 stars are weighted positively and strongly influence the profile. Books rated 1–2 stars push the profile away from similar titles. A rating of 3 is treated as near-neutral.
- If a rated book isn't found in the Kaggle dataset, the system falls back to your CSV data and then to a live ISBNdb lookup to still include it in the profile.
