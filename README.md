# Preference-Based Book Recommender

A personalized, explainable book recommendation system built on semantic embeddings and weighted collaborative signals. It learns your taste from books you've already rated, then scores and ranks the entire Kaggle Book Recommendation dataset to surface the books you're most likely to enjoy — and predicts the rating you'd give them.

---

## How It Works

```
Kaggle Book Dataset (271k books)
        ↓
Filter to popular books (≥ 50 ratings) + deduplicate editions
        ↓
ISBNdb API enrichment — synopsis + subjects (cached to disk)
        ↓
Sentence-BERT embedding of all candidate books
        ↓
Your Rated Books (MY_RATINGS dict or books_rated.csv)
        ↓
Three-tier metadata lookup per rated book:
  1. Found in Kaggle pool → use pre-computed embedding
  2. Not in Kaggle + CSV has synopsis/subjects → embed from CSV
  3. Not in Kaggle + CSV empty → query ISBNdb, embed on the fly
        ↓
Weighted preference profile vector
  (loved books pull toward, disliked books push away)
        ↓
Cosine similarity: profile vs. every candidate
        ↓
Weighted KNN rating prediction
  predicted_rating = Σ(sim × your_rating) / Σ(sim)
        ↓
Top N recommendations + predicted rating + explainability
```

---

## Key Features

- **Kaggle dataset as candidate pool** — 270k+ books, filtered and deduplicated to a clean set of well-rated titles
- **ISBNdb enrichment** — synopsis and subjects fetched for every candidate book; results cached to `data/isbn_cache.json` so the API is only hit once
- **Open preference input** — your rated books don't have to be in the Kaggle dataset; any book you've read can be looked up via ISBNdb and still contribute to your profile
- **CSV or dict mode** — toggle `USE_CSV` to load ratings from `books_rated.csv` or edit the `MY_RATINGS` dict directly in the notebook
- **CSV-first metadata** — when using a CSV, manually filled synopsis and subjects columns take priority over API calls; ISBNdb is only queried when those fields are empty
- **Weighted profile vector** — high-rated books contribute more, low-rated books actively push the profile away from disliked content
- **Weighted KNN rating prediction** — for every recommendation, predicts the rating you'd give it (1–5 stars) based on similarity to your rated books
- **Explainability** — each recommendation shows which of your liked books drove it ("Because you liked…")
- **On-demand rating prediction** — `predict_isbn("9780439023481")` fetches a book by ISBN, embeds it, and predicts your rating

---

## Pipeline in Detail

### Step 1 — Load Kaggle Data
`Books.csv` and `Ratings.csv` are loaded via `kagglehub`. Implicit ratings (score = 0) are dropped — only explicit 1–10 ratings are kept.

### Step 2 — Clean and Filter
Books with fewer than `MIN_RATINGS` (default: 50) explicit ratings are removed. Duplicate editions of the same book are collapsed — parentheticals like `(Book 1)` and subtitles after `:` are stripped from titles, and the edition with the most ratings is kept.

### Step 2.5 — Enrich with ISBNdb
For every book that survives the filter, the ISBNdb API is called to retrieve synopsis and subjects. Results are saved to `data/isbn_cache.json`. On subsequent runs this step is instant. Coverage is printed at the end (typically >99% for well-known titles).

### Step 3 — Embed with Sentence-BERT
Each book's text string (`"Title by Author | synopsis | Genres: subjects"`) is encoded into a 384-dimensional unit vector by `all-MiniLM-L6-v2`. Similar books end up with vectors pointing in similar directions.

### Step 4 — Your Preferences
Set `USE_CSV = True` to load from `books_rated.csv`, or `False` to use the `MY_RATINGS` dict. For each rated book, the system follows the three-tier fallback:
- **Tier 1** `[dataset]` — book found in the Kaggle pool; uses its pre-computed embedding
- **Tier 2** `[CSV]` — book not in Kaggle, but your CSV has synopsis/subjects; embedded directly from CSV data, no API call
- **Tier 3** `[ISBNdb]` — book not in Kaggle and CSV fields are empty; ISBNdb is queried and the book is embedded on the fly

Rating weights applied to the profile:

| Your Rating | Weight |
|-------------|--------|
| 5 ★         | +5.0   |
| 4 ★         | +4.0   |
| 3 ★         | +1.0   |
| 1–2 ★       | −0.5   |

### Step 5 — Recommend
The weighted sum of your rated embeddings forms your taste profile vector. Cosine similarity is computed between the profile and every candidate. The weighted KNN formula then predicts a rating for each top result:

```
predicted_rating = Σ(sim(candidate, rated_book_i) × rating_i) / Σ(sim)
```

Only positively similar books (sim > 0) contribute to the prediction.

---

## Predicted Rating for Any ISBN

After running the full notebook, call `predict_isbn()` from the last cell with any ISBN-10 or ISBN-13:

```python
predict_isbn("9780439023481")   # The Hunger Games
```

Output:
```
querying ISBNdb for ISBN 9780439023481...

  Title   : The Hunger Games
  Author  : Suzanne Collins
  Synopsis: yes (843 chars)
  Subjects: Fiction, Science Fiction, Dystopia, Young Adult

  ── Results ────────────────────────────────
  Similarity to your profile : 0.6821
  Predicted rating           : 4.37 / 5
  Because you liked          : Divergent & The Hobbit
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Candidate dataset | [Kaggle — Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) |
| Dataset loading | `kagglehub` |
| Metadata enrichment | ISBNdb API v2 |
| Semantic embedding | Sentence-BERT `all-MiniLM-L6-v2` |
| Similarity scoring | Cosine similarity |
| Rating prediction | Weighted KNN in embedding space |
| Environment | Python / Jupyter Notebook |

---

## Project Structure

```
preference-based-book-recommender/
│
├── data/
│   ├── books_rated.csv          # Your personal ratings (title, authors,
│   │                            #   synopsis, subjects, user_rating, ...)
│   └── isbn_cache.json          # ISBNdb response cache (auto-generated)
│
├── notebooks/
│   └── book_recommender.ipynb   # Full pipeline notebook
│
├── README.md
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

1. Get a free API key from [isbndb.com](https://isbndb.com) and paste it into the `ISBNDB_API_KEY` variable in the first cell
2. Open `notebooks/book_recommender.ipynb`
3. Set `USE_CSV = True` to load from `books_rated.csv`, or edit `MY_RATINGS` directly
4. Run all cells top to bottom

The ISBNdb enrichment step (Step 2.5) only runs on the first execution — after that the cache handles everything.

---

## Configuration

All tunable parameters are in the first cell:

| Variable | Default | Description |
|---|---|---|
| `TOP_N` | `10` | Number of recommendations to return |
| `MIN_RATINGS` | `50` | Minimum community ratings for a book to enter the candidate pool |
| `USE_CSV` | `False` | `True` = load from `books_rated.csv`, `False` = use `MY_RATINGS` dict |
| `RATINGS_CSV` | `../data/books_rated.csv` | Path to your ratings CSV |
| `ISBNDB_API_KEY` | — | Your ISBNdb API key |

---

## Evaluation

The system addresses the following tasks from the project proposal:

| Task | Implementation |
|---|---|
| **Recommendation task** | Cosine similarity ranking of Kaggle candidate pool against preference profile |
| **Rating prediction task** | Weighted KNN in embedding space — `predicted_rating = Σ(sim × rating) / Σ(sim)` |
| **Cold-start handling** | Any book can be used in `MY_RATINGS` regardless of whether it's in the dataset; ISBNdb fills in missing metadata automatically |
| **Explainability** | Every recommendation identifies which of your liked books most influenced it |
