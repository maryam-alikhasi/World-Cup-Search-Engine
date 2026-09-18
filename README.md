# World Cup Search Engine

## Overview

This project implements an **Information Retrieval (IR) system** for searching a collection of **FIFA World Cup matches from 1930 to 2022**.

The system is designed to handle different types of user queries, ranging from simple keyword searches to structured field queries and Boolean expressions. Retrieved documents are ranked using a **TF-IDF** vector space model.

The project was developed from scratch to implement the main components of an information retrieval system, including text preprocessing, document construction, inverted indexing, query processing, ranking, and evaluation.

---

## Dataset

The input dataset is:

```text
matches_1930_2022.csv
```

It contains **World Cup match records**, where each row represents one match.

The dataset includes structured and semi-structured information such as:

* Home and away teams
* Match score
* Goals and assist
* Penalties
* Own goals
* Yellow and red cards
* Substitutions
* Penalty shootouts
* Referees
* Stadium and city
* Match date
* Attendance

Many event-related fields contain semi-structured information that needs to be parsed before being used by the search system.

---

## System Architecture

The overall retrieval pipeline is:

```text
CSV Dataset
     │
     ▼
Data Loading
     │
     ▼
Document Construction
     │
     ▼
Text Preprocessing
     │
     ├───────────────┐
     ▼               ▼
Inverted Index   Field Index
     │               │
     └───────┬───────┘
             ▼
      Query Processing
             │
             ▼
       Candidate Retrieval
             │
             ▼
        TF-IDF Ranking
             │
             ▼
       Search Results
```

---

## 1. Data Configuration

The `config.py` file acts as the central configuration module.

It defines:

* Dataset path
* CSV separator
* Mapping between project field names and dataset columns

The dataset is configured as:

```python
DATA_PATH = "matches_1930_2022.csv"
CSV_SEP = ","
```

The `COLUMN_MAP` dictionary provides a consistent way to access the different fields throughout the project.

---

## 2. Text Preprocessing

The `preprocess.py` module is responsible for cleaning and normalizing text before indexing and searching.

The preprocessing pipeline includes:

### Special Character Cleaning

HTML-encoded and special characters are normalized using a predefined character map.

For example, different apostrophe and quotation characters are converted into standard forms.

### Accent Removal

Unicode normalization is used to remove accents and diacritical marks.

This allows queries such as:

```text
Mbappe
```

to match names containing:

```text
Mbappé
```

### Lowercasing

All text is converted to lowercase to make the search case-insensitive.

### Punctuation Removal

Punctuation is removed using regular expressions, while apostrophes and hyphens are preserved.

### Tokenization

The normalized text is split into individual tokens.

### Stopword Removal

Common English stopwords such as:

```text
the, and, or, is, are, in, on, ...
```

are removed.

The complete preprocessing pipeline is:

```text
Raw Text
   ↓
Special Character Cleaning
   ↓
Accent Removal
   ↓
Lowercasing
   ↓
Punctuation Removal
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Clean Tokens
```

---

## 3. Document Construction

The `document_builder.py` module converts each row of the dataset into a searchable document.

Each match is represented using three main components:

### Text

A unified natural-language representation of the match.

For example, the generated document can contain information such as:

```text
Argentina vs France.
Round: Final.
Score: 3-3.
Stadium: Lusail Stadium.
Referee: Szymon Marciniak.
...
```

Match events such as goals, cards, substitutions, penalties, and shootouts are also converted into searchable text.

### Fields

Structured fields are stored separately to support field-specific queries.

Available fields include:

```text
team
home_team
away_team
round
stage
stadium
city
venue
referee
captain
coach
player
score
year
host
attendance
```

The `player` field is constructed by collecting player names appearing across different match events.

### Metadata

Metadata is used mainly for displaying search results to the user.

It includes information such as:

* Home team
* Away team
* Round
* Stadium
* City
* Score
* Referee
* Date
* Attendance
* Year

Metadata is not used for TF-IDF ranking.

---

## 4. Inverted Index

The `index_builder.py` module implements the main indexing structures.

### Inverted Index

The `InvertedIndex` class is used for free-text search.

The index stores, for each term:

* Document IDs
* Term frequency (`TF`)
* Positions of the term inside the document

Conceptually:

```text
term
 └── document
      ├── TF
      └── positions
```

The index also maintains:

* Document frequency (`DF`)
* Document lengths
* Total number of documents

The stored positions allow the system to support **phrase search**.

### Field Index

The `FieldIndex` class supports structured queries such as:

```text
team:Argentina
round:Final
referee:"Szymon Marciniak"
```

It maintains a separate index for each searchable field.

---

## 5. Query Processing

The `query_processor.py` module parses user queries and retrieves matching document IDs.

The system supports four main query types.

### Simple Keyword Search

Example:

```text
messi
```

Retrieves documents containing the term.

### Boolean Search

The system supports:

* `AND`
* `OR`
* `NOT`

Example:

```text
messi NOT france
```

### Field Search

Users can search within specific fields:

```text
team:Argentina
```

```text
round:Final
```

```text
referee:"Szymon Marciniak"
```

### Combined Queries

Field queries can also be combined with Boolean operators:

```text
team:Argentina AND round:Final
```

### Phrase Search

Quoted phrases are supported:

```text
"extra time"
```

The system uses the term positions stored in the inverted index to verify that the query terms occur consecutively.

---

## 6. TF-IDF Ranking

After candidate documents are retrieved, they are ranked using a **TF-IDF** scoring model.

The system uses logarithmic term-frequency normalization:

$$
tf_{norm}(t,d)=\log(1+tf(t,d))
$$

and smoothed inverse document frequency:

$$
idf(t)=\log\left(\frac{N+1}{df(t)+1}\right)
$$

The document score is calculated as:

$$
score(q,d)=\sum_{t\in q}tf_{norm}(t,d)\times idf(t)
$$

where:

* \(N\) is the total number of documents
* \(df(t)\) is the number of documents containing term \(t\)
* \(tf(t,d)\) is the frequency of term \(t\) in document \(d\)

The resulting documents are sorted in descending order according to their TF-IDF scores.

The scores can also be normalized relative to the highest-scoring document.

---

## 7. Evaluation

The system includes a dedicated evaluation module in `evaluation.py`.

A total of **10 evaluation queries** are defined using programmatically generated relevance judgments.

Examples include:

| Query                            | Description                         |
| -------------------------------- | ----------------------------------- |
| `messi`                          | Matches where Messi scored          |
| `penalty shootout`               | Matches decided by penalty shootout |
| `round:Final`                    | Matches in the Final stage          |
| `team:Argentina AND round:Final` | Argentina matches in the Final      |
| `extra time goal`                | Matches with extra-time goals       |
| `referee:Marciniak`              | Matches refereed by Marciniak       |
| `own goal`                       | Matches containing an own goal      |
| `round:Semi-finals team:Morocco` | Morocco matches in the Semi-finals  |
| `0-0`                            | Matches ending 0-0                  |
| `red card`                       | Matches with a red card             |

---

## 8. Evaluation Metrics

The retrieval system is evaluated using 10 predefined evaluation queries with programmatically constructed relevance judgments, which serve as a silver standard. The following information retrieval metrics are implemented:

### Precision@K

Measures the proportion of retrieved documents in the top K that are relevant.

$$
P@K=\frac{|Relevant\cap TopK|}{K}
$$

### Recall@K

Measures the proportion of all relevant documents retrieved within the top K results.

$$
R@K=\frac{|Relevant\cap TopK|}{|Relevant|}
$$

### F1@K

The harmonic mean of Precision@K and Recall@K.

$$
F1@K=\frac{2PR}{P+R}
$$

### Average Precision (AP)

Measures ranking quality by considering the precision at positions where relevant documents are retrieved.

### Mean Average Precision (MAP)

MAP is the mean Average Precision across all evaluation queries.

### NDCG@K

Normalized Discounted Cumulative Gain takes the ranking position of relevant documents into account, giving greater importance to relevant documents appearing near the top of the result list.

---

## 9. Evaluation Results

Using the implemented evaluation queries with **K = 10**, the system achieved the following average metrics:

| Metric  | Average |
| ------- | ------: |
| P@10    |   0.800 |
| R@10    |   0.485 |
| F1@10   |   0.427 |
| MAP     |   0.482 |
| NDCG@10 |   0.971 |

The results show that the system generally places relevant documents near the top of the ranked results, while Recall@10 is naturally limited for queries with a large number of relevant documents because only the top 10 results are retrieved.

---

## 10. Running the Project

Place the dataset in the project directory:

```text
matches_1930_2022.csv
```

Then run the system using one of the following modes.

### Interactive Search

```bash
python main.py
```

This starts an interactive search interface.

### Direct Query

```bash
python main.py --query "messi"
```

### Full Evaluation

```bash
python main.py --evaluate
```

This runs all predefined evaluation queries and reports the retrieval metrics.

---

## Project Structure

```text
world-cup-search-engine/
│
├── main.py
├── config.py
├── preprocess.py
├── document_builder.py
├── index_builder.py
├── query_processor.py
├── ranking.py
├── evaluation.py
├── matches_1930_2022.csv
└── README.md
```

### Module Description

| File                  | Description                                      |
| --------------------- | ------------------------------------------------ |
| `main.py`             | Main entry point and search interface            |
| `config.py`           | Dataset configuration and column mapping         |
| `preprocess.py`       | Text normalization and tokenization              |
| `document_builder.py` | Converts match records into searchable documents |
| `index_builder.py`    | Builds inverted and field indexes                |
| `query_processor.py`  | Parses and resolves search queries               |
| `ranking.py`          | TF-IDF document ranking                          |
| `evaluation.py`       | Relevance judgments and evaluation metrics       |

---

## Technologies

* Python
* Pandas
* NumPy
* Regular Expressions
* TF-IDF
* Inverted Index
* Information Retrieval

---

## Key Concepts

This project demonstrates the implementation of several core Information Retrieval concepts:

* Text Preprocessing
* Tokenization
* Stopword Removal
* Unicode Normalization
* Document Representation
* Inverted Index
* Positional Indexing
* Field Indexing
* Boolean Retrieval
* Phrase Search
* TF-IDF Ranking
* Precision
* Recall
* F1 Score
* Average Precision
* MAP
* NDCG

---

## Limitations

The current implementation has several limitations:

* **No fuzzy search:** Typographical errors in queries may result in no matches.
* **No stemming:** Words such as `goal`, `goals`, and `scored` may be indexed as separate terms.
* **Limited Recall@10:** Queries with many relevant documents are inherently limited when only the top 10 results are considered.

---

## Course Information

This project was developed as part of the **Information Retrieval** course at the **University of Isfahan**.