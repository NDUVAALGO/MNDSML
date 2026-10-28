import marimo

__generated_with = "0.23.6+alx.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building Shamba Ndogo's Recommendation Engine
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Algorithmic personalization determines platform retention. When a recommendation engine merely reflects global popularity trends, it creates a generic experience that degrades user engagement. In this project, you construct a dual-mode recommendation engine for Shamba Ndogo by combining the architectural strengths of Content-Based and Collaborative Filtering. You will engineer a Term Frequency-Inverse Document Frequency (TF-IDF) feature matrix from dense textual metadata "soups" to solve the cold-start problem via cosine similarity, while simultaneously assembling a sparse user-variety utility matrix to calculate mean-centered taste profiles. By validating structural choices with Jaccard overlap scores and measuring prediction errors through Root Mean Squared Error (RMSE), you establish a robust, production-ready hybridization framework capable of translating metadata and rating histories into personalized discovery pipelines.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **This challenge is not graded.** It is here to build your footing before the graded work begins.

    ### Instructions

    - Do not add or remove cells in this notebook.
    - Answer the questions according to the specifications provided.
    - Use the provided **Expected output** blocks and test cells to verify your work before continuing.
    - The use of StackOverflow, Google, Generative AI tools, and any other online resources is permitted. Use AI to help you understand — not to shortcut the thinking. [Read the honor code here](https://drive.google.com/file/d/1atFOPUQRLz5slb4Q1ASXh8QQfKyXVqrw/preview).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    The Maji Ndogo Ministry of Agriculture has a distribution problem, not a data problem. Its **Shamba Ndogo** programme — the digital advisory platform rolled out to smallholder farmers after the survey digitisation work of earlier modules — now has thousands of registered farmers and a national seed catalog of 2,500 certified varieties. But when a farmer opens the app to choose next season's seed, the platform just shows the most-planted varieties nationwide: the same wall of popular maize hybrids every farmer sees, whatever their soil, rainfall, or crop history.

    You've been brought in as the consulting data scientist. In your first week, the programme lead — **Zuri Kamanzi**, who moved over from the digitisation team and can read a confusion matrix faster than a budget line — lays out the brief over cold tea:

    > *"I don't need a research paper. I need an engine that does two things. One: when a farmer is looking at a variety that worked for them, show ten more they'll actually want — even for a variety we certified last month that nobody has planted yet. Two: when we **do** have planting history, predict what rating a specific farmer would give a variety they haven't tried, and tell me how wrong we are in a number I can put on a slide for the Ministry. Build me both. Then show me where each one breaks."*

    That brief maps cleanly onto the two pillars of recommender systems, and onto this project:

    **Part A — Content-Based Filtering.** Recommend varieties by what a variety *is*: its agronomic traits, breeding institute, trial agronomists, and the free-text tags farmers have attached to it.

    **Part B — Collaborative Filtering.** Recommend by what farmers *do*: end-of-season satisfaction ratings (0.5–5 stars) from the Shamba Ndogo app, no agronomy metadata required.

    By the end, you will have built both engines, evaluated them honestly, and demonstrated exactly where each one fails — the cold-start problem for collaborative filtering, and the popularity-blindness of content-based filtering.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Instructions

    - Do **not** add or remove cells in this notebook.
    - Do **not** edit or remove the `### START FUNCTION` or `### END FUNCTION` comments.
    - Do **not** add any code outside of the functions you are required to edit.
    - Doing any of the above will lead to a mark of **0%**!
    - Answer the questions according to the specifications provided.
    - Use the cell beneath each question to check your function against the expected output.
    - Do **not** hard-code answers to the questions.
    - The use of online resources is permitted, but copying another student's code is a breach of the Honor Code.

    Good luck! 🎥
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import mean_squared_error

    return mean_squared_error, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the Data
    """)
    return


@app.cell
def _(pd):
    varieties = pd.read_csv('varieties.csv')
    agronomy  = pd.read_csv('agronomy_data.csv')
    tags      = pd.read_csv('tags.csv')
    ratings   = pd.read_csv('train.csv')

    print("Varieties shape:", varieties.shape)
    print("Agronomy shape: ", agronomy.shape)
    print("Tags shape:     ", tags.shape)
    print("Ratings shape:  ", ratings.shape)
    return agronomy, ratings, tags, varieties


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Expected output:**
    ```
    Varieties shape: (2500, 3)
    Agronomy shape:  (2202, 6)
    Tags shape:      (12000, 4)
    Ratings shape:   (98187, 4)
    ```
    The four files: `varieties.csv` is the certified national seed catalog (ID, name with release year, pipe-separated agronomic traits); `agronomy_data.csv` holds trial metadata from the certification registry (trial agronomists, breeding institute, days to maturity, seed cost, trait keywords) for ~88% of varieties; `tags.csv` holds free-text labels farmers attach to varieties in the app; `train.csv` holds end-of-season satisfaction ratings (0.5–5 stars).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part A — Content-Based Filtering

    Content-based filtering recommends varieties from their attributes: traits, breeder, trial team, keywords, and farmer tags. It needs no rating history, so it answers the cold-start problem directly — a brand-new variety can be recommended the moment its metadata exists.

    The engine has two moving parts: a **TF-IDF** representation of a text "soup" built from each variety's content, and **cosine similarity** to find the nearest varieties to any query.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 1: Building the Content Metadata Table

    Before vectorizing, we need one clean table: a single row per variety with all content columns merged in.

    ### Task
    Write a function `build_content_table` that merges and cleans the variety metadata.

    **Function specifications:**
    - Takes the `varieties` and `agronomy` DataFrames.
    - Merges them on `varietyId` with a **left join** (keep every variety, even those without registry data).
    - Adds an integer column `'year'` — the release year — extracted from the end of `variety`. Variety names end in `'(YYYY)'`, so the year is the four characters inside the final parentheses. Varieties with no parsable year get `0`.
    - Cleans `traits`: replace the literal string `'(no traits listed)'` with an empty string.
    - Fills missing values in `trial_agronomists`, `breeder`, and `trait_keywords` with empty strings.
    - Returns the merged, cleaned DataFrame.

    > **No regex needed.** Because every variety name ends in `(YYYY)`, you can grab the year with plain string slicing: take the last six characters (`variety.str[-6:]`, e.g. `'(1995)'`), then characters 1–4 of that (`.str[1:5]`, e.g. `'1995'`). Wrap it in `pd.to_numeric(..., errors='coerce')` so unparsable names become `NaN`, then `fillna(0)` and cast to `int`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Content table shape: (2500, 9)
    Year range: 1980 - 2023
    NaN in breeder after fillna: 0
     varietyId                      variety  year                                                 traits
             1   Kilimani Gold Maize (1995)  1995 DroughtTolerant|EarlyMaturing|HighYield|Staple|Rainfed
             2 Kilimani Gold Maize 2 (1999)  1999 DroughtTolerant|EarlyMaturing|HighYield|Staple|Rainfed
             3 Kilimani Gold Maize 3 (2010)  2010 DroughtTolerant|EarlyMaturing|HighYield|Staple|Rainfed
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def build_content_table(varieties, agronomy):
        # Your code here
        pass
    ### END FUNCTION
    return (build_content_table,)


@app.cell
def _(agronomy, build_content_table, varieties):
    content_df = build_content_table(varieties, agronomy)

    print(f"Content table shape: {content_df.shape}")
    print(f"Year range: {content_df['year'].min()} - {content_df['year'].max()}")
    print(f"NaN in breeder after fillna: {content_df['breeder'].isna().sum()}")
    print(content_df[['varietyId', 'variety', 'year', 'traits']].head(3).to_string(index=False))
    return (content_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 2: Incorporating User Tags

    User tags are a rich, human signal. When viewers tag a variety `'thought-provoking'` or `'visually stunning'`, they describe it in ways studio metadata never captures. But tags are noisy, so we keep only those used often enough to be meaningful.

    ### Task
    Write a function `aggregate_tags` that collapses tags into one string per variety, dropping rare tags.

    **Function specifications:**
    - Takes the `tags` DataFrame and an integer `min_count` (default `5`).
    - Lowercases every tag.
    - Drops any tag that appears fewer than `min_count` times **across the whole dataset**.
    - For each variety, joins all surviving tags into a single space-separated string.
    - Returns a DataFrame with columns `['varietyId', 'agg_tags']`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    (the exact tag strings depend on the data, but the shape is fixed):
    ```
    Varieties with aggregated tags: 2016
     varietyId                                agg_tags
             1  reliable high yield stores well drou...
             2  early maturing market favorite relia...
             3  strong stalks high yield recommended...
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def aggregate_tags(tags, min_count=5):
        # Your code here
        pass
    ### END FUNCTION
    return (aggregate_tags,)


@app.cell
def _(aggregate_tags, tags):
    tag_agg = aggregate_tags(tags)

    print(f"Varieties with aggregated tags: {len(tag_agg)}")
    print(tag_agg.head(3).to_string(index=False))
    return (tag_agg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 3: Building the TF-IDF Content Soup

    Each variety becomes a vector in a space where every dimension is a vocabulary term. **TF-IDF** down-weights terms that appear across many varieties (like `'staple'`) and up-weights distinctive ones (like `'UkulimaSeedInstitute'`).

    We build a "soup" per variety — traits, breeding institute, top trial agronomists, trait keywords, and farmer tags concatenated into one string — then vectorize the whole corpus.

    ### Task
    Write a function `build_tfidf_matrix` that merges the tags, constructs the soup, and fits a TF-IDF vectorizer.

    **Function specifications:**
    - Takes `content_df` and `tag_agg`.
    - Left-merges `tag_agg` onto `content_df` on `varietyId`; fills missing `agg_tags` with `''`.
    - Builds a `'soup'` string per variety by space-joining:
      - `traits` with `'|'` replaced by spaces,
      - `breeder` with spaces **removed** (`'Ukulima Seed Institute'` → `'UkulimaSeedInstitute'`),
      - the **first 5** trial agronomists from `trial_agronomists` (split on `'|'`, take `[:5]`), each with spaces removed,
      - `trait_keywords` with `'|'` replaced by spaces,
      - `agg_tags`.
    - Fits a `TfidfVectorizer(stop_words='english')` on the `soup` column.
    - Returns `(tfidf_matrix, vectorizer, content_df)` — the sparse matrix, the fitted vectorizer, and the updated DataFrame with `'soup'` added.

    > **Why remove spaces from names?** `'Amara Chirwa'` → `'AmaraChirwa'` becomes one distinctive token instead of two separate words, so the vectorizer treats the agronomist — or the institute — as a single feature. Varieties trialled by the same team end up sharing rare, high-weight tokens.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    (vocabulary size varies a little with the data version):
    ```
    TF-IDF matrix shape: (2500, 391)
    Vocabulary size: 391
    Kilimani Gold soup (first 90 chars): DroughtTolerant EarlyMaturing HighYield Staple Rainfed UkulimaSeedInstitute AmaraChirwa Kw...
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def build_tfidf_matrix(content_df, tag_agg):
        # Your code here
        pass
    ### END FUNCTION
    return (build_tfidf_matrix,)


@app.cell
def _(build_tfidf_matrix, content_df, tag_agg):
    tfidf_matrix, vectorizer, content_df_1 = build_tfidf_matrix(content_df, tag_agg)
    print(f'TF-IDF matrix shape: {tfidf_matrix.shape}')
    print(f'Vocabulary size: {len(vectorizer.vocabulary_)}')
    kg_idx = content_df_1[content_df_1['variety'].str.startswith('Kilimani Gold Maize (1995)')].index[0]
    print(f"Kilimani Gold soup (first 90 chars): {content_df_1.loc[kg_idx, 'soup'][:90]}...")
    return content_df_1, tfidf_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 4: Content-Based Recommendations

    With the TF-IDF matrix built, a recommendation is just a cosine-similarity lookup.

    ### Task
    Write a function `get_content_recommendations` returning the top-N varieties most similar to a query title.

    **Function specifications:**
    - Takes `title` (str), `content_df`, `tfidf_matrix`, and `top_n` (int, default `10`).
    - Finds the row index of the first variety whose `title` **contains** `title` (case-insensitive substring). Return an empty DataFrame if there's no match.
    - Computes cosine similarity between that variety's vector and all varieties.
    - Returns the top `top_n` varieties (excluding the query itself) as a DataFrame with columns `['variety', 'traits', 'breeder', 'similarity_score']`, sorted descending by `similarity_score`.

    > **Note:** `cosine_similarity(tfidf_matrix[idx], tfidf_matrix)` returns shape `(1, n_varieties)` — flatten it before sorting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    (a coherent cluster of sibling releases from the same breeding programme, scores descending):
    ```
                           variety  similarity_score
      Kilimani Gold Maize 3 (2010)          0.474380
     Kilimani Pearl Sorghum (1998)          0.420025
         Kilimani Star Bean (2013)          0.370199
    Ukulima Crest Groundnut (2016)          0.357212
      Kilimani Gold Maize 2 (1999)          0.355963
      Kilimani Gold Maize 4 (2019)          0.353956
      Sweet Potato Giant 38 (2014)          0.318295
        Groundnut Spring 18 (2015)          0.307680
              Wheat Star 20 (1990)          0.296585
         Hawassa Crest Bean (1989)          0.291874
    ```
    The neighbors should be visibly sensible — the top six are all releases from the Kilimani Gold / Ukulima Seed Institute programme, clustering on shared traits, breeder, and trial team — because the soup captures all three at once.
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def get_content_recommendations(variety_name, content_df, tfidf_matrix, top_n=10):
        # Your code here
        pass
    ### END FUNCTION
    return (get_content_recommendations,)


@app.cell
def _(content_df_1, get_content_recommendations, tfidf_matrix):
    recs = get_content_recommendations('Kilimani Gold Maize (1995)', content_df_1, tfidf_matrix, top_n=10)
    print(recs[['variety', 'similarity_score']].to_string(index=False))
    return (recs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 5: Trait Overlap Validation

    Cosine scores say how close two varieties sit in content space, but we can sanity-check them: do the recommended varieties actually share traits with the query?

    ### Task
    Write a function `trait_overlap_score` returning the mean **Jaccard similarity** between the query's trait set and each recommendation's trait set.

    Jaccard similarity of sets A and B is `|A ∩ B| / |A ∪ B|`.

    **Function specifications:**
    - Takes `query_title` (str), `recommendations` (a DataFrame with a `'traits'` column), and `content_df`.
    - Returns the mean Jaccard trait overlap across all recommendations as a float in `[0, 1]`.
    - Returns `0.0` if the query can't be found or the recommendations are empty.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Mean trait Jaccard overlap: 0.5210 (typically 0.30 - 0.85 for well-matched content recommendations)
    ```
    A solid overlap confirms the soup is capturing real agronomic similarity, not keyword coincidence.
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def trait_overlap_score(query_title, recommendations, content_df):
        # Your code here
        pass
    ### END FUNCTION
    return (trait_overlap_score,)


@app.cell
def _(content_df_1, recs, trait_overlap_score):
    overlap = trait_overlap_score('Kilimani Gold Maize (1995)', recs, content_df_1)
    print(f'Mean trait Jaccard overlap: {overlap:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part B — Collaborative Filtering

    Content-based filtering uses what a variety *is*. Collaborative filtering uses what *similar farmers* thought of it. The premise — people who agreed in the past tend to agree in the future — is one of the most empirically reliable ideas in recommender systems.

    We build a **user–variety utility matrix**, measure user-to-user similarity with mean-centered cosine similarity, and predict a target user's rating for an unseen variety as a similarity-weighted average of similar farmers' ratings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 6: Building the Utility Matrix

    The utility matrix is the core of collaborative filtering: rows are farmers, columns are varieties, cells are end-of-season ratings (or empty). The full 2,200 × 2,500 matrix is impractical, so we focus on the most active farmers and most-rated varieties.

    ### Task
    Write a function `build_utility_matrix`.

    **Function specifications:**
    - Takes `ratings`, `n_farmers` (int, default `200`), and `n_varieties` (int, default `100`).
    - Selects the `n_farmers` farmers with the most ratings.
    - Among those farmers' ratings, selects the `n_varieties` varieties with the most ratings.
    - Keeps only rows where both `farmerId` and `varietyId` are in the selected sets.
    - Pivots into a utility matrix: index `farmerId`, columns `varietyId`, values `rating`; unobserved cells stay `NaN`.
    - Returns the utility matrix DataFrame.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Utility matrix shape: (200, 100)
    Sparsity: 71.78% missing
    ```
    Even the most active farmers have planted fewer than a third of the popular varieties — the sparsity that makes collaborative filtering both necessary and hard.
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def build_utility_matrix(ratings, n_farmers=200, n_varieties=100):
        # Your code here
        pass
    ### END FUNCTION
    return (build_utility_matrix,)


@app.cell
def _(build_utility_matrix, ratings):
    utility_matrix = build_utility_matrix(ratings)

    print(f"Utility matrix shape: {utility_matrix.shape}")
    sparsity = utility_matrix.isna().sum().sum() / utility_matrix.size
    print(f"Sparsity: {sparsity:.2%} missing")
    return (utility_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even among the most active farmers and most-rated varieties, many cells are empty. That sparsity is the central obstacle collaborative filtering has to work around.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 7: User Similarity

    Now we measure how alike any two farmers are, from their overlapping ratings. We use **cosine similarity on mean-centered ratings**: subtracting each user's average rating first removes the bias of farmers who habitually rate high or low, so we compare *taste*, not generosity.

    ### Task
    Write a function `compute_user_similarity` returning the user–user similarity matrix.

    **Function specifications:**
    - Takes the `utility_matrix`.
    - Mean-centers each user: subtract that user's mean rating from their entries, then fill `NaN` with `0`.
    - Computes cosine similarity across all farmers.
    - Returns a DataFrame with `farmerId` as both index and columns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Similarity matrix shape: (200, 200)
    Diagonal (should be 1.0): [1. 1. 1.]
    Symmetric: True
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def compute_user_similarity(utility_matrix):
        # Your code here
        pass
    ### END FUNCTION
    return (compute_user_similarity,)


@app.cell
def _(compute_user_similarity, np, utility_matrix):
    user_sim = compute_user_similarity(utility_matrix)

    print(f"Similarity matrix shape: {user_sim.shape}")
    print(f"Diagonal (should be 1.0): {user_sim.values.diagonal()[:3]}")
    print(f"Symmetric: {np.allclose(user_sim.values, user_sim.values.T)}")
    return (user_sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 8: Predicting a Rating

    Given a target user and a variety they haven't seen, we predict their rating by weighting similar farmers' ratings by how similar they are to the target.

    $$\hat{r}_{u,m} = \bar{r}_u + \frac{\sum_{v \in N} \text{sim}(u, v)\,(r_{v,m} - \bar{r}_v)}{\sum_{v \in N} |\text{sim}(u, v)|}$$

    where $\bar{r}_u$ is user $u$'s mean rating and $N$ is the top-k most similar farmers who *have* rated variety $m$.

    ### Task
    Write a function `predict_rating`.

    **Function specifications:**
    - Takes `farmer_id`, `variety_id`, `utility_matrix`, `user_sim`, and `top_k` (int, default `10`).
    - Finds the `top_k` most similar farmers to `farmer_id` who have rated `variety_id`.
    - Applies the formula above.
    - Returns the prediction clipped to `[0.5, 5.0]`.
    - Edge cases: if `farmer_id` isn't in the matrix **or** the user has no ratings at all, return `3.0`. If `variety_id` isn't a column, or no similar user has rated it, return the user's clipped mean rating.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Predicted rating for farmer 10 on variety 3: 2.76
    In valid range [0.5, 5.0]: True
    Cold (unknown) farmer falls back to 3.0: 3.0
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def predict_rating(farmer_id, variety_id, utility_matrix, user_sim, top_k=10):
        # Your code here
        pass
    ### END FUNCTION
    return (predict_rating,)


@app.cell
def _(predict_rating, user_sim, utility_matrix):
    sample_farmer = utility_matrix.index[0]
    unrated = utility_matrix.columns[utility_matrix.loc[sample_farmer].isna()]
    sample_variety = unrated[0]

    predicted = predict_rating(sample_farmer, sample_variety, utility_matrix, user_sim)
    print(f"Predicted rating for farmer {sample_farmer} on variety {sample_variety}: {predicted:.2f}")
    print(f"In valid range [0.5, 5.0]: {0.5 <= predicted <= 5.0}")
    print(f"Cold (unknown) farmer falls back to 3.0: {predict_rating(-1, sample_variety, utility_matrix, user_sim)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 9: Evaluating with RMSE

    $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{r}_i - r_i)^2}$$

    This is the number Zuri wants on her slide for the Ministry.

    ### Task
    Write a function `evaluate_collaborative_rmse` measuring accuracy on held-out ratings.

    **Function specifications:**
    - Takes `utility_matrix`, `user_sim`, `ratings`, `test_fraction` (float, default `0.2`), and `random_state` (int, default `42`).
    - For each farmer with at least 2 ratings in the utility matrix, randomly hold out `test_fraction` of their rated varieties (at least 1).
    - Predict each held-out pair with `predict_rating` (using the full matrix — a simple in-place evaluation).
    - Return the RMSE across all held-out pairs as a float.

    > Use `np.random.default_rng(random_state)` so the sampling is reproducible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Collaborative Filtering RMSE: 0.4448
    Global-mean baseline RMSE:    0.8633
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def evaluate_collaborative_rmse(utility_matrix, user_sim, ratings,
                                    test_fraction=0.2, random_state=42):
        # Your code here
        pass
    ### END FUNCTION
    return (evaluate_collaborative_rmse,)


@app.cell
def _(
    evaluate_collaborative_rmse,
    mean_squared_error,
    np,
    ratings,
    user_sim,
    utility_matrix,
):
    rmse = evaluate_collaborative_rmse(utility_matrix, user_sim, ratings)
    baseline = np.sqrt(mean_squared_error(
        ratings['rating'][:10000], [ratings['rating'].mean()] * 10000))
    print(f"Collaborative Filtering RMSE: {rmse:.4f}")
    print(f"Global-mean baseline RMSE:    {baseline:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Beating the global-mean baseline is the bar: it means the engine is learning real taste, not just echoing the average rating.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Challenge 10: The Cold-Start Problem in Practice

    This is the trade-off Zuri asked you to make concrete: where does each mode break?

    Every season the Ministry certifies new varieties. On release day, a new variety has full agronomy metadata — traits, breeder, trial team — and **zero** farmer ratings. Collaborative filtering has nothing to work with; content-based filtering works immediately.

    ### Task
    Write a function `cold_start_comparison` showing what happens when collaborative filtering meets a variety with almost no rating history.

    **Function specifications:**
    - Takes `content_df`, `tfidf_matrix`, `utility_matrix`, `new_variety_name` (str), and `top_n` (int, default `5`).
    - Finds the variety's `varietyId` via `new_variety_name` in `content_df`, then checks whether any farmer in the utility matrix has rated it.
    - Generates content-based recommendations for the variety via `get_content_recommendations`.
    - Returns a dict `{'content_recs': DataFrame, 'collab_possible': bool}`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected output
    ```
    Query variety: Dawn Millet of Sokoto (1991)
    Collaborative filtering possible: False

    Content-based recommendations:
                             variety  similarity_score
    Sokoto Royal Sweet Potato (2006)          0.376321
    Shield Cowpea of Kilimani (2013)          0.372109
              Maize Shield 75 (1993)          0.326138
                  Tea Dawn 98 (1981)          0.323195
            Hawassa Pearl Tea (1994)          0.311394
    ```
    """)
    return


@app.cell
def _():
    ### START FUNCTION
    def cold_start_comparison(content_df, tfidf_matrix, utility_matrix, new_variety_name, top_n=5):
        # Your code here
        pass
    ### END FUNCTION
    return (cold_start_comparison,)


@app.cell
def _(
    cold_start_comparison,
    content_df_1,
    ratings,
    tfidf_matrix,
    utility_matrix,
):
    # Find a variety that is rarely rated overall — a realistic cold-start case.
    rating_counts = ratings['varietyId'].value_counts()
    rare_ids = rating_counts[rating_counts < 5].index
    rare_with_meta = content_df_1[content_df_1['varietyId'].isin(rare_ids) & content_df_1['breeder'].ne('')]['variety'].values
    test_name = rare_with_meta[0] if len(rare_with_meta) > 0 else 'Tanari Short-Season Bean (2023)'
    result = cold_start_comparison(content_df_1, tfidf_matrix, utility_matrix, test_name)
    print(f'Query variety: {test_name}')
    print(f"Collaborative filtering possible: {result['collab_possible']}")
    print('\nContent-based recommendations:')
    print(result['content_recs'][['variety', 'similarity_score']].to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the whole argument in one cell: collaborative filtering has nothing to say about a variety with no rating history, while content-based filtering responds instantly from metadata alone. Shamba Ndogo needs both — content-based to survive the cold start, collaborative to sharpen recommendations once ratings accumulate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Wrapping up

    You delivered exactly what Zuri asked for: a dual-mode recommendation engine for Shamba Ndogo, and an honest account of where each mode fails.

    **Part A — Content-Based Filtering.** You merged the seed catalog with the certification registry into a clean content table (no regex required), folded in farmer tags, built a TF-IDF "soup" per variety, and ranked neighbors by cosine similarity. You validated the output with a Jaccard trait-overlap check. This is Shamba Ndogo's cold-start engine: it works the moment a variety is certified, with zero plantings.

    **Part B — Collaborative Filtering.** You built a farmer–variety utility matrix, measured taste with mean-centered cosine similarity, predicted unseen ratings with a similarity-weighted average, and scored the system with RMSE against a global-mean baseline — the number for Zuri's slide to the Ministry.

    **The trade-off, made concrete.** The cold-start comparison showed the seam between the two: collaborative filtering is silent on varieties with no planting history; content-based filtering speaks immediately. A production recommender hands newly certified varieties to the content engine and lets the collaborative engine take over as ratings accumulate — the architecture Shamba Ndogo now has the pieces to build.

    For your portfolio, this is a complete recommender-systems case study — text preprocessing, TF-IDF, cosine similarity, collaborative filtering, and a defensible evaluation — framed end to end around a real agricultural product decision.

    Ship it.

    — Zuri
    """)
    return




# ════════════════════════════════════════════════════════════════════════
# Pre-submission readiness check
#
# This cell has NO declared dependencies — it runs even when other cells
# have not. It uses dynamic name lookup (eval + try/except) so the learner
# always gets actionable feedback, including a "run all cells first" hint
# when nothing has been defined yet.
#
# The spec below is auto-generated at conversion time from the test_suite
# (which functions pytest references) and the model solution (the canonical
# parameter names + body characteristics).
# ════════════════════════════════════════════════════════════════════════

@app.cell
def cell_readiness():
    import inspect as _inspect
    import ast as _ast
    _spec = {'aggregate_tags': {'expected_params': ['tags', 'min_count']}, 'build_content_table': {'expected_params': ['varieties', 'agronomy']}, 'build_tfidf_matrix': {'expected_params': ['content_df', 'tag_agg']}, 'build_utility_matrix': {'expected_params': ['ratings', 'n_farmers', 'n_varieties']}, 'cold_start_comparison': {'expected_params': ['content_df', 'tfidf_matrix', 'utility_matrix', 'new_variety_name', 'top_n']}, 'compute_user_similarity': {'expected_params': ['utility_matrix']}, 'evaluate_collaborative_rmse': {'expected_params': ['utility_matrix', 'user_sim', 'ratings', 'test_fraction', 'random_state']}, 'get_content_recommendations': {'expected_params': ['variety_name', 'content_df', 'tfidf_matrix', 'top_n']}, 'predict_rating': {'expected_params': ['farmer_id', 'variety_id', 'utility_matrix', 'user_sim', 'top_k']}, 'trait_overlap_score': {'expected_params': ['query_title', 'recommendations', 'content_df']}}
    _msgs = []
    _missing = 0
    _stubs = 0
    _mismatches = 0
    for _name, _checks in _spec.items():
        try:
            _fn = eval(_name)
        except NameError:
            _msgs.append(f"❌ `{_name}` — not defined yet")
            _missing += 1
            continue
        if not callable(_fn):
            _msgs.append(f"❌ `{_name}` — exists but is not callable")
            _missing += 1
            continue
        try:
            _sig = _inspect.signature(_fn)
            _actual_params = list(_sig.parameters.keys())
        except (ValueError, TypeError):
            _msgs.append(f"✓ `{_name}` — defined (signature unavailable)")
            continue
        _expected = _checks.get("expected_params")
        if _expected and _actual_params != _expected:
            _msgs.append(
                f"⚠️ `{_name}` — parameter mismatch: expected {_expected}, got {_actual_params}"
            )
            _mismatches += 1
            continue
        # AST check — flag empty stubs (function body is just `pass` or `return None`)
        _is_stub = False
        try:
            import textwrap as _textwrap
            _src = _textwrap.dedent(_inspect.getsource(_fn))
            _tree = _ast.parse(_src)
            _body = _tree.body[0].body if _tree.body else []
            # Strip any leading docstring (an Expr with a Constant str)
            _real_body = [
                _stmt for _stmt in _body
                if not (
                    isinstance(_stmt, _ast.Expr)
                    and isinstance(_stmt.value, _ast.Constant)
                    and isinstance(_stmt.value.value, str)
                )
            ]
            _is_stub = (
                not _real_body
                or (
                    len(_real_body) == 1
                    and (
                        isinstance(_real_body[0], _ast.Pass)
                        or (isinstance(_real_body[0], _ast.Return) and _real_body[0].value is None)
                    )
                )
            )
        except (OSError, TypeError, SyntaxError, IndexError):
            pass  # source unavailable; skip stub check
        if _is_stub:
            _msgs.append(
                f"⚠️ `{_name}` — body looks empty (just `pass` or bare `return`). Did you implement it?"
            )
            _stubs += 1
            continue
        _msgs.append(f"✓ `{_name}` — defined with parameters {_actual_params}")
    print("Readiness check:")
    for _m in _msgs:
        print(f"  {_m}")
    print()
    if _missing == len(_spec):
        print("It looks like none of the required functions are defined yet.")
        print("Click 'Run all cells' (the ▶▶ button at the top of the page),")
        print("or run each function-defining cell individually first.")
    elif _missing > 0:
        print(f"{_missing} function(s) not yet defined — run the cells that define them and re-run this check.")
    elif _stubs > 0 or _mismatches > 0:
        print(f"Some functions need fixing before submitting ({_stubs} stub(s), {_mismatches} signature mismatch(es)).")
    else:
        print("All required functions are present and ready. You can submit.")
    return


if __name__ == "__main__":
    app.run()

