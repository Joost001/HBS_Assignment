# Installation

pip install -r requirements.txt

# File Inventory
- HBS_retrieval_assignment.ipynb : original code
- HBS_retrieval_assignment.py : original code converted to .py
- requirements.txt : required packages
- recommendation_engine.py : response to coding prompts
- test_recommendation_engine.py : unittest for methods and classes

# Recommendation Engine

This recommendation engine script allows you to run a search engine and evaluate its performance using different methods and preprocessing options. The script can be run with various arguments to customize the search and evaluation process.

## Arguments

### `--directory`
- **Type:** `str`
- **Default:** `"WANDS/dataset"`
- **Description:** Specifies the directory where the dataset files (`query.csv`, `product.csv`, `label.csv`) are located.

### `--method`
- **Type:** `str`
- **Default:** `'bm25'`
- **Choices:** `['bm25', 'tfidf']`
- **Description:** Defines the search engine method to use. Options are `'bm25'` for BM25 search engine or `'tfidf'` for TF-IDF search engine.

### `--query`
- **Type:** `str`
- **Description:** The search query to use. If provided, the script will run the search for this query.

### `--lemmatize_off`
- **Type:** `bool`
- **Action:** `store_false`
- **Default:** `True`
- **Description:** Turn lemmatization off. By default, lemmatization is applied to the text.

### `--stemming_off`
- **Type:** `bool`
- **Action:** `store_false`
- **Default:** `True`
- **Description:** Turn stemming off. By default, stemming is applied to the text.

### `--use_synonyms`
- **Type:** `bool`
- **Action:** `store_true`
- **Default:** `False`
- **Description:** Turn synonyms on. If enabled, the text will be expanded with synonyms.

### `--dont_remove_stop_words`
- **Type:** `bool`
- **Action:** `store_false`
- **Default:** `True`
- **Description:** Don't remove stop words. By default, stop words are removed from the text.

### `--run_demo`
- **Type:** `bool`
- **Action:** `store_true`
- **Default:** `False`
- **Description:** Run the demo function. If enabled, the script will run a demo showcasing the search engine's functionality.

### `--k`
- **Type:** `int`
- **Default:** `10`
- **Description:** Number of top elements to consider in the search results.

## Usage

To run the recommendation engine with the default settings, you can use the following command:

```bash
python recommendation_engine.py
```

To specify a search query and use the BM25 method, you can use:

```bash
python recommendation_engine.py --query "comfortable chair" --method bm25
```

To turn off lemmatization and stemming, and use synonyms:

```bash
python recommendation_engine.py --query "comfortable chair" --lemmatize_off --stemming_off --use_synonyms
```

To run the demo function:

```bash
python recommendation_engine.py --run_demo
```

This README file provides an overview of how to use the different arguments to customize the search and evaluation process with the recommendation engine script. Adjust the arguments as needed to fit your specific use case.

---