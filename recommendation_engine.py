import argparse
import logging
import re
from typing import Tuple, Any, List, Optional

import nltk
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Preprocessing setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatize=False, stemming=False, synonyms=False, remove_stop_words=False):
        """
        Initialize the TextPreprocessor with various preprocessing options.

        Parameters:
        lemmatize (bool): Whether to apply lemmatization.
        stemming (bool): Whether to apply stemming.
        synonyms (bool): Whether to expand words with their synonyms.
        remove_stop_words (bool): Whether to remove stop words.
        """
        self.lemmatize = lemmatize
        self.stemming = stemming
        self.synonyms = synonyms
        self.remove_stop_words = remove_stop_words
        if lemmatize:
            from nltk.stem import WordNetLemmatizer
            self.lemmatizer = WordNetLemmatizer()
        if stemming:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
        if remove_stop_words:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def get_synonyms(word):
        """
        Retrieve a list of synonyms for a given word.

        Parameters:
        word (str): The word to find synonyms for.

        Returns:
        list: A list of synonym words.
        """
        from nltk.corpus import wordnet
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    def preprocess_text(self, text):
        """
        Preprocess a single text document by applying specified text processing steps.

        Parameters:
        text (str): The text document to preprocess.

        Returns:
        str: The preprocessed text.
        """
        if pd.isna(text):
            return ''  # Return empty string if the input text is NaN
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()  # Split text into words

        # Remove stop words if specified
        if self.remove_stop_words:
            words = [word for word in words if word not in self.stop_words]

        # Apply lemmatization if specified
        if self.lemmatize:
            words = [self.lemmatizer.lemmatize(word) for word in words]

        # Apply stemming if specified
        if self.stemming:
            words = [self.stemmer.stem(word) for word in words]

        # Expand words with synonyms if specified
        if self.synonyms:
            expanded_words = []
            for word in words:
                expanded_words.append(word)  # Add original word
                expanded_words.extend(self.get_synonyms(word))  # Add synonyms
            words = expanded_words

        return ' '.join(words)  # Join words back into a single string

    def fit(self, X, y=None):
        """
        Fit the transformer (does not perform any fitting in this case).

        Parameters:
        X (pd.Series): Series of text documents to fit.
        y (optional): Ignored.

        Returns:
        self: The fitted transformer.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform the input data by applying the preprocessing steps.

        Parameters:
        X (pd.Series): Series of text documents to transform.
        y (optional): Ignored.

        Returns:
        pd.Series: Series of preprocessed text documents.
        """
        if isinstance(X, pd.Series):
            return X.apply(self.preprocess_text)  # Apply preprocessing to each text in the Series
        else:
            return pd.Series(
                [self.preprocess_text(text) for text in X])  # Convert list of texts to a Series and preprocess


class SearchEngines:
    class BM25SearchEngine(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self.bm25 = None
            self.product_ids = None

        def fit(self, X: pd.Series, y: pd.Series = None):
            """
            Fit the BM25 model on the provided text data.

            Parameters:
            X (pd.Series): Series of documents (product descriptions).
            y (pd.Series): Series of product IDs corresponding to the documents.
            """
            # Assert that X is a pandas Series of strings
            assert isinstance(X, pd.Series) and all(
                isinstance(doc, str) for doc in X), "X must be a pandas Series of strings"
            # Assert that y is a pandas Series
            assert isinstance(y, pd.Series), "y must be a pandas Series"

            # Tokenize the corpus
            tokenized_corpus = [doc.split(" ") for doc in X]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.product_ids = y.tolist()
            return self

        def predict(self, X: pd.Series, k: int = 10) -> pd.Series:
            """
            Predict the top product IDs for the given queries.

            Parameters:
            X (pd.Series): Series of queries.
            k (int): Number of top elements to consider.

            Returns:
            pd.Series: Series of lists containing top product IDs for each query.
            """
            # Assert that X is a pandas Series of strings
            assert isinstance(X, pd.Series) and all(
                isinstance(query, str) for query in X), "X must be a pandas Series of strings"

            top_product_indices = []
            for query in X:
                tokenized_query = query.split(" ")
                top_product_indices.append(self.bm25.get_top_n(tokenized_query, self.product_ids, n=k))
            return pd.Series(top_product_indices)

    class TFIDFSearchEngine(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self.vectorizer = None
            self.tfidf_matrix = None
            self.product_ids = None

        def fit(self, X: pd.Series, y: pd.Series = None):
            """
            Fit the TF-IDF model on the provided text data.

            Parameters:
            X (pd.Series): Series of documents (product descriptions).
            y (pd.Series): Series of product IDs corresponding to the documents.
            """
            # Assert that X is a pandas Series of strings
            assert isinstance(X, pd.Series) and all(
                isinstance(doc, str) for doc in X), "X must be a pandas Series of strings"
            # Assert that y is a pandas Series
            assert isinstance(y, pd.Series), "y must be a pandas Series"

            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(X)
            self.product_ids = y.tolist()
            return self

        def predict(self, X: pd.Series, k: int = 10) -> pd.Series:
            """
            Predict the top product IDs for the given queries.

            Parameters:
            X (pd.Series): Series of queries.
            k (int): Number of top elements to consider.

            Returns:
            pd.Series: Series of lists containing top product IDs for each query.
            """
            # Assert that X is a pandas Series of strings
            assert isinstance(X, pd.Series) and all(
                isinstance(query, str) for query in X), "X must be a pandas Series of strings"

            query_vectors = self.vectorizer.transform(X)
            top_product_indices = []
            for query_vector in query_vectors:
                cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                top_indices = cosine_similarities.argsort()[-k:][::-1]
                top_product_indices.append([self.product_ids[i] for i in top_indices])
            return pd.Series(top_product_indices)


class PerformanceMetrics:
    @staticmethod
    def map_at_k(true_ids: np.ndarray, predicted_ids: List[Any], true_labels: Optional[np.ndarray] = None, k: int = 10,
                 partial_matches: bool = False) -> float:
        """
        Calculate the Mean Average Precision at K (MAP@K) with optional partial match handling.

        Parameters:
        true_ids (np.ndarray): Array of relevant product IDs.
        predicted_ids (List[Any]): List of predicted product IDs.
        true_labels (Optional[np.ndarray]): Array of labels corresponding to the true_ids
                    ("Exact", "Partial", "Irrelevant"). Required if partial_matches is True.
        k (int): Number of top elements to consider.
        partial_matches (bool): Whether to consider partial matches in the scoring.

        Returns:
        float: MAP@K score.

        Raises:
        AssertionError: If the input types are not as expected.
        """
        # Assert that true_ids is a numpy array
        assert isinstance(true_ids, np.ndarray), "true_ids must be a numpy.ndarray"

        # Assert that predicted_ids is a list
        assert isinstance(predicted_ids, list), "predicted_ids must be a list"

        # Assert that true_labels is a list of strings if partial_matches is True
        if partial_matches:
            assert true_labels is not None, "true_labels must be provided if partial_matches is True"
            assert isinstance(true_labels, np.ndarray), "true_labels must be a numpy.ndarray"
            assert all(isinstance(label, str) for label in true_labels), "true_labels must be a list of strings"

        # Assert that k is a positive integer
        assert isinstance(k, int) and k > 0, "k must be a positive integer"

        # If either list is empty, return 0.0
        if not len(true_ids) or not len(predicted_ids):
            return 0.0

        score = 0.0
        num_hits = 0.0
        seen_ids = set()

        for i, p_id in enumerate(predicted_ids[:k]):
            if p_id in true_ids and p_id not in seen_ids:
                seen_ids.add(p_id)
                if partial_matches and true_labels is not None:
                    idx = np.where(true_ids == p_id)[0][0]
                    label = true_labels[idx]
                    if label == "Exact":
                        num_hits += 1.0
                    elif label == "Partial":
                        num_hits += 0.5
                else:
                    num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(true_ids), k)


def load_data(directory: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from CSV files located in the specified directory.

    Parameters:
    directory (str): The directory where the dataset files are located.

    Returns:
    tuple: A tuple containing three pandas DataFrames: query_df, product_df, and label_df.

    Raises:
    AssertionError: If the directory is not a string or the files are not found.
    """
    # Assert that the directory is a string
    assert isinstance(directory, str), "directory must be a string"

    # Construct file paths
    query_file = f"{directory}/query.csv"
    product_file = f"{directory}/product.csv"
    label_file = f"{directory}/label.csv"

    try:
        # Load data from CSV files
        query_df = pd.read_csv(query_file, sep='\t')
        product_df = pd.read_csv(product_file, sep='\t')
        label_df = pd.read_csv(label_file, sep='\t')

        # Log success message
        logging.info("Data loaded successfully")

        return query_df, product_df, label_df
    except FileNotFoundError as fnf_error:
        logging.error("File not found: %s", fnf_error)
        raise
    except pd.errors.ParserError as parser_error:
        logging.error("Error parsing CSV file: %s", parser_error)
        raise
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise


def initialize_search_engine(product_df: pd.DataFrame, method: str = 'bm25', lemmatize: bool = True,
                             stemming: bool = True, synonyms: bool = False,
                             remove_stop_words: bool = True) -> Pipeline:
    """
    Initialize a search engine pipeline with the specified method.

    Parameters:
    product_df (pd.DataFrame): DataFrame containing product information with 'product_name', 'product_description', and 'product_id' columns.
    method (str): The search engine method to use. Can be 'bm25' or 'tfidf'. Defaults to 'bm25'.

    Returns:
    Pipeline: A scikit-learn Pipeline object with the configured search engine.

    Raises:
    AssertionError: If the input DataFrame does not contain the required columns.
    """
    # Assert that product_df is a DataFrame
    assert isinstance(product_df, pd.DataFrame), "product_df must be a pandas DataFrame"

    # Assert that the DataFrame contains the necessary columns
    required_columns = ['product_name', 'product_description', 'product_id']
    for column in required_columns:
        assert column in product_df.columns, f"Missing required column: {column}"

    # Assert that method is a valid search engine method
    assert method in ['bm25', 'tfidf'], "Invalid search method. Choose 'bm25' or 'tfidf'."

    try:
        # Initialize search engine pipeline based on the specified method
        if method == 'bm25':
            search_engine = Pipeline([
                ('text_preprocessing', TextPreprocessor(
                    lemmatize=lemmatize, stemming=stemming, synonyms=synonyms, remove_stop_words=remove_stop_words)),
                ('bm25_search', SearchEngines.BM25SearchEngine())
            ])
        elif method == 'tfidf':
            search_engine = Pipeline([
                ('text_preprocessing', TextPreprocessor(
                    lemmatize=lemmatize, stemming=stemming, synonyms=synonyms, remove_stop_words=remove_stop_words)),
                ('tfidf_search', SearchEngines.TFIDFSearchEngine())
            ])

        # Combine 'product_name' and 'product_description' into a single text column
        product_texts = product_df['product_name'] + ' ' + product_df['product_description']

        # Fit the search engine on the product texts and IDs
        search_engine.fit(product_texts, product_df['product_id'])

        logging.info("Search engine initialized successfully with method %s", method)
        return search_engine
    except Exception as e:
        logging.error("Error initializing search engine with method %s: %s", method, e)
        raise


def get_matches_for_query(query_id: Any, label_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Get relevant product IDs and labels for a given query ID.

    Parameters:
    query_id (Any): The query ID to get matches for.
    label_df (pd.DataFrame): DataFrame containing labels with 'query_id', 'product_id', and 'label' columns.

    Returns:
    Tuple[pd.Series, pd.Series]: A tuple containing relevant product IDs and their corresponding labels.

    Raises:
    AssertionError: If the input types are not as expected or required columns are missing.
    """
    # Assert that label_df is a DataFrame
    assert isinstance(label_df, pd.DataFrame), "label_df must be a pandas DataFrame"

    # Assert that label_df contains the required columns
    required_columns = ['query_id', 'product_id', 'label']
    for column in required_columns:
        assert column in label_df.columns, f"Missing required column in label_df: {column}"

    try:
        # Group the DataFrame by 'query_id' and get the group corresponding to the given query_id
        query_group = label_df.groupby('query_id').get_group(query_id)

        # Extract relevant product IDs and labels
        relevant_ids = query_group['product_id'].values
        relevant_labels = query_group['label'].values

        return relevant_ids, relevant_labels
    except Exception as e:
        logging.error("Error getting matches for query_id %s: %s", query_id, e)
        raise


def calculate_map_k(query_df: pd.DataFrame, search_engine: Pipeline, label_df: pd.DataFrame,
                    k: int = 10) -> pd.DataFrame:
    """
    Calculate the Mean Average Precision at K (MAP@K) for each query in the query DataFrame.

    Parameters:
    query_df (pd.DataFrame): DataFrame containing search queries with 'query' and 'query_id' columns.
    search_engine (Pipeline): A scikit-learn Pipeline object with the configured search engine.
    label_df (pd.DataFrame): DataFrame containing labels with 'query_id', 'product_id', and 'label' columns.
    k (int): Number of top elements to consider.

    Returns:
    pd.DataFrame: Updated query_df with calculated MAP@K scores.

    Raises:
    AssertionError: If the input types are not as expected or required columns are missing.
    """
    # Assert that query_df, search_engine, and label_df are of correct types
    assert isinstance(query_df, pd.DataFrame), "query_df must be a pandas DataFrame"
    assert isinstance(search_engine, Pipeline), "search_engine must be a scikit-learn Pipeline object"
    assert isinstance(label_df, pd.DataFrame), "label_df must be a pandas DataFrame"

    # Assert that query_df contains the required columns
    required_query_columns = ['query', 'query_id']
    for column in required_query_columns:
        assert column in query_df.columns, f"Missing required column in query_df: {column}"

    # Assert that label_df contains the required columns
    required_label_columns = ['query_id', 'product_id', 'label']
    for column in required_label_columns:
        assert column in label_df.columns, f"Missing required column in label_df: {column}"

    try:
        # Predict top product IDs for each query
        query_df['top_product_ids'] = query_df['query'].apply(lambda x: search_engine.predict([x], k=k).iloc[0])

        # Get relevant product IDs and labels for each query
        query_df[['relevant_ids', 'relevant_labels']] = query_df['query_id'].apply(
            lambda qid: pd.Series(get_matches_for_query(qid, label_df)))

        # Calculate MAP@K without partial matches
        query_df['map@k_no_partial'] = query_df.apply(
            lambda x: PerformanceMetrics.map_at_k(x['relevant_ids'], x['top_product_ids'], k=k), axis=1)

        # Calculate MAP@K with partial matches
        query_df['map@k_with_partial'] = query_df.apply(
            lambda x: PerformanceMetrics.map_at_k(x['relevant_ids'], x['top_product_ids'], x['relevant_labels'], k=k,
                                                  partial_matches=True), axis=1)

        logging.info("MAP@K calculated successfully")
        return query_df
    except Exception as e:
        logging.error("Error calculating MAP@K: %s", e)
        raise


def get_top_product_ids_for_query(search_engine: Pipeline, query: str, k: int = 10) -> List[str]:
    """
    Get the top product IDs for a given query using the specified search engine.

    Parameters:
    search_engine (Pipeline): A scikit-learn Pipeline object with the configured search engine.
    query (str): The search query.
    k (int): Number of top elements to consider.

    Returns:
    List[str]: A list of top product IDs for the given query.

    Raises:
    AssertionError: If the input types are not as expected.
    """
    # Assert that search_engine is a Pipeline
    assert isinstance(search_engine, Pipeline), "search_engine must be a scikit-learn Pipeline object"

    # Assert that query is a string
    assert isinstance(query, str), "query must be a string"

    try:
        # Get top product IDs for the query
        top_product_ids = search_engine.predict([query], k=k).iloc[0]
        return top_product_ids
    except Exception as e:
        logging.error("Error getting top product IDs for query '%s': %s", query, e)
        raise


def demo(k: int = 10):
    try:
        query_df, product_df, label_df = load_data("WANDS/dataset")
        search_engine_bm25 = initialize_search_engine(product_df, method='bm25')
        search_engine_tfidf = initialize_search_engine(product_df, method='tfidf')

        # Define the test query
        armchair_query = "armchair"

        # Obtain top product IDs using BM25
        top_product_ids_bm25 = get_top_product_ids_for_query(search_engine_bm25, armchair_query, k=k)
        print(f"Top products for '{armchair_query}' using BM25:")
        for product_id in top_product_ids_bm25:
            product = product_df.loc[product_df['product_id'] == product_id]
            print(product_id, product['product_name'].values[0])

        # Obtain top product IDs using TF-IDF
        top_product_ids_tfidf = get_top_product_ids_for_query(search_engine_tfidf, armchair_query, k=k)
        print(f"\nTop products for '{armchair_query}' using TF-IDF:")
        for product_id in top_product_ids_tfidf:
            product = product_df.loc[product_df['product_id'] == product_id]
            print(product_id, product['product_name'].values[0])

        print("\nEvaluating BM25 Search Engine")
        query_df_bm25 = calculate_map_k(query_df.copy(), search_engine_bm25, label_df, k=k)
        print("MAP@K without partial matches (BM25):", query_df_bm25['map@k_no_partial'].mean())
        print("MAP@K with partial matches (BM25):", query_df_bm25['map@k_with_partial'].mean())

        print("\nEvaluating TF-IDF Search Engine")
        query_df_tfidf = calculate_map_k(query_df.copy(), search_engine_tfidf, label_df, k=k)
        print("MAP@K without partial matches (TF-IDF):", query_df_tfidf['map@k_no_partial'].mean())
        print("MAP@K with partial matches (TF-IDF):", query_df_tfidf['map@k_with_partial'].mean())

    except Exception as e:
        logging.error("Error in demo execution: %s", e)


def main(directory: str = "WANDS/dataset", method: str = 'bm25', query: str = None,
         lemmatize: bool = True, stemming: bool = True, synonyms: bool = False, remove_stop_words: bool = True,
         run_demo: bool = False, k: int = 10):
    """
    Main function to load data, initialize the search engine, and optionally run the demo.

    Parameters:
    directory (str): The directory where the dataset files are located.
    method (str): The search engine method to use. Can be 'bm25' or 'tfidf'.
    run_demo (bool): Whether to run the demo function.
    query (str): The search query to use.
    k (int): Number of top elements to consider.
    """
    try:
        if run_demo:
            demo(k=k)
        elif query:
            # Load data
            query_df, product_df, label_df = load_data(directory)

            # Initialize search engine
            search_engine = initialize_search_engine(product_df, method, lemmatize=lemmatize, stemming=stemming,
                                                     synonyms=synonyms, remove_stop_words=remove_stop_words)

            top_product_ids = get_top_product_ids_for_query(search_engine, query, k=k)
            print(f"Top products for '{query}' using {method.upper()}:")
            for product_id in top_product_ids:
                product = product_df.loc[product_df['product_id'] == product_id]
                print(product_id, product['product_name'].values[0])

    except Exception as e:
        logging.error("Error in main execution: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run search engine and evaluate performance")
    parser.add_argument("--directory", type=str, default="WANDS/dataset",
                        help="Directory where the dataset files are located")
    parser.add_argument("--method", type=str, default='bm25', choices=['bm25', 'tfidf'],
                        help="Search engine method to use")
    parser.add_argument("--query", type=str, help="The search query to use")
    parser.add_argument("--lemmatize_off", action='store_false', default=True, help="Turn lemmatization off")
    parser.add_argument("--stemming_off", action='store_false', default=True, help="Turn stemming off")
    parser.add_argument("--use_synonyms", action='store_true', default=False, help="Turn synonyms on")
    parser.add_argument("--dont_remove_stop_words", action='store_false', default=True, help="Don't remove stop words")
    parser.add_argument("--run_demo", action='store_true', default=False, help="Run the demo function")
    parser.add_argument("--k", type=int, default=10, help="Number of top elements to consider")

    args = parser.parse_args()
    main(args.directory, args.method, args.query, args.lemmatize_off, args.stemming_off, args.use_synonyms,
         args.dont_remove_stop_words, args.run_demo, args.k)
