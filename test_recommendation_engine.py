import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from recommendation_engine import TextPreprocessor, SearchEngines, PerformanceMetrics, load_data, \
    initialize_search_engine, calculate_map_k


class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.product_data = {
            'product_id': ['p1', 'p2', 'p3'],
            'product_name': ['chair', 'table', 'sofa'],
            'product_description': ['comfortable chair', 'wooden table', 'leather sofa']
        }
        self.query_data = {
            'query_id': [1, 2],
            'query': ['comfortable chair', 'wooden table']
        }
        self.label_data = {
            'query_id': [1, 1, 2, 2],
            'product_id': ['p1', 'p2', 'p2', 'p3'],
            'label': ['Exact', 'Partial', 'Exact', 'Partial']
        }
        self.product_df = pd.DataFrame(self.product_data)
        self.query_df = pd.DataFrame(self.query_data)
        self.label_df = pd.DataFrame(self.label_data)

    def test_text_preprocessor(self):
        preprocessor = TextPreprocessor(lemmatize=True, stemming=True, synonyms=True, remove_stop_words=True)
        processed_text = preprocessor.preprocess_text("This is a test sentence.")
        self.assertIsInstance(processed_text, str)

    def test_bm25_search_engine(self):
        search_engine = SearchEngines.BM25SearchEngine()
        search_engine.fit(self.product_df['product_name'] + ' ' + self.product_df['product_description'],
                          self.product_df['product_id'])
        top_products = search_engine.predict(pd.Series(['comfortable chair']), k=2)
        self.assertIsInstance(top_products, pd.Series)
        self.assertGreater(len(top_products.iloc[0]), 0)

    def test_tfidf_search_engine(self):
        search_engine = SearchEngines.TFIDFSearchEngine()
        search_engine.fit(self.product_df['product_name'] + ' ' + self.product_df['product_description'],
                          self.product_df['product_id'])
        top_products = search_engine.predict(pd.Series(['comfortable chair']), k=2)
        self.assertIsInstance(top_products, pd.Series)
        self.assertGreater(len(top_products.iloc[0]), 0)

    def test_map_at_k(self):
        true_ids = np.array(['p1', 'p2'])
        predicted_ids = ['p1', 'p3']
        true_labels = np.array(['Exact', 'Partial'])
        score = PerformanceMetrics.map_at_k(true_ids, predicted_ids, true_labels=true_labels, k=2, partial_matches=True)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_load_data(self):
        query_df, product_df, label_df = load_data("WANDS/dataset")
        self.assertIsInstance(query_df, pd.DataFrame)
        self.assertIsInstance(product_df, pd.DataFrame)
        self.assertIsInstance(label_df, pd.DataFrame)

    def test_initialize_search_engine(self):
        search_engine = initialize_search_engine(self.product_df, method='bm25', lemmatize=True, stemming=True,
                                                 synonyms=False, remove_stop_words=True)
        self.assertIsInstance(search_engine, Pipeline)

    def test_calculate_map_k(self):
        search_engine = initialize_search_engine(self.product_df, method='bm25', lemmatize=True, stemming=True,
                                                 synonyms=False, remove_stop_words=True)
        query_df = calculate_map_k(self.query_df, search_engine, self.label_df, k=2)
        self.assertIn('map@k_no_partial', query_df.columns)
        self.assertIn('map@k_with_partial', query_df.columns)


if __name__ == '__main__':
    unittest.main()
