import re
import nltk
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# Preprocessing setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


class EcommerceSearchEngine:
    def __init__(self, product_df, lemmatize=False, stemming=False, synonyms=False, remove_stop_words=False):
        self.product_df = product_df
        self.bm25 = None
        self.lemmatize = lemmatize
        self.stemming = stemming
        self.synonyms = synonyms
        self.remove_stop_words = remove_stop_words
        if lemmatize:
            from nltk.stem import WordNetLemmatizer
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None
        if stemming:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None
        if remove_stop_words:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = None
        self.preprocess_documents()

    def get_synonyms(self, word):
        from nltk.corpus import wordnet
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    def preprocess_text(self, text):
        if pd.isna(text):
            return ''
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        if self.remove_stop_words:
            words = [word for word in words if word not in self.stop_words]
        if self.lemmatize:
            words = [self.lemmatizer.lemmatize(word) for word in words]
        if self.stemming:
            words = [self.stemmer.stem(word) for word in words]
        if self.synonyms:
            expanded_words = []
            for word in words:
                expanded_words.append(word)
                expanded_words.extend(self.get_synonyms(word))
            words = expanded_words
        return ' '.join(words)

    def preprocess_documents(self):
        combined_text = self.product_df['product_name'] + ' ' + self.product_df['product_description']
        self.product_df['processed_text'] = combined_text.apply(self.preprocess_text)
        self.calculate_bm25()

    def calculate_bm25(self):
        tokenized_corpus = [doc.split(" ") for doc in self.product_df['processed_text']]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_top_product_ids_for_query(self, query, top_n=10):
        tokenized_query = self.preprocess_text(query).split(" ")
        top_product_indices = self.bm25.get_top_n(tokenized_query, self.product_df['product_id'].tolist(), n=top_n)
        return top_product_indices

    @staticmethod
    def map_at_k(true_ids, predicted_ids, true_labels=None, k=10, partial_matches=False):
        """
        Calculate the Mean Average Precision at K (MAP@K) with optional partial match handling.

        Parameters:
        true_ids (list): List of relevant product IDs.
        predicted_ids (list): List of predicted product IDs.
        true_labels (list): List of labels corresponding to the true_ids ("Exact", "Partial", "Irrelevant"). Required if partial_matches is True.
        k (int): Number of top elements to consider.
        partial_matches (bool): Whether to consider partial matches in the scoring.

        Returns:
        float: MAP@K score.
        """
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


def main():
    query_df = pd.read_csv("WANDS/dataset/query.csv", sep='\t')  # get search queries
    product_df = pd.read_csv("WANDS/dataset/product.csv", sep='\t')  # get products
    label_df = pd.read_csv("WANDS/dataset/label.csv", sep='\t')  # get manually labeled ground truth labels

    search_engine = EcommerceSearchEngine(product_df, lemmatize=True, stemming=True, synonyms=False,
                                          remove_stop_words=True)

    def get_matches_for_query(query_id):
        query_group = label_df.groupby('query_id').get_group(query_id)
        relevant_ids = query_group['product_id'].values
        relevant_labels = query_group['label'].values
        return relevant_ids, relevant_labels

    query_df['top_product_ids'] = query_df['query'].apply(search_engine.get_top_product_ids_for_query)
    query_df[['relevant_ids', 'relevant_labels']] = query_df['query_id'].apply(
        lambda qid: pd.Series(get_matches_for_query(qid)))

    # Calculate MAP@K without considering partial matches
    query_df['map@k_no_partial'] = query_df.apply(
        lambda x: EcommerceSearchEngine.map_at_k(x['relevant_ids'], x['top_product_ids'], k=10), axis=1)

    # Calculate MAP@K considering partial matches
    query_df['map@k_with_partial'] = query_df.apply(
        lambda x: EcommerceSearchEngine.map_at_k(x['relevant_ids'], x['top_product_ids'], x['relevant_labels'], k=10,
                                                 partial_matches=True), axis=1)

    print("MAP@K without partial matches:", query_df.loc[:, 'map@k_no_partial'].mean())
    print("MAP@K with partial matches:", query_df.loc[:, 'map@k_with_partial'].mean())


if __name__ == "__main__":
    main()
