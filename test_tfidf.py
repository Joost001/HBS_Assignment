import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing setup for enhanced preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


class EcommerceSearchEngine:
    def __init__(self, product_df):
        self.product_df = product_df
        self.vectorizer = None
        self.tfidf_matrix = None
        self.preprocess_documents()

    def preprocess_text(self, text):
        """
        Preprocess text by removing punctuation, converting to lowercase, removing stopwords, and lemmatizing.
        """
        if pd.isna(text):
            text = ''
        else:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            words = text.split()
            processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            text = ' '.join(processed_words)
        return text

    def preprocess_documents(self):
        """
        Preprocess all documents in the product dataframe.
        """
        combined_text = self.product_df['product_name'] + ' ' + self.product_df['product_description']
        self.product_df['processed_text'] = combined_text.apply(self.preprocess_text)
        self.calculate_tfidf()

    def calculate_tfidf(self, stop_words_=None, ngram_range_=(1, 3), max_df_=0.95, min_df_=1, norm_="l1"):
        """
        Calculate the TF-IDF for combined product name and description.
        """
        vectorizer = TfidfVectorizer(stop_words=stop_words_, ngram_range=ngram_range_, max_df=max_df_,
                                     min_df=min_df_, norm=norm_)
        self.tfidf_matrix = vectorizer.fit_transform(self.product_df['processed_text'].values.astype('U'))
        self.vectorizer = vectorizer

    def get_top_products(self, query, top_n=10):
        """
        Get top N products for a given query based on TF-IDF similarity.
        """
        preprocessed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([preprocessed_query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_product_indices = cosine_similarities.argsort()[-top_n:][::-1]
        return top_product_indices

    def get_top_product_ids_for_query(self, query):
        """
        Get top product IDs for a given query.
        """
        top_product_indices = self.get_top_products(query, top_n=10)
        top_product_ids = self.product_df.iloc[top_product_indices]['product_id'].tolist()
        return top_product_ids

    @staticmethod
    def map_at_k(true_ids, predicted_ids, k=10):
        """
        Calculate the Mean Average Precision at K (MAP@K).
        """
        if not len(true_ids) or not len(predicted_ids):
            return 0.0

        score = 0.0
        num_hits = 0.0

        for i, p_id in enumerate(predicted_ids[:k]):
            if p_id in true_ids and p_id not in predicted_ids[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(true_ids), k)


def main():
    query_df = pd.read_csv("WANDS/dataset/query.csv", sep='\t')  # get search queries
    product_df = pd.read_csv("WANDS/dataset/product.csv", sep='\t')  # get products
    label_df = pd.read_csv("WANDS/dataset/label.csv", sep='\t')  # get manually labeled ground truth labels

    search_engine = EcommerceSearchEngine(product_df)

    def get_exact_matches_for_query(query_id):
        query_group = label_df.groupby('query_id').get_group(query_id)
        exact_matches = query_group.loc[query_group['label'] == 'Exact']['product_id'].values
        return exact_matches

    query_df['top_product_ids'] = query_df['query'].apply(search_engine.get_top_product_ids_for_query)
    query_df['relevant_ids'] = query_df['query_id'].apply(get_exact_matches_for_query)
    query_df['map@k'] = query_df.apply(
        lambda x: EcommerceSearchEngine.map_at_k(x['relevant_ids'], x['top_product_ids'], k=10), axis=1)

    print(query_df.loc[:, 'map@k'].mean())


if __name__ == "__main__":
    main()
