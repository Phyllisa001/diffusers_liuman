from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names()
    
    return tfidf_matrix, feature_names

# Example usage
documents = [
    "Python is a popular programming language",
    "Machine learning is an important application of Python",
    "Natural Language Processing deals with text data"
]

tfidf_matrix, feature_names = calculate_tfidf(documents)

# Print the TF-IDF values for each word in the documents
for i, doc in enumerate(documents):
    print(f"Document {i+1}:")
    for j, word in enumerate(feature_names):
        tfidf_value = tfidf_matrix[i, j]
        if tfidf_value > 0:
            print(f"   Word: {word}, TF-IDF: {tfidf_value:.4f}")
