from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text corpus
docs = [
    "I love machine learning",
    "Machine learning is fun",
    "I love deep learning"
]

# Initialize vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text
tfidf_matrix = vectorizer.fit_transform(docs)

# Convert to array for better view
print(tfidf_matrix.toarray())

# Show feature (word) names
print(vectorizer.get_feature_names_out())
