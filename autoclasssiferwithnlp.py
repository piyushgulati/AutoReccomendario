import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of automotive products with descriptions
data = {
    'title': ['Sedan', 'SUV', 'Truck', 'Sports Car', 'Electric Vehicle'],
    'description': ['A comfortable sedan for everyday use', 
                    'A spacious SUV for family adventures',
                    'A rugged truck for heavy-duty work',
                    'A sleek sports car for high-performance driving',
                    'An eco-friendly electric vehicle for sustainable transportation']
}

df = pd.DataFrame(data)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Top 3 similar products
    product_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[product_indices]

# Example usage
product_title = 'SUV'
print(f"Recommendations for '{product_title}':")
print(get_recommendations(product_title))
