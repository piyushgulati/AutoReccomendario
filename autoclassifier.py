import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Sample dataset of automotive products with user ratings
data = {
    'user_id': [1, 2, 3, 1, 2, 3],
    'product_id': [1, 1, 1, 2, 2, 2],
    'rating': [5, 4, 3, 4, 3, 5],
    'title': ['Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV'],
}

df = pd.DataFrame(data)

# Pivot table for user-item matrix
pivot_table = pd.pivot_table(df, values='rating', index='user_id', columns='title', fill_value=0)

# Convert to sparse matrix for memory efficiency
sparse_matrix = csr_matrix(pivot_table.values)

# Collaborative filtering model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)

# Function to get recommendations
def get_recommendations(user_id, product_title, k=3):
    product_index = pivot_table.columns.get_loc(product_title)
    user_ratings = pivot_table.iloc[user_id - 1].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_ratings, n_neighbors=k+1)
    similar_products = []
    for i in range(1, len(distances.flatten())):
        similar_products.append(pivot_table.columns[indices.flatten()[i]])
    return similar_products

# Example usage
user_id = 1
product_title = 'Sedan'
print(f"Top 3 Recommendations for User {user_id} based on '{product_title}':")
print(get_recommendations(user_id, product_title))
