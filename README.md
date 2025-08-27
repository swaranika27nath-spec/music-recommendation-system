# music-recommendation-system
Collaborative filtering-based recommender system predicting user music preferences using historical listening patterns.
# music_recommender.py
# Collaborative filtering-based music recommendation system using scikit-learn

import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Example: Load user-song interaction data from CSV (should have user_id, song_id, play_count columns)
# Replace 'user_song_data.csv' with your actual file
data = pd.read_csv('user_song_data.csv')

# Pivot to create user-song matrix
user_song_matrix = data.pivot_table(index='user_id', columns='song_id', values='play_count', fill_value=0)

# Fit NearestNeighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_song_matrix)

# Recommend songs for a sample user (change user_index as needed)
user_index = 0  # For demonstration, using the first user
distances, indices = model.kneighbors([user_song_matrix.iloc[user_index].values], n_neighbors=6)

print(f"Recommendations for user {user_song_matrix.index[user_index]}:")
for idx in indices[0][1:]:  # [1:] skips the user itself
    print(f"Recommended user: {user_song_matrix.index[idx]} (similar listening pattern)")
