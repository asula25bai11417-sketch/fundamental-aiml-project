import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('movies_dataset.csv')
credits = pd.read_csv('movies_dataset.csv')

# Merge datasets
movies = movies.merge(credits, left_on='id', right_on='movie_id')

# Select important columns
movies = movies[['id','title','overview']]

# Remove missing values
movies.dropna(inplace=True)

# Use only overview (simple version)
movies['tags'] = movies['overview']

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print("\nRecommended Movies:\n")
    for i in movies_list:
        print(movies.iloc[i[0]].title)

# Take input
movie_name = input("Enter a movie name: ")
recommend(movie_name)
