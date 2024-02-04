## MOVIE RECOMMENDER SYSTEM ##
# Created by: Hermes Ponce
#===================================================================
# Description:
# This is one of the simplest implementations of machine learning,
# analysing data and clustering them into similar groups. This code
# utilises cosine similarity to return 5 movies you should watch if 
# you like a movie already.

#===================================================================

# Here's several helpful packages to load:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

# Running this will list all files under the input directory:
import os
Current_Dir = os.getcwd()
for dirname, _, filenames in os.walk(Current_Dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Loading the data sets onces donwloaded from Kaggle:
genres = pd.read_csv(filenames[0])
movies = pd.read_csv(filenames[1])
movies.info()

# Let us convert the genre IDs into array datatype
# So that we can merge both the datasets into a single dataset easily
import ast
movies['genre_ids'] = movies['genre_ids'].apply(ast.literal_eval)
movies.shape

genres['genre_ids']=genres['id']
genres.drop('id', axis=1, inplace=True)

# We exploded the vector from 'genre_ids'
movies_exploded = movies.explode('genre_ids')

# We merge the movies with genres through the code from 'gengre_ids'
merged_df = pd.merge(movies_exploded, genres, on='genre_ids')
merged_df.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1, inplace=True)

# Processing the datasets to finally obtain a final dataset to begin our work
grouped_df = merged_df.groupby('original_title').agg({
    'genre_ids': list,
    'name': list
}).reset_index()

final = pd.merge(merged_df[['id', 'original_title', 'overview']], grouped_df, on='original_title')
final['genres'] = final['name']
final = final[['original_title', 'overview', 'genre_ids', 'genres']]
final = final.drop_duplicates(subset=['original_title'])

final['genres'] = final['genres'].astype(str)
final['keywords'] = final['overview'] + ' ' + final['genres']
final.drop(['genres', 'overview', 'genre_ids'], axis=1, inplace=True)
final = final.reset_index(drop=True)

# We are beginning text processing to feed into our cosine similarity object
final['keywords'] = final['keywords'].astype(str)
final['keywords'] = final['keywords'].apply(lambda x:x.lower())

# We create the object which convert text in vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

# Transformed vectors (ie 5000 words for each movie)
vectors = cv.fit_transform(final['keywords']).toarray()
vectors.shape 

# We load the algorithm for suffix striping based on Porter stemming algorithm
import nltk
from nltk.stem.porter import *
ps = PorterStemmer()

# We create a function which returns string after stemming
def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

final['keywords'] = final['keywords'].apply(stem)

# After processing the data, let's import the cosine similarity model
from sklearn.metrics.pairwise import cosine_similarity
# Calculates the distance between each vector with another vector
similarity = cosine_similarity(vectors) 

# How does it work?
# If the vectors have the same direction, their angle is 0º, therefore their cosine value is 1
# On the contray, if their angle is 90º, their cosine value is 0.
# Cosine similarity algorithms infers data points as these vectors.
# So, similar data points are clustered together by detecting the higher cosine similarity values
# For all diagonal elements means movie with respect to the same movie thus angle 0 thus cos(0)=1

# Creates a list of tuples mentioning distance with respect to index as well, only first 10 displayed here.
list(enumerate(similarity[0]))[0:10]

# Sort in reverse order with respect to 1st index, only first 10 displayed here.
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[0:10] 

# We create the recommender function
def recommend(movie):
    # We find the index for specific movie
  movie_index = final[final['original_title'] == movie].index[0]
  # Gives a list of all distances for a movie
  distances = similarity[movie_index] 
  # Since, we want the top 5 recommendations (from 1 to 6, because 0 correponds to the same movie)
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6] 
  # Movies_list will return tuple with index of movie and the corresponding distance
  for i in movies_list:
    print(final.iloc[i[0]].original_title)
    
########### FINAL INPUT ###########    
recommend("Schindler's List")    
recommend("American History X")    













