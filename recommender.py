import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import gdown
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_data_url = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Pioneers/Recommender%20System/movie_dataset.csv'
movie_data_path = './movie_data.csv'
gdown.download(movie_data_url, movie_data_path, True)
movie_data = pd.read_csv(movie_data_path)
movie_data = movie_data.drop(columns=['homepage', 'id', 'popularity', 'status'])


def find_title_from_index(index):
    return movie_data[movie_data.index == index]["title"].values[0]
def find_index_from_title(title):
    return movie_data[movie_data.title == title]["index"].values[0]


minimum_votes = 150
mean_average_vote = movie_data['vote_average'].mean() 
def weighted_rating(x, m=minimum_votes, C=mean_average_vote):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

movie_data['score'] = movie_data.apply(weighted_rating, axis=1)

features = ['keywords','cast','genres','director'] # change features as desired

def combine_features(row):
    combined_row = ''
    for feature in features:
      combined_row += row[feature]
    return combined_row

for feature in features:
    movie_data[feature] = movie_data[feature].fillna('')
movie_data["combined_features"] = movie_data.apply(combine_features,axis=1)

def all_similarity(data):
  cv = CountVectorizer() 
  count_matrix = cv.fit_transform(data["combined_features"])
  cosine_sim = cosine_similarity(count_matrix)
  return cosine_sim

def k_largest_indices(sim_list, K):
  similar_movies = list(enumerate(sim_list))
  sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
  similar_movies = [x[0] for x in sorted_similar_movies[:K]]
  return similar_movies

def k_most_similar_movies(movie, K):
  similarity_matrix = all_similarity(movie_data)
  movie_index = find_index_from_title(movie)
  similarity_to_movie = similarity_matrix[movie_index] # this is the list of similarity scores for the given movie
  
  similar_movies = k_largest_indices(similarity_to_movie, K)

  movie_names = []
  for index in similar_movies:
    movie_names.append(find_title_from_index(index))
  print(movie_names)
  movie_names = {"movie titles": movie_names}
  return pd.DataFrame(movie_names)


def most_popular_movies(data, num_movies):
  sorted_data = data.sort_values('score', ascending=False)
  return sorted_data[["original_title", "score"]][:num_movies]



    