import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import gdown
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import * 

m_data_url = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Pioneers/Recommender%20System/movies.csv'
m_data_path = './movies.csv'
gdown.download(m_data_url, m_data_path, True)
movies = pd.read_csv(m_data_path)

ratings_data_url = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Pioneers/Recommender%20System/ratings.csv'
ratings_data_path = './ratings.csv'
gdown.download(ratings_data_url, ratings_data_path, True)
ratings = pd.read_csv(ratings_data_path)

movie_ratings_data = pd.merge(ratings, movies, on='movieId')

def data_fix(res):
  mean = ratings.groupby(['userId'], as_index = False, sort = False).mean().rename(columns={"rating":"rating_mean"})
  movie_ratings_norm = pd.merge(movie_ratings_data[['userId','title', 'movieId', 'rating', 'genres']], mean[['userId', 'rating_mean']], on='userId', how="left", sort="False")
  movie_ratings_norm['ratings_adjusted'] = movie_ratings_norm['rating'] - movie_ratings_norm['rating_mean']

  norm_ratings_table = movie_ratings_norm.pivot_table(index='userId', columns='title', values='ratings_adjusted')
  norm_ratings_table = norm_ratings_table.append(res, ignore_index=True)
  norm_ratings_table=norm_ratings_table.fillna(0)
  return norm_ratings_table


def user_similarity(data_ratings, K):
  all_users = data_ratings.values
  our_ratings = all_users[-1].reshape(1, -1)
  sim = cosine_similarity(our_ratings, all_users[:-1])
  similar_users = list(enumerate(sim[0]))
  sorted_similar_users = sorted(similar_users,key=lambda x:x[1],reverse=True)
  k_similar_users = [(671, 1)] + sorted_similar_users[:10]
  return [x[1] for x in k_similar_users], data_ratings.take([x[0] for x in k_similar_users])

def estimate_ratings(sim_user_ratings):
  user_sim_vals, top10usersdf = sim_user_ratings
  denom = sum(user_sim_vals)
  values = []
  inx = 0
  all_values = top10usersdf.values
  for x in top10usersdf.loc[671]:
    totalsum = 0
    if x==0.0:
      for v in range(1, 11):
        totalsum+=all_values[v-1][inx]*user_sim_vals[v-1]
      top10usersdf.loc[671][inx] = totalsum/denom
    inx+=1
  return top10usersdf


def colb_filter(rated_movies, ratings_for_movies, K):
  if len(ratings_for_movies)==0:
    avg = 0
  else:
    avg = sum(ratings_for_movies)/len(ratings_for_movies)
  ratings_for_movies = [i-avg for i in ratings_for_movies]
  res = dict(zip(rated_movies, ratings_for_movies))

  norm_ratings_table = data_fix(res)
  estimated_ratings = estimate_ratings(user_similarity(norm_ratings_table, K))

  top_recs = []
  for inx, x in enumerate(estimated_ratings.loc[671].values):
    if x > 0.0:
      top_recs.append((inx, x))
  top_recs.sort(key=lambda x:x[1], reverse=True)

  columns = estimated_ratings.columns
  print(columns)
  recs = []
  for rec, val in top_recs:
    if len(recs)>K:
      break
    else:
      if columns[rec] not in rated_movies:
        movie_name = format_movie(columns[rec])
        recs.append(movie_name)
  if len(recs)==0:
    recs = "We couldn't find any movies with high estimated ratings for you!"
  else:
    recs = {"movie titles": recs}
    recs = pd.DataFrame(recs) # sometimes will be the string message case 
  return recs
