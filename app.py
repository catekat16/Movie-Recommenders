import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import tensorflow as tf
from header import *
from recommender import *
from recommender2 import *
from response import *
from utils import * 

create_header()

def get_movie_recs(rec_method, movie, K):
  if rec_method == "Popular Movies":
    return most_popular_movies(movie_data, K)
  elif rec_method == "Content-Based Recommendation":
    if movie is not None and movie is not []:
      return k_most_similar_movies(movie, K)
  else:
    if movie is not None:
      return colb_filter(movie[0], movie[1], K)

choice = st.radio("Choose recommendation method: ", ["Popular Movies", "Content-Based Recommendation", "Collaborative Recommendation"])
if choice == "Content-Based Recommendation":
  left_column, right_column = st.columns(2)
  with left_column:
    k = st.slider("Pick a value for K", 1, 20)
  with right_column:
    select = st.multiselect("Pick a movie!", movie_data["original_title"], default = None)
    if select == []:
      movie = None
    else:
      movie = select[0]
  display_code_content()
elif choice == "Collaborative Recommendation":
  k = st.slider("Pick a value for K", 1, 20)
  left_column, right_column = st.columns(2)
  with left_column:
    select = st.multiselect("Pick seven movies", movies["title"], default = None)
    if select == []:
      movie = None
    else:
      movie = select
  with right_column:
    m1 = st.slider("Pick a rating for Movie 1", 0, 5)
    m2 = st.slider("Pick a rating for Movie 2", 0, 5)
    m3 = st.slider("Pick a rating for Movie 3", 0, 5)
    m4 = st.slider("Pick a rating for Movie 4", 0, 5)
    m5 = st.slider("Pick a rating for Movie 5", 0, 5)
    m6 = st.slider("Pick a rating for Movie 6", 0, 5)
    m7 = st.slider("Pick a rating for Movie 7", 0, 5)
    rate = [m1,m2,m3,m4,m5, m6, m7]
  if movie!= None and len(movie)==len(rate):
    movie = [movie, rate]
  else:
    movie = None
  display_code_collab()
else:
  k = st.slider("Pick how many movies you want to see!", 1, 20)
  movie = None
  display_code_popular()

recs = get_movie_recs(choice, movie, k)
with st.sidebar:
  st.header("Recommendations Output:")
  get_app_response(recs)
  create_footer()
