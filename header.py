import streamlit as st
import streamlit.components.v1 as components
from utils import *
from recommender import *
from recommender2 import *


def create_header():
  st.title("Movie Recommender System 🎬🍿🎟")
  st.write("This recommender system gives movie recommendations based on popular recommendations, content-based filtering, and collaborative filtering")
    #if st.checkbox("Show Code 👀"):
    #st.code(lines_to_display, language='python')
  
  st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def create_footer():
  with st.sidebar:
    
    if st.button("🚀 Open in Colab"):
      #cols = st.betacolumns(3)
      link1 = '[Notebook 1](https://colab.research.google.com/drive/1KyI_idhD9tWxCjHDohxuq1qXwTzoAyoY)'
      st.markdown(link1, unsafe_allow_html=True)
      link2 = '[Notebook 2](https://colab.research.google.com/drive/1f-gb5kMhZvDnmlx5JL9jE4oGcaemXsvT)'
      st.markdown(link2, unsafe_allow_html=True)
      link3 = '[Notebook 3](https://colab.research.google.com/drive/1RS3GmGfFjAdBaf72-Yxxr9Foua-uDCQ6)'
      st.markdown(link3, unsafe_allow_html=True)
    st.write("Notebooks made with ❤️ by Inspirit AI")

def display_code_popular():
  with maybe_echo():
    # Code for Popular Recommendation

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

    def most_popular_movies(data, num_movies):
      sorted_data = data.sort_values('score', ascending=False)
      return sorted_data[["original_title", "score"]][:num_movies]


def display_code_content():
  with maybe_echo():
    # Code for Content-Based Recommendation 

    features = ['keywords','cast','genres','director']

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
      movie_names = {"movie titles": movie_names}
      return pd.DataFrame(movie_names)
    

def display_code_collab():
  with maybe_echo():
    # Code for Collaborative Recommendation 

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
      recs = []
      for rec, val in top_recs:
        if len(recs)>K:
          break
        else:
          if columns[rec] not in rated_movies:
            recs.append(columns[rec])
      if len(recs)==0:
        recs = "We couldn't find any movies with high estimated ratings for you!"
      else:
        recs = {"movie titles": recs}
        recs = pd.DataFrame(recs)
      return recs

        