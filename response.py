import streamlit as st
import streamlit.components.v1 as components

def get_app_response(recommended_movies):
  if recommended_movies is not None:
    st.write(recommended_movies)