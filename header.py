import streamlit as st
import streamlit.components.v1 as components

def create_header():
  st.title("Recommendation System")
  st.subheader("This recommender system gives movie recommendations based on popular recommendations, content-based filtering, and collaborative filtering")