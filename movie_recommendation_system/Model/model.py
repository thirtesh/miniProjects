import joblib as jb
import os
import sklearn as sk
import pandas as pd
import streamlit as st

tfidf_matrix=jb.load(os.path.expanduser('~/Desktop/Project/movie_recommendation_system/Model/tfidf_matrix.joblib'))

cos_sim=sk.metrics.pairwise.cosine_similarity(tfidf_matrix, tfidf_matrix)

file='~/Desktop/Project/movie_recommendation_system/Dataset/movies.csv'
df=pd.DataFrame(pd.read_csv(file))

st.title('Movie Recommendation System')
movie=st.selectbox('Select one of your favourite movie', sorted(df['title']))
number=st.number_input('Enter the number of movies you want', min_value=1, step=1)

if st.button('Find similar movies'):
    movie_idx=df[df['title']==movie].index[0]
    sim_movies=list(enumerate(cos_sim[movie_idx]))
    if number>len(sim_movies):
        number=len(sim_movies)
    sim_movies=sorted(sim_movies, key=lambda x:x[1], reverse=True)
    top_movies=sim_movies[0:number]
    movies_indices=[i[0] for i in top_movies]
    st.success(df['title'].iloc[movies_indices].tolist())