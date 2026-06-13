import joblib as jb
import os
import streamlit as st
import warnings
warnings.filterignore('ignore')

file=os.path.expanduser("~/Desktop/Project/spam_email/Model/model.joblib")
model=jb.load(file)

st.title('SPAM DETECTOR')
email=st.text_area('Enter Email Text')

if st.button('CHECK'):
    a=model.predict([email])
    if a[0]==0:
        st.success("It's a ham email")
    else:
        st.success("It's a spam email")
