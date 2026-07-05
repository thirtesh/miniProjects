import joblib as jb
import os
import streamlit as st
import warnings
#warnings.filterignore('ignore')

file=os.path.expanduser("~/Desktop/Project/spam_email/Model/model.joblib")
model=jb.load(file)

st.title('SPAM DETECTOR')
email=st.text_area('Enter Email Text')

if st.button('CHECK'):
    a=model.predict([email])
    prob=model.predict_proba([email])
    if a[0]==0:
        st.success(f"It's a ham email({round(prob[0][0]*100, 1)}% not spam)")
    else:
        st.success(f"It's a spam email({round(prob[0][1]*100, 1)}% spam)")
