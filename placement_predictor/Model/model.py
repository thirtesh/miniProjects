import joblib as jb
import os
import streamlit as st

file=os.path.expanduser('C:/Users/Thirtesh/Desktop/Project/placement_predictor/Model/model.joblib')
model=jb.load(file)

st.title('Are you likely to get a placement offer?')

cgpa=st.number_input('Enter your CGPA:', max_value=10.0, step=0.05)
p=st.number_input('Enter the number of major projects:', min_value=0)
wc=st.number_input('Enter the number of workshops/certifications:', min_value=0)
mp=st.number_input('Enter the number of mini Projects:', min_value=0)
skills=st.number_input('Enter the number of skills:', min_value=0)
csr=st.number_input('Enter your communication skill rating(out of 5):',min_value=0.0 , max_value=5.0, step=0.1)
i=st.text_input('Do you have an internship:').lower()
h=st.text_input('Have you participated in any hackatons:').lower()
p12=st.number_input('Enter your 12th percentage:', max_value=100, step=1)
p10=st.number_input('Enter your 10th percentage:', max_value=100, step=1)
b=st.number_input('Enter the number of backlogs:', min_value=0, step=1)

i=1 if i=='yes' else 0
h=1 if h=='yes' else 0

if st.button('Check'):
    pred=model.predict([(cgpa, int(p), int(wc), int(mp), int(skills), csr, i, h, int(p12), int(p10), int(b))])
    st.success("Your records are great! You're more likely to be placed" if pred==1 else "You're not competent enough! You could have done better :(")