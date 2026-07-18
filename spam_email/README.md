# Spam Email Detector

A machine learning project that checks whether the given email is a spam email using on the trained `Logistic Regression` algorithm


## Features

-Data cleaning and preprocessing
-Duplicate data handling
-Feature engineering
-Vectorization of email text
-Model training using `Logistic Regression`
-`Streamlit` web Application
-Checks whether an email is a spam email


## Dataset
```
-v1(email text)
-v2(spam or ham)
```

## Data Preprocessing

The raw dataset was processed through the following pipeline:

- Clean the dataset
- Removed duplicate records
- Selected relevant features
- Encode categorial features
- Vectorized email text



## Machine Learning Pipeline

-Load unprocessed dataset
-Data cleaning
-Feature engineering
-Detection and removal of duplicate data
-Encode categorial features
-Split train and test dataset
-Training `Logistic Regression`
-Evaluation of model using relevant metrics
-Deployment of a `streamlit` application


## Model Used

The final deployed model uses:

- **Logistic Regression**


## Evaluation Metrics

- `Accuracy score`
- `Precision score`
- `Confusion matrix`
![screenshot1](screenshots/conf_matrix.png)


## Technologies Used

- `Python`
- `Pandas`
- `Scikit-learn`
- `Matplotlib`
- `Streamlit`
- `Joblib`


## Installation

```bash
git clone https://github.com/thirtesh/Spam-Email-Detection.git

cd Spam-Email-Detection

pip install -r requirements.txt

streamlit run app.py
```


## Challenges Faced

- Vectorizing the spam email
- Finding the best algorithm for the model
- Inclusion of surity of the prediction


## Screenshots

![screenshot1](screenshots/Capture.PNG)

![screenshot2](screenshots/Capture2.PNG)

![screenshot2](screenshots/Capture2.PNG)

## Author

**Thirtesh P U**