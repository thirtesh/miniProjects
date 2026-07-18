import pandas as pd
from pathlib import Path
import joblib as jb

BASE_DIR=Path(__file__).resolve().parent.parent

def get_df():   
    file=BASE_DIR/'dataset'/'spam.csv'
    df=pd.read_csv(file, encoding='ISO-8859-1')
    return df

def drop_duplicates(df):
    df=df.drop_duplicates()
    return df

def rename_columns(df):
    df=df.rename(columns={'v1':'category', 'v2':'text'})
    return df

def drop_unwanted(df):
    df.drop(columns={'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'})
    return df

def encode_category(df):
    df['spam']=df['category'].apply(lambda x:1 if x=='spam' else 0)
    return df

def save_model(pipe):
    file=BASE_DIR/'model'/'model.joblib'
    jb.dump(pipe, file)