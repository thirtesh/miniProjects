import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent

def get_df():   
    file=BASE_DIR/'dataset'/'spam.csv'
    df=pd.read_csv(file)
    return df

def drop_duplicates(df):
    df=df.drop_duplicates()
    return df

def rename_columns(df):
    df=df.rename(columns={'v1':'category', 'v2':'text'})
    return df

def drop_unwanted(df)
    df.drop(columns={'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'})
    return df

def encode_category(df):
    df['spam']=df['category'].apply(lambda x:1 if x=='spam' else 0)
    return df

