import pandas as pd
from pathlib import Path

def get_df():
    BASE_DIR=Path(__file__).resolve().parent.parent
    df=pd.DataFrame(pd.read_csv(BASE_DIR/"Dataset"/"Bengaluru_House_Data.csv"))
    return df

def drop_unwanted(df):
    data1=df[['area_type','location','size','total_sqft','bath','balcony','price']].copy()
    data1.dropna(inplace=True)
    data1.drop_duplicates(inplace=True)
    return data1

def clean_data(df):
    df['bhk']=df['size'].apply(lambda x: int(x.split(' ')[0]))
    df.drop('size', axis=1, inplace=True)
    df['total_sqft']=df['total_sqft'].where(df['total_sqft'].str.isdigit())
    df=df.dropna()
    return df

def change_datatype(df):
    df['price']=df['price'].astype(float)
    df['total_sqft']=df['total_sqft'].astype(float)
    df['bhk']=df['bhk'].astype(int)
    df['bath']=df['bath'].astype(int)
    df['balcony']=df['balcony'].astype(int)
    df['price_per_sqft']=df['price']*100000/df['total_sqft']
    return df

def remove_all_outliers(df):
    df=df[(df['total_sqft']/df['bhk'])>=300]
    locations=df.groupby('location')['location'].agg('count').sort_values(ascending=False)
    locations_less10 = locations[locations<10]
    df['location']=df['location'].apply(lambda x: 'Others' if x in locations_less10 else x)

    df_out=pd.DataFrame()
    for _, subdf in df.groupby('location'):
    m = subdf.price_per_sqft.mean()
    st = subdf.price_per_sqft.std()
    reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
    df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    data=pd.DataFrame(df_out)

    return df
