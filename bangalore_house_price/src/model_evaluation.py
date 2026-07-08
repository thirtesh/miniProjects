import sklearn as sk
import pandas as pd
import joblib as jb
from pathlib import Path
import preprocess

BASE_DIR=Path(__file__).resolve().parent.parent
file=BASE_DIR/'Model'/'model.joblib'
model=jb.load(file)

file=BASE_DIR/'Dataset'/'processed_dataset.csv'
df=pd.read_csv(file)

X=df.drop('price', axis=1)
y=df.price

X_train, X_test, y_train, y_test= sk.model_selection.train_test_split(X,y,random_state=77, test_size=0.2)

y_pred=model.predict(X_test)