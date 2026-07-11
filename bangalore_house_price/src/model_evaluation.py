import sklearn as sk
import pandas as pd
import numpy as np
import joblib as jb
from pathlib import Path
import preprocess

BASE_DIR=Path(__file__).resolve().parent.parent
file=BASE_DIR/'Model'/'model.joblib'
model=jb.load(file)

file=BASE_DIR/'Dataset'/'processed_dataset.csv'
df=pd.read_csv(file)

X=df.drop('price', axis=1)
y=np.log1p(df.price)

X_train, X_test, y_train, y_test= sk.model_selection.train_test_split(X,y,random_state=77, test_size=0.2)

y_pred=model.predict(X_test)


r2=sk.metrics.r2_score(y_test, y_pred)
rmse=sk.metrics.mean_squared_error(y_test, y_pred)


print(f"R2 Score: {r2}")
print(f"RMSE: {rmse}")

print(f"Model Type: {type(model)}")