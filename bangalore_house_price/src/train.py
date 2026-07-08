import pandas as pd
from pathlib import Path
import preprocess
import sklearn as sk
import numpy as np
import xgboost as xgb
import joblib as jb
import os


df=preprocess.get_df()

df=preprocess.drop_unwanted(df)
df=preprocess.clean_data(df)
df=preprocess.change_datatype(df)
df=preprocess.remove_all_outliers(df)
data=preprocess.ohe(df)

df=data[0]
ohe=data[1]

print(df.head())

BASE_DIR=Path(__file__).resolve().parent.parent
file=BASE_DIR/'Dataset'/'processed_dataset.csv'
df.to_csv(file, index=False)

X=df.drop('price', axis=1)
y=df.price

X_train, X_test, y_train, y_test= sk.model_selection.train_test_split(X,y,random_state=77, test_size=0.2)
y_train=np.log1p(y_train)
y_test=np.log1p(y_test)

pipe = sk.pipeline.make_pipeline(sk.preprocessing.StandardScaler(), xgb.XGBRegressor())

param={ 'xgbregressor__learning_rate': [0.3],
        'xgbregressor__max_depth':[6],
        'xgbregressor__n_estimators':[150],
        'xgbregressor__colsample_bytree':[0.3]
}#If you're wondering why there's only one value per paramter, the hyper parameters are tuned and are set to the best possible value to reduce runtime
grid= sk.model_selection.RandomizedSearchCV(pipe, param_distributions=param, cv=5, random_state=77, scoring='r2', n_iter=1, n_jobs=-1) 

grid.fit(X_train, y_train)


fd=os.path.expanduser(BASE_DIR/'Model'/'model.joblib')
fd1=os.path.expanduser(BASE_DIR/'Model'/'OHE.joblib')

jb.dump(grid, fd)
jb.dump(ohe,fd1)
