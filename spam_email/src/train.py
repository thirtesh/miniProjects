import sklearn as sk
import pandas as pd
import preprocess as pre
from pathlib import Path

df=pre.get_df()
df=pre.drop_duplicates(df)
df=pre.rename_columns(df)
df=pre.drop_unwanted(df)
df=pre.encode_category(df)

BASE_DIR=Path(__file__).resolve().parent.parent
file=BASE_DIR/'dataset'/'processed spam.csv'
df.to_csv(file)

X=df['text']
y=df['spam']

X_train, X_test, y_train, y_test=sk.model_selection.train_test_split(X,y, test_size=0.2, random_state=77)

pipe=sk.pipeline.Pipeline([('vectorizer',sk.feature_extraction.text.CountVectorizer()),
                            ('logReg',sk.linear_model.LogisticRegression())])

pipe.fit(X_train, y_train)

pre.save_model(pipe)