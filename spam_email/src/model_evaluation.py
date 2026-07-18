from pathlib import Path
import joblib as jb
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR=Path(__file__).resolve().parent.parent

file=BASE_DIR/'dataset'/'processed spam.csv'
df=pd.read_csv(file, encoding='ISO-8859-1')
X,y=df['text'], df['spam']

file=BASE_DIR/'model'/'model.joblib'
model=jb.load(file)

X_train, X_test, y_train, y_test=sk.model_selection.train_test_split(X,y, test_size=0.2, random_state=77)

y_pred=model.predict(X_test)

print(sk.metrics.accuracy_score(y_test, y_pred))
print(sk.metrics.precision_score(y_test, y_pred))

conf=sk.metrics.confusion_matrix(y_test, y_pred)
disp=sk.metrics.ConfusionMatrixDisplay(conf)
disp.plot()
plt.show()