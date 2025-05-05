from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


finished_data = pd.read_json('C:/Users/kimwa/Documents/GitHub/Applied-ML-35/project_name/data/finished_data.json', lines=True)

X = finished_data['text'].tolist()
y = finished_data['Genre'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print(finished_data.head())

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

mlb = MultiLabelBinarizer()
y_train_binary = mlb.fit_transform(y_train)
y_test_binary = mlb.transform(y_test)

classifier = OneVsRestClassifier(LinearSVC())
classifier.fit(X_train_vec, y_train_binary)

y_pred = classifier.predict(X_test_vec)
print(classification_report(y_test_binary, y_pred, target_names=mlb.classes_))
