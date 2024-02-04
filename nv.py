# Import library
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Data contoh
texts = ["Ini adalah contoh kalimat positif", "Kalimat ini juga positif", "Kalimat ini negatif", "Kalimat ini campuran"]

labels = ["positif", "positif", "negatif", "campuran"]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

naive_bayes_model = MultinomialNB()

naive_bayes_model.fit(X_train_vectorized, y_train)


predictions = naive_bayes_model.predict(X_test_vectorized)


accuracy = accuracy_score(y_test, predictions)
print(f'Akurasi: {accuracy:.2f}')
print(classification_report(y_test, predictions))
