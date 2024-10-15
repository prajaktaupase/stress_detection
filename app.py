from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string

app = Flask(__name__)

# Load the model
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/stress.csv")

# Text preprocessing function
def clean(text, stopword, stemmer):  # Pass stopword and stemmer as arguments
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Clean the text data
stopword = set(stopwords.words('english'))  # Define stopword here
stemmer = nltk.SnowballStemmer("english")  # Define stemmer here
data["text"] = data["text"].apply(lambda x: clean(x, stopword, stemmer))  # Pass stopword and stemmer to clean function

# Convert label to categorical
data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]

# Prepare data for training
x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
model = BernoulliNB()
model.fit(xtrain, ytrain)

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)[0]
        return render_template('result.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
