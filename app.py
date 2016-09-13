import pandas as pd
import cPickle as pickle
from build_model import build_model
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
vectorizer, model = build_model('data/articles.csv')
# home page
@app.route('/')
def index():
    return render_template('index.html', title = 'Doc Classifier')

@app.route('/submit', methods = ['GET','POST'])
def submit():
    return render_template('submit.html', title= 'Doc Submission')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    text = str(request.form['input_text'])
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return render_template('predict.html', prediction = prediction, title = 'Results')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
