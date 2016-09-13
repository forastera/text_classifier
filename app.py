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


'''
Try it:
Create a new .html file in the templates folder. Feel free to copy a fancy template
from the internet or just add some really simple html. Note this is a super
simple app without external stylesheets so add any CSS in the head or inline.
(if that makes no sense, then don't worry about it-- keep it simple!)

Edit the function below to add your page to the app.
'''

@app.route('/your-url-address-here')
def new_page():
    # if you copy one of the existing html files in the folder, you'll also need to add a title argument below
    return render_template('your-html-filename.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
