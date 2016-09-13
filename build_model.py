import cPickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_model(filename):
    ### write your code to build a model
    df = pd.read_csv(filename)
    X = df['body']
    y = df['section_name']

    vectorizer = TfidfVectorizer(decode_error = 'ignore')
    train_data = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(train_data, y)

    return vectorizer, model


if __name__ == '__main__':
    vectorizer, model = build_model('data/articles.csv')
    with open('data/vectorizer.pkl', 'w') as f:
        pickle.dump(vectorizer, f)
    with open('data/model.pkl', 'w') as f:
        pickle.dump(model, f)
