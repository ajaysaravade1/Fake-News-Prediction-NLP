import joblib
from flask import Flask,render_template,request,flash
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import pickle

app = Flask(__name__)
 
model = joblib.load('filename.pkl')


@app.route("/")
def index():

    return render_template("index.html", prediction_text='')

@app.route('/predict',methods=['POST'])
def predict():
    input = str(request.form['input_news'])
    transformer = TfidfTransformer()
    with open('tfidf_v.pkl', 'rb') as f:
        voc = pickle.load(f)
    loaded_vec_bow = CountVectorizer(decode_error="replace",vocabulary=voc)
    tfidf_vectorizer = transformer.fit_transform(loaded_vec_bow.fit_transform(np.array([input])))


    # vector = tfidf_v.transform([input])
    # arr = vector.toarray()

    result = model.predict(tfidf_vectorizer)
    if result == 1:
        #fake
        return render_template('index.html', prediction_text='This news is fake')
    else:
        return render_template('index.html', prediction_text='This news is True.')

if __name__ == '__main__':
    app.debug = True
    app.run()