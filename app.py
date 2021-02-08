from flask import Flask, request, jsonify, render_template, redirect, send_file, flash
import numpy as np
import pandas as pd
import string
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import time

app = Flask(__name__)


@app.route('/')
def main_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def toxic_comments_classifier():
    start = time.time()
    # Preprocessing steps
    df = pd.read_csv("train.csv")
    df = df.reindex(np.random.permutation(df.index))
    comment = df['comment_text']
    comment = comment.to_numpy()
    label = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    label = label.to_numpy()

    comments = []
    labels = []

    for ix in range(comment.shape[0]):
        if len(comment[ix]) <= 400:
            comments.append(comment[ix])
            labels.append(label[ix])

    punctuation_edit = string.punctuation.replace('\'', '') + "0123456789"
    outtab = "                                         "
    trantab = str.maketrans(punctuation_edit, outtab)

    # create objects for stemmer and lemmatizer
    lemmatiser = WordNetLemmatizer()
    stemmer = PorterStemmer()
    # download words from wordnet library
    nltk.download('wordnet')

    stop_words = get_stop_words('english')
    stop_words.append('')

    for x in range(ord('b'), ord('z') + 1):
        stop_words.append(chr(x))

    for i in range(len(comments)):
        comments[i] = comments[i].lower().translate(trantab)
        l = []
        for word in comments[i].split():
            l.append(stemmer.stem(lemmatiser.lemmatize(word, pos="v")))
        comments[i] = " ".join(l)

    # create object supplying our custom stop words
    count_vector = CountVectorizer(stop_words=stop_words)
    # fitting it to converts comments into bag of words format
    # tf = count_vector.fit_transform(comments).toarray()
    tf = count_vector.fit_transform(comments)

    # load model
    with open('BR-SVC.pkl', 'rb') as f:
        clf = pickle.load(f)

    input_text = request.form['comment']

    new_sample = list(input_text.split(" "))
    ori_text = list(input_text.split(" "))

    for i in range(len(new_sample)):
        new_sample[i] = new_sample[i].lower().translate(trantab)
        l = []
        for word in new_sample[i].split():
            l.append(stemmer.stem(lemmatiser.lemmatize(word, pos="v")))
        new_sample[i] = " ".join(l)

    new_sample_ft = count_vector.transform(new_sample)

    def show_prediction(predict, comments):
        tag = []
        text = []
        censored = []
        j = 0

        for i in range(len(comments)):

            # string = " "
            censored.append([])
            tag.append([])
            if predict[i][0] == 1:
                tag[j].append('toxic')

            if predict[i][1] == 1:
                tag[j].append('severe_toxic')

            if predict[i][2] == 1:
                tag[j].append('obscene')

            if predict[i][3] == 1:
                tag[j].append('threat')

            if predict[i][4] == 1:
                tag[j].append('insult')

            if predict[i][5] == 1:
                tag[j].append('identity_hate')

            if tag[j]:
                string = str("*" * len(ori_text[j]))
                censored[j].append(string)

            if not tag[j]:
                tag[j].append('None')

            text.append(ori_text[i])
            tag[j] = ', '.join(tag[j])
            censored[j] = ' '.join(censored[j])
            j += 1

            df_ex = pd.DataFrame({'Comments': text,
                                  'Predicted tags': tag,
                                  'Censored text': censored})

            df_ex.index += 1

        return df_ex

    predict = clf.predict(new_sample_ft)

    result = show_prediction(predict.toarray(), new_sample)


    censored_sentence = result['Censored text'].values.tolist()
    final = []
    for i in range(len(censored_sentence)):
        if censored_sentence[i].__contains__("*"):

            final.append(censored_sentence[i])
        else:

            final.append(ori_text[i])

    final = ' '.join(final)

    end = time.time()

    time_taken = end - start

    return render_template('index.html', tables=[result.to_html()], comment_here=final, time_here=time_taken)


if __name__ == '__main__':
    app.run()
