from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

result = {}


def sentiment(sentence):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(sentence)
    return score


@app.route("/", methods=["GET", "POST"])
def sentimentRequest():
    if request.method == "POST":
        sentence = request.form['q']
        sent = sentiment(sentence)
    else:
        sentence = request.args.get('q')
        sent = sentiment(sentence)
    # print(sentence)
    result['neg'] = sent['neg']
    result['pos'] = sent['pos']
    result['neu'] = sent['neu']
    result['compound'] = sent['compound']

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
