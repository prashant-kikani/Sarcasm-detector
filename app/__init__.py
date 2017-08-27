import os
import flask, flask.views
from flask import Markup
from flask import jsonify
from . import evaluate

app = flask.Flask(__name__)


class Main(flask.views.MethodView):
    def get(self):
        return flask.render_template('index.html')



app.add_url_rule('/', view_func=Main.as_view('main'), methods=["GET"])



@app.route('/_compute')
def compute():
    sentence = flask.request.args.get('sentence')
    percentage = evaluate.tweetscore(str(sentence))
    return jsonify(result=percentage)
