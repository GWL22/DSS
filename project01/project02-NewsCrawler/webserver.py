# -*- coding: utf-8 -*-

from flask import Flask, jsonify
from newsdao import NewsDAO

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/news/search/<keyword>')
def search_news(keyword):
    print type(keyword)
    newsdao = NewsDAO()
    data = newsdao.get_news_by_keyword_in_content(str(keyword))
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
