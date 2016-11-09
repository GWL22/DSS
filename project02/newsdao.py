# -*- coding: utf-8 -*-

import datetime

from sqlalchemy import PrimaryKeyConstraint
from model import News
from connection import connection_string, Session, engine

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class NewsDAO(object):
    def __init__(self):
        pass

    def save_news(self, news_id, title, content):
        session = Session()
        if not self.get_news_by_id(news_id):
            print news_id
            news = News(link=news_id,
                        title=title,
                        content=content,
                        crawl_time=datetime.datetime.now()
                        )
            session.add(news)
            session.commit()
        session.close()

    def get_news_by_id(self, news_id):
        try:
            session = Session()
            row = session.query(News) \
                         .filter(News.link == news_id) \
                         .first()
            print row
            return row
        except Exception as e:
            print e
        finally:
            session.close()

    def get_news_by_keyword_in_title(self, keyword):
        pass

    def get_news_by_keyword_in_content(self, keyword):
        data = []
        session = Session()
        result = session.query(News) \
                        .filter(News.content.like('%' + keyword + '%')) \
                        .all()
        for row in result:
            news = {}
            news['link'] = row.link
            news['title'] = row.title
            news['content'] = row.content

            data.append(news)
        return data
