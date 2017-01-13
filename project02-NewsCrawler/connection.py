# -*- coding: utf-8 -*-

import redis
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

server = '# add your server address'
redis_port = '# add your port for redis'
r = redis.Redis(host=server, port=redis_port)

my_id = '# add your mysql id'
my_pw = '# add your mysql pw'
my_port = '# add your password'
my_schema = '# add your schema name'
connection_string = 'mysql+mysqldb://{}:{}{}:{}/{}'.format(my_id,
                                                           my_pw,
                                                           server,
                                                           my_port,
                                                           my_schema
                                                           )

engine = create_engine(connection_string, pool_recycle=3600, encoding='utf-8')
Session = sessionmaker(bind=engine)

connection_url = 'http://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=105'
