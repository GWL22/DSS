# -*- coding: utf-8 -*-

import redis
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

server = 'ec2-54-191-110-146.us-west-2.compute.amazonaws.com'
r = redis.Redis(host=server, port=6379)

my_id = 'root'
my_pw = 'windows48'
my_port = '3306'
my_schema = 'navernews'
connection_string = 'mysql+mysqldb://{}:{}{}:{}/{}'.format(my_id,
                                                           my_pw,
                                                           my_port,
                                                           my_schema,
                                                           server
                                                           )

engine = create_engine(connection_string, pool_recycle=3600, encoding='utf-8')
Session = sessionmaker(bind=engine)

connection_url = 'http://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=105'
