# -*- coding: utf-8 -*-

import os
import sys
from connection import my_schema
from sqlalchemy.dialects.mysql import INTEGER, BIT, TINYINT, TIME, DOUBLE, TEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, Integer, CHAR, Date, String, Time,
Index, DateTime, TIMESTAMP, func

Base = declarative_base()


class News(Base):
    __tablename__ = '{}'.format(my_schema)

    link            = Column(String(200), primary_key=True, nullable=False)
    title           = Column(String(100), nullable=False)
    content         = Column(TEXT, nullable=False)
    crawl_time      = Column(DateTime, nullable=False)
