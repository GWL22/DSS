# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import requests
from bs4 import BeautifulSoup
from newsdao import NewsDAO

def find_a(tags):
    return tags.name == 'a' and not tags.has_attr('class') and tags.has_attr('href')

class NaverNewsCrawler(object):
    def __init__(self, newsdao, urls):
        self.newsdao = newsdao
        self.urls = urls

    def crawl_link(self):
        for url in self.urls:
            res = requests.get(url)
            content = res.content

            soup = BeautifulSoup(content)

            table = soup.find('table', attrs = {'class' : 'container'})
            for a in table.find_all(find_a):
                link = a['href']
                self.crawl_title_content(link)

    def crawl_title_content(self, link):
        res = requests.get(link)
        content = res.content

        soup = BeautifulSoup(content)

        title = soup.find('h3', attrs = {'id' : 'articleTitle'}).get_text()
        content = soup.find('div', attrs = {'id' : 'articleBodyContents'}).get_text().strip()
        #print link
        #print str(title)
        #print str(content)

        self.newsdao.save_news(link, str(title), str(content))



urls = ['http://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=105']
newsdao = NewsDAO()

crawler = NaverNewsCrawler(newsdao, urls)
crawler.crawl_link()
