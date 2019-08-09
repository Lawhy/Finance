# -*- coding: utf-8 -*-
import scrapy
import pandas as pd
import time
import re
from selenium import webdriver


# +
def get_attrs(url):
    pa = r'https://www.estimize.com/(.+?)/fq(.+?)-(.+?)\?metric_name=eps&chart=historical'
    attrs = re.findall(pa, url)
    return {
        'ticker': attrs[0][0],
        'fq': attrs[0][1],
        'year': attrs[0][2]
    }


def gen_link(ticker, fq, year):
    main = 'https://www.estimize.com/'
    return main + ticker + '/' + 'fq' + str(fq) + '-' + str(year) + '?metric_name=eps&chart=historical'


def get_actual_eps(sel):
    for s in sel.css('.reported-release-display-information-main-information-column'):
        label = s.css('.reported-release-display-information-main-information-column-label').xpath('text()').extract()
        label = ''.join(label)
        if label == 'EPS Actual':
            actual_eps = s.css('.reported-release-display-information-main-information-column-value').xpath('text()').extract()
            return(float(actual_eps[0]))


# -

class ActualEpsSpider(scrapy.Spider):

    name = 'actual_eps'
    start_urls = []
    
    # get urls
    # urls_1 = pd.read_excel('urls_1.xlsx', encoding='UTF-8')['url'].tolist()
    urls_2 = pd.read_excel('urls_2.xlsx', encoding='UTF-8')['url'].tolist()
    # start_urls += urls_1
    start_urls += urls_2
    
    
    def __init__(self):
        # init driver
        self.driver = webdriver.Chrome('D://chromedriver/chromedriver.exe')
        # init tickers for generating urls
        tickers_df = pd.read_excel('data/tickers.xlsx')
        self.tickers = tickers_df['ticker'].apply(lambda x: str(x).lower()).tolist()
            
    def parse(self, response):
        self.driver.get(response.url)
        time.sleep(1)
        sel = scrapy.Selector(text=self.driver.page_source)
        eps = get_actual_eps(sel)
        attrs = get_attrs(response.url)
        yield {
            'ticker': attrs['ticker'],
            'year': attrs['year'],
            'fq': attrs['fq'],
            'actual_eps': eps
        }
