# -*- coding: utf-8 -*-
import scrapy
import pandas as pd
import re


# +
def get_user(url):
    pa = r'https://www.estimize.com/users/(.+)'
    return re.findall(pa, url)

get_user('https://www.estimize.com/users/zzztimbo')[0]


# -

class UserInfoSpider(scrapy.Spider):
    
    # ...
    name = 'user_info'
    start_urls = []
    
    # the main link
    link = 'https://www.estimize.com/users/'
    
    # user list
    users = pd.read_excel('D://Finance/EstimizeCrawler/usernameset.xlsx')['username'].tolist()
    
    for user in users:
        start_urls.append(link + str(user))

    def parse(self, response):
        
        yield {
            'username': get_user(response.url),
            'status': response.xpath('//div[@class=\'profile-left-column\']/div[@class=\'profile-bio\']/ul[@class=\'profile-bio-categorizations\']/li/text()').extract(),
        }

