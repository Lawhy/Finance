# -*- coding: utf-8 -*-
import scrapy
import pandas as pd


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
            'username': response.xpath('//div[@class=\'profile-left-column\']/div[@class=\'profile-bio\']/p[@class=\'profile-username\']/a/text()').extract(),
            'status': response.xpath('//div[@class=\'profile-left-column\']/div[@class=\'profile-bio\']/ul[@class=\'profile-bio-categorizations\']/li/text()').extract(),
        }

