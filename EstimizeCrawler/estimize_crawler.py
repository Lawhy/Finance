import scrapy


class EstimizeCrawler(scrapy.Spider):
    name = 'estimates'

    # crawls data from init_id down to 0, init_id should be ending with 00
    init_id = 2789100
    record_per_page = 20
    link = 'https://www.estimize.com/estimates?starting_estimate_id='

    start_urls = []
    for i in range(1, (init_id//record_per_page)+1):
        start_urls.append(link + str(i*20))

    def parse(self, response):
        i = 0
        for row in response.xpath('//table[@id=\'estimates\']/tbody/tr'):
            yield {
                'flagged': 'true' if row.xpath('//td[@class=\'flag\']')[i].xpath('./div[@class=\'flag-container\']').get() else 'false',
                'username': row.xpath('//span[@class=\'username user-tooltip-item\']/@data-user')[i].get(),
                'ticker': row.xpath('//span[@class=\'ticker\']/text()')[i].get(),
                'time': row.xpath('//td[@class=\'release\']/a/span[@class=\'name\']/text()')[i].get(),
                'market-time': row.xpath('//td[@class=\'release\']/a/span[@class=\'market-time\']/abbr/text()')[i].get(),
                'published-at': row.xpath('//td[@class=\'published-at\']/a/time/@datetime')[i].get(),
                'user-eps': row.xpath('//tr[@class=\'eps\']/td[@class=\'data user\']/text()')[i].get(),
                'ws-eps': row.xpath('//tr[@class=\'eps\']/td[@class=\'data ws\']/text()')[i].get(),
                'estimize-eps': row.xpath('//tr[@class=\'eps\']/td[@class=\'data estimize\']/text()')[i].get(),
                'user-rev': row.xpath('//tr[@class=\'revenue\']/td[@class=\'data user\']/text()')[i].get(),
                'ws-rev': row.xpath('//tr[@class=\'revenue\']/td[@class=\'data ws\']/text()')[i].get(),
                'estimize-rev': row.xpath('//tr[@class=\'revenue\']/td[@class=\'data estimize\']/text()')[i].get()
            }
            i += 1
