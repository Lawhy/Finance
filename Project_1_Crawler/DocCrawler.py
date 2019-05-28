import urllib.request
import re
import os
import time
import random
# banned_charlst_for_naming = '*\/:?"<>|'
http_proxy_lst = ['150.95.190.102', '47.52.222.165', '187.188.137.148', '167.160.80.117', '92.51.163.113', '103.78.213.147']

# open the url and read to retrieve HTML with proxy
def getHtml_Proxy(url, proxy):
    proxy_support = urllib.request.ProxyHandler(proxy)
    opener = urllib.request.build_opener(proxy_support)
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36')]
    urllib.request.install_opener(opener)
    response = urllib.request.urlopen(url)
    html = response.read().decode('utf-8')
    response.close()
    return html

# open the url and read to retrieve HTML without Proxy
def getHtml(url):
    head = {}
    head['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36'
    req = urllib.request.Request(url, headers=head)
    response = urllib.request.urlopen(req)
    html = response.read().decode('utf-8')
    response.close()
    return html

# compile the regular expressions and find the documents
def getUrl(html):
    # reg captures (adress, company_name, date, category)
    reg = r'(?:href=")((?:http://.+?)\.(?:DOC|DOCX|PDF)\?www.cninfo.com.cn)"'
          # r'.+?' \
          # r'title="(.+?)：(.+?)"'
    url_re = re.compile(reg)
    url_lst = url_re.findall(html)
    return url_lst

# def getStockNum(html):
#     reg = r'(?:href=")(?:http://irm.cninfo.com.cn/ssessgs/)(.+?)"'
#     stocknum_re = re.compile(reg)
#     stockNum_lst = stocknum_re.findall(html)
#     return stockNum_lst

def getFile(url):
    # old: urltuple : (adress, company_name, year, month, day, category)
    format = url.split('/')[-1].split('?')[0]
    # company = urltuple[1]
    # date_n_category = urltuple[2]
    # for banned_char in banned_charlst_for_naming:
    #     if banned_char in urltuple[1]:
    #         company = company.replace(str(banned_char), '')
    # file_name = date_n_category + '_' + stock_num + '_' + company + '_' + format
    file_name = format
    block_size = 8192
    # 如果程序中断，不会重复下载已经下载的文件
    if not os.path.exists(file_name):
        offset = random.uniform(0, 0.25)
        time.sleep(offset)
        u = urllib.request.urlopen(url)
        f = open(file_name, 'wb')
        while True:
            buffer = u.read(block_size)
            if not buffer:
                break
            f.write(buffer)
        print("Successfully download" + " " + file_name)
        f.close()
    else:
        print("Arealdy exist" + " " + file_name)


raw_url = 'http://irm.cninfo.com.cn/ircs/interaction/irmInformationList.do?'
pageNo = 'pageNo='
# url suffix containing contents from 2012-01-07 to 2018-02-07
suffix = '&stkcode=&beginDate=2012-01-07&endDate=2018-02-07&keyStr=&irmType=251314'

if not os.path.exists('doc_download_dir'):
    os.mkdir('doc_download_dir')
if not os.path.exists("page_num.txt"):
    f = open("page_num.txt", "w+")
    f.close()
page_num_file = open("page_num.txt", "r+")
page_num_file.seek(0)
page_str = page_num_file.readline()
page_num_file.seek(0)
page = 0
if page_str == '':
    page = 1
    page_num_file.write(str(page))
    page_num_file.seek(0)
else:
    page = int(page_str)
num_of_cur_exceptions = 0
os.chdir(os.path.join(os.getcwd(), 'doc_download_dir'))
while True:
    try:
        if page <= 557:
            page_num_file.seek(0)
            page_num_file.write(str(page))
            page_num_file.seek(0)
            html = getHtml_Proxy(raw_url + pageNo + str(page) + suffix, {'http': http_proxy_lst[(page + random.randint(0, 100)) % len(http_proxy_lst)]})
            if page % 51 == 0:
                html = getHtml(raw_url + pageNo + str(page) + suffix)
            url_lst = getUrl(html)
            print("Current page num is: " + str(page))
            for url in url_lst:
                getFile(url)
            page += 1
            time.sleep(0.1)
            print("Attention: Next Page!")
            time.sleep(0.1)
            # print("Current download amount: " + str(num_of_articles))
            # time.sleep(0.1)
        else:
            break
    except Exception:
        num_of_cur_exceptions += 1
        print("Exception occur: " + str(num_of_cur_exceptions))
        time.sleep(30)
        continue
page_num_file.close()
print("Finished")
print("Total page number: " + str(page-1))


# for Test use
# page = 1
# html = getHtml_Proxy(raw_url + pageNo + str(page) + suffix, {'http': http_proxy_lst[0]})
# url_lst = getUrl(html)
# stocknum_lst = getStockNum(html)
# print(len(stocknum_lst))
# print(len(url_lst))
