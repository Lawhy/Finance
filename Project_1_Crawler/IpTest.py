from urllib import request

def ipTest(url):
    proxy = {'http': '143.0.188.39'}
    proxy_support = request.ProxyHandler(proxy)
    opener = request.build_opener(proxy_support)
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36')]
    request.install_opener(opener)
    response = request.urlopen(url)
    html = response.read()
    response.close()
    print(html.decode('UTF-8'))

ipTest('http://www.whatismyip.com.tw/')

# 如果观察到如下特征便是可用
# <!DOCTYPE HTML>
# ...
# <h1>IP位址</h1> <span data-ip='150.95.190.102'><b style='font-size: 1.5em;'>150.95.190.102</b></span> <span data-ip-country='JP'><i>JP</i></span>
