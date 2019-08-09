from requests_html import HTMLSession
session = HTMLSession()
r = session.get('https://www.estimize.com/cohr/fq2-2018?metric_name=eps&chart=historical')
r.html.render()
es = r.html.search('<div class="reported-release-display-information-main-information-column-value ">{es}</div>')['es']
print(es)