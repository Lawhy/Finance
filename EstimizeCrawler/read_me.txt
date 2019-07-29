Environment:
Python 3
Scrapy (See https://scrapy.org/)

1. Open https://www.estimize.com/estimates,  hover mouse over "LOAD NEXT 20", you can see a link at the botton of you browser (Chrome), remember the number at the end, e.g. 2834024

2. Open estimize_crawler.py, edit variable init_id to 2834000 (must end with 00)

3. In the terminal (mac or linux) or powershell (windows), in the folder where estimize_crawler.py is in,  type the command:

scrapy runspider estimize_crawler.py -o results.json

Then the crawler will run automatically, and a raw json result will be stored in results.json

4. Then run the following command to covert the results.json into csv format:

python formater.py 

Or if you have both python 2 and 3 in your pc, use this:

python3 formater.py 

5. Check your result in results.csv
