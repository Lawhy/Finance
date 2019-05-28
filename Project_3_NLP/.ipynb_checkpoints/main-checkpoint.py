# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from aip import AipNlp

# +
APP_ID = "16323168"
API_KEY = "azMXa7UuqFH8qsXTsumh1XoF"
SECRET_KEY = "CfYjxNl4SMAkorMpXIhAclyIQ3nAzwE9"

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
# -

print("Hello")
