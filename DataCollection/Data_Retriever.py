import requests
import pandas as pd
import os
#os.getcwd()
import json
import requests, zipfile
import io

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

with open('../../Data/Cookies/throne_cookies.txt', 'r') as f:
    cookies_data = f.read()

cookies_data = json.loads(cookies_data)

with open('../../Data/Auth/github.txt', 'r') as f:
    github_auth = f.read()

github_auth = json.loads(github_auth)

# remove data if exists
try:
    if os.path.isfile('english_premier_league_historical_data.csv'):
        os.remove('english_premier_league_historical_data.csv')
except:
    pass

try:
    if os.path.isfile('english_premier_league_competition_data.csv'):
        os.remove('english_premier_league_competition_data.csv')
except:
    pass

# get new historical data and extract
url = 'https://www.throne.ai/competition/58652da24825085def5a5baa/get_historical_data'

r = requests.get(url, stream=True, headers = headers, auth=(github_auth['user'],github_auth['pass']), cookies = cookies_data)
content = r.content

z = zipfile.ZipFile(io.BytesIO(content))
z.extractall()

# get new competition data and extract

url = 'https://www.throne.ai/competition/58652da24825085def5a5baa/get_upcoming_fixtures'

r = requests.get(url, stream=True, headers = headers, auth=(github_auth['user'],github_auth['pass']), cookies = cookies_data)
content = r.content

z = zipfile.ZipFile(io.BytesIO(content))
z.extractall()
