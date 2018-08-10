
# coding: utf-8

# In[1]:

import requests
from bs4 import BeautifulSoup as bs

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import pandas as pd


# In[2]:

# prepare dataframe
column_headers = ['G','GS','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA',
'FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']

df_name = pd.DataFrame(columns = ['NAME'])

df = pd.DataFrame(columns=column_headers)
# dfplace = pd.DataFrame(player_dict)
# df.append(dfplace)
# or df.append(player_dict)


#initialize chromedriver
chromedriver = "/usr/local/bin/chromedriver" # path to the chromedriver executable
os.environ["webdriver.chrome.driver"] = chromedriver

driver = webdriver.Chrome(chromedriver)


# In[3]:

# get & format webpage url
url = 'https://www.basketball-reference.com/players/a/'
soup = bs(requests.get(url).text, 'lxml')

# go to webpage
driver.get(url)

# make alphabet tuple
letters = tuple(['b','c','d','g','h','j','m','p','r','s','w'])

letters_number_list = tuple([2,3,4,7,8,10,13,16,18,19,23])

dates = ['1979-80','1980-81','1981-82','1982-83','1983-84','1984-85','1985-86','1986-87',
'1987-88','1988-89','1989-90','1990-91','1991-92','1992-93','1993-94','1994-95','1995-96',
'1996-97','1997-98','1998-99','1999-00','2000-01','2001-02','2002-03','2003-04','2004-05',
'2005-06','2006-07','2007-08','2008-09','2009-10','2010-11','2011-12','2012-13','2013-14',
'2014-15','2015-16','2016-17','2017-18']


# In[4]:

for round in range(11):

    df.to_csv('bball_{}.csv'.format(round))
    df_name.to_csv('names_{}.csv'.format(round))

    # click to next letter
    letter_path = '//*[@id="div_alphabet"]/ul/li[{}]/a'.format(letters_number_list[round])
    letter_button = driver.find_elements_by_xpath(letter_path)

    letter_button[0].click()

    # get xpath of player names
    name_locate = '//*[@id="players"]/tbody/tr/th/a'

    # get xpath of origin date
    date_locate = '//*[@id="per_game"]/tbody/tr/th/a'

    # get list of all players in the current alphabet letter
    name_list = driver.find_elements_by_xpath(name_locate)

    length_name_list = len(name_list)

    # clean up name_list
    # del name_list[0:9]
    # del name_list[-1:-6:-1]

    # click on a player, try range?
    for player in range(length_name_list):

        name_list_2 = driver.find_elements_by_xpath(name_locate)[player]
        # name_locate = '//a[contains(@href, "/players/{}"]'.format(letters[round])
        # name_list = driver.find_elements_by_xpath(name_locate)

        current_name = name_list_2.text
        name_dict = {'NAME':current_name}

        # make name_list again here???????????


        # go to a player's info page
        name_list_2.click()

        # check if player was part of 3-point era
        origin_date = driver.find_elements_by_xpath(date_locate)[0].text.split(' ')[0]
        # get url of a player's info page
        player_specific_url = driver.current_url

        soup_player = bs(requests.get(player_specific_url).text, 'lxml')
        player_text = soup_player.find('tbody')

        # get list of rows of season stats
        season_list = player_text.find_all('tr')
        n_seasons = len(season_list)


        # if played less than 3 seasons, just skip this player during data colection
        if (n_seasons > 2) and (origin_date in dates):
            pre_player_0 = season_list[0].find_all('td')
            len_0 = len(pre_player_0)

            pre_player_1 = season_list[1].find_all('td')
            len_1 = len(pre_player_1)

            pre_player_2 = season_list[2].find_all('td')
            len_2 = len(pre_player_2)

            if (len_0 != len_1) or (len_0 != len_2):
                driver.back()

            else:

                df_player = pd.DataFrame(columns=column_headers)

                pre_player_0 = [i.text for i in pre_player_0]
                pre_player_1 = [i.text for i in pre_player_1]
                pre_player_2 = [i.text for i in pre_player_2]

                del pre_player_0[0:4]
                del pre_player_1[0:4]
                del pre_player_2[0:4]

                pre_player_0 = ['0.0' if i == '' else i for i in pre_player_0]
                pre_player_1 = ['0.0' if i == '' else i for i in pre_player_1]
                pre_player_2 = ['0.0' if i == '' else i for i in pre_player_2]

                pre_player_0 = [float(i) for i in pre_player_0]
                pre_player_1 = [float(i) for i in pre_player_1]
                pre_player_2 = [float(i) for i in pre_player_2]

                player_dict_0 = dict(zip(column_headers, pre_player_0))
                player_dict_1 = dict(zip(column_headers, pre_player_1))
                player_dict_2 = dict(zip(column_headers, pre_player_2))

                df_player = df_player.append(player_dict_0, ignore_index=True)
                df_player = df_player.append(player_dict_1, ignore_index=True)
                df_player = df_player.append(player_dict_2, ignore_index=True)

                df_player = df_player.sum()/3
                df = df.append(df_player,ignore_index=True)

                df_name = df_name.append([name_dict])

                # pickle here

                driver.back()

        else:
            driver.back()



# In[5]:




# In[ ]:
