from importlib import reload
import datetime as dt
import numpy as np
from numpy.linalg import svd, pinv
from scipy.stats import ttest_ind
import sqlalchemy as sq
import pandas as pd
import json
from time import time
from os import path
import os
from pathlib import Path
import pickle as pk
import matplotlib.pyplot as plt
from csss.SolarDisagg import createTempInput
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import Custom_Functions.error_functions as ef
from IPython import get_ipython

def load_setup():

    START_DATE = '01-01-2015'
    END_DATE   = '01-01-2017'
    fp = 'data/netloadsolaridentify_{}_{}.csv'.format(START_DATE, END_DATE)
    fw = 'data/weather_netloadsolaridentify_{}_{}.csv'.format(START_DATE, END_DATE)
    print(fp)
    
    # In[6]:
    
    
    with open('keys/pecanstkey.txt', 'r') as f:
        key = f.read().strip()
        f.close()
    engine = sq.create_engine("postgresql+psycopg2://{}@dataport.pecanstreet.org:5434/postgres".format(key))
    
    
    # In[61]:
    
    
    if not path.exists(fp):
        ti = time()
        # Find sites with complete data for the requested time period and join
        print('determining sites with full data...')
        query = """
            SELECT e.dataid
            FROM university.electricity_egauge_15min e
            WHERE local_15min
            BETWEEN '{}' AND '{}'
            AND e.dataid IN (
                SELECT m.dataid
                FROM university.metadata m
                WHERE m.city = 'Austin'
            )
    
            GROUP BY dataid
            HAVING count(e.use) = (
                SELECT MAX(A.CNT)
                FROM (
                    SELECT dataid, COUNT(use) as CNT
                    FROM university.electricity_egauge_15min
                    WHERE local_15min
                    BETWEEN '{}' AND '{}'
                    GROUP BY dataid
                ) AS A
            );
        """.format(START_DATE, END_DATE, START_DATE, END_DATE)
        metadata = pd.read_sql_query(query, engine)
        duse = metadata.values.squeeze()
        print('querying load and generation data...')
        query = """
            SELECT dataid, local_15min, use, gen 
            FROM university.electricity_egauge_15min
            WHERE local_15min
            BETWEEN '{}' AND '{}'
            AND electricity_egauge_15min.dataid in (
        """.format(START_DATE, END_DATE) + ','.join([str(d) for d in duse]) + """)
            ORDER BY local_15min;
        """
        load_data = pd.read_sql_query(query, engine)
        tf = time()
        deltat = (tf - ti) / 60.
        print('query of {} values took {:.2f} minutes'.format(load_data.size, deltat))
        load_data.to_csv(fp)
        
        # Weather data
        print('querying ambient temperature data from weather table...')
        locs = pd.read_sql_query(
            """
            SELECT distinct(latitude,longitude), latitude
            FROM university.weather
            ORDER BY latitude
            LIMIT 10;
            """,
            engine
        )
        locs['location'] = ['Austin', 'San Diego', 'Boulder']           # Ascending order by latitude
        locs.set_index('location', inplace=True)
        weather = pd.read_sql_query(
            """
            SELECT localhour, temperature
            FROM university.weather
            WHERE localhour
            BETWEEN '{}' and '{}'
            AND latitude = {}
            ORDER BY localhour;
            """.format(START_DATE, END_DATE, locs.loc['Austin']['latitude']),
            engine
        )
        weather.rename(columns={'localhour': 'time'}, inplace=True) # Rename 
        weather['time'] = weather['time'].map(lambda x: x.replace(tzinfo=None))
        weather['time'] = pd.to_datetime(weather['time'])
        weather.set_index('time', inplace=True)
        weather = weather[~weather.index.duplicated(keep='first')]
        weather = weather.asfreq('15Min').interpolate('linear')         # Upsample from 1hr to 15min to match load data
        weather.to_csv(fw)
    else:
        ti = time()
        load_data = pd.read_csv(fp)
        weather = pd.read_csv(fw)
        tf = time()
        deltat = (tf - ti)
        print('reading {} values from csv took {:.2f} seconds'.format(load_data.size, deltat))
    
    
    #%% Load Setup
    
    load_data.rename(columns={'local_15min': 'time'}, inplace=True)
    load_data['time'] = pd.DatetimeIndex(load_data['time'])
    load_data.set_index('time', inplace=True)
    load_data.fillna(value=0, inplace=True)
    del load_data['Unnamed: 0'] # useless column
    
    # Weather Setup
    weather['time'] = pd.DatetimeIndex(weather['time'])
    weather.set_index('time', inplace=True)
    
    # Redefine START_DATE and END_DATE so that the weather and load_data dataset match in time stamps
    
    START_DATE = max(weather.index[0],load_data.index[0])
    END_DATE = min(weather.index[-1],load_data.index[-1])
    weather = weather[(weather.index >= pd.to_datetime(START_DATE)) & (weather.index <= pd.to_datetime(END_DATE))]
    lst = list(set(weather.index)-set(load_data['use'].index)) # when you interpolate hourly data to 15m resolution it also interpolates in the changing time hours. This code inidividuates those times and then I drop them
    weather = weather.drop(lst)
    load_data = load_data[(load_data.index >= pd.to_datetime(START_DATE)) & (load_data.index <= pd.to_datetime(END_DATE))]
    
    # NetLoad
    load_data['netload'] = load_data['use'] - load_data['gen']
    load_data.head()
    
    # Group data by ids
    load_data = load_data[~load_data['dataid'].isin([484, 871, 9609, 9938, 8282, 2365, 5949])] #I removed those Ids that showed  negative netload at some point but zero generation
    grouped_data = load_data.groupby('dataid')
    ids_all = grouped_data.groups.keys()                                # should be the same as 'duse'
    load_data = pd.concat([grouped_data.get_group(k) for k in ids_all], axis=1, keys=ids_all).swaplevel(axis=1)
    del load_data['dataid'] # no longer need as IDs are in the column header
    
    
    # Print to see it makes sense. 
    load_data['netload'][list(ids_all)[0:2]]
    
    ##%% MatLAb data
    #HOME_IDS_SOLAR      = [3367, 7793, 7875, 9932]
    #HOME_IDS_NO_SOLAR   = [4998, 5949, 7641, 8292]
    #SOLAR_PROXY_IDS     = [4297, 8084]
    ###matlab_data = load_data['netload'][list(HOME_IDS_NO_SOLAR+HOME_IDS_SOLAR )].copy()
    matlab_data = load_data['netload']
    NewColNames = ['id'+str(matlab_data.columns[i]) for i in range(len(matlab_data.columns))]
    matlab_data.rename(columns = dict(zip(matlab_data.columns,NewColNames)),inplace = True)
    #matlab_data.to_csv('data/netload_{}_{}.csv'.format(START_DATE.date(), END_DATE.date()))
    # In[9]: 
    
    sums = grouped_data.sum()
    solar_ids = {
        'solar': list(sums.index[sums['gen'] > 0]),
        'nosolar': list(sums.index[sums['gen'] <= 0])
    }
    n = len(solar_ids['solar'])
    
    print('There are %d homes with complete data' % len(sums))
    print('%d homes solar' % len(solar_ids['solar']))
    print('%d homes with no solar' % len(solar_ids['nosolar']))
    
    return load_data, weather, grouped_data, ids_all, solar_ids