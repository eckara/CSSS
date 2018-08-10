#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script queries the Pecan Street PostgreSQL database and creates the data file required to run the
"SolarDisagg Individual Home Tutroial" Jupyter notebook.

- 8 homes
- 4 home have solar, 4 do not
- 2 solar proxies
- 10 days of training data
- 5 days of test data
- 2 tuning systems
"""

import sqlalchemy as sq
import pandas as pd
import pytz
from time import time
import datetime as dt

HOME_IDS_SOLAR      = [3367, 7793, 7875, 9932]
HOME_IDS_NO_SOLAR   = [4998, 5949, 7641, 8292]
SOLAR_PROXY_IDS     = [4297, 8084]

def main():
    # Read key for PecanSt, Stored in text file
    print('loading Pecan Street key file...')
    with open('../keys/pecanstkey.txt', 'r') as f:
        key = f.read().strip()
        f.close()

    # Create engine
    print('creating sql engine...')
    engine = sq.create_engine("postgresql+psycopg2://{}@dataport.pecanstreet.org:5434/postgres".format(key))

    # Query Pecan Street database
    print('querying database...')
    ti = time()
    duse = HOME_IDS_SOLAR + HOME_IDS_NO_SOLAR + SOLAR_PROXY_IDS
    query = """
        SELECT dataid, local_15min, use, gen, air1, air2, air3, airwindowunit1, furnace1, furnace2 
        FROM university.electricity_egauge_15min
        WHERE local_15min
        BETWEEN '04-01-2015' AND '04-16-2016' AND electricity_egauge_15min.dataid in (
    """ + ','.join([str(d) for d in duse]) + """);"""
    load_data = pd.read_sql_query(query, engine)
    tf = time()
    deltat = (tf - ti) / 60.
    print('Query took {:.2f} minutes'.format(deltat))

    # Localize the time
    load_data['time'] = load_data.set_index('local_15min').index.tz_localize(pytz.timezone('America/Chicago'), ambiguous=True)
    load_data['date'] = [dt.datetime(d.year, d.month, d.day, 0, 0, 0, 0) for d in load_data['time']]
    load_data.set_index('time', inplace=True)

    return load_data


if __name__ == "__main__":
    k = main()
    print(k.head())