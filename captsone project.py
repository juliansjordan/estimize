# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 07:05:17 2021

@author: full team
"""

#%%Import packages
from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # this is a version that allows us to write the model 
import statsmodels.api as sm
import seaborn as sns 

#%%Import files
file_path = 'C:/Users/jjordan/Documents/Personal/Wharton/FNCE 737/Data/'

consensus = pd.read_csv(file_path+'combined_unadjusted_consensus_new.csv')

consensus.head()

estimates = pd.read_csv(file_path+'combined_unadjusted_estimates_new.csv')

estimates.head()
estimates.describe()
#Columns
#Consensus: ticker, date, estimize stats (weighted avg. high, low, SD, count, for EPS, revenue), actuals
#Estimates (granular): EPS, revenue, ticker, fiscal year, fiscal quarter, reported EPS, reported revenue, user bio, etc.

#%%Clean up data
# Update data types
# Add binary flag to EPS / Revenue
# Diff between reporting date 

consensus.dtypes

#Make a smaller file
consensus_small = consensus.head(1000)
consensus_small['reports_at'] = pd.to_datetime(consensus_small['reports_at'])
consensus_small['reports_at'] = consensus_small[['reports_at']].to_string()

consensus_small.dtypes

#Convert to substring, then cast as date --- got stuck here.
consensus_small['reports_at_substr'] = consensus_small['reports_at'].str[:9]
 


#Cast data types to date
consensus['date'] = pd.to_datetime(consensus['date'])
consensus['reports_at'] = pd.to_datetime(consensus['reports_at'], utc = True)

#Has wall street reached consensus?
#consensus['ws_flag'] = pd.to_datetime(consensus['date'])
consensus['ws_flag'] = consensus['wallstreet.eps'].notnull()
    #For others: the flag is a True, False (not 1,0) --> remember that it's case sensitive
    
#Difference in dates
consensus['datediff'] = pd. consensus['reports_at'] - consensus['date']

#%%Add new data elements


#%%Exploratory data analysis