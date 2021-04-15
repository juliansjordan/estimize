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

#%%Clean up data and add new data elements
# Update data types
# Add binary flag to EPS / Revenue

consensus['ws_flag'] = consensus['wallstreet.eps'].notnull()

# Diff between reporting date 
consensus['reports_at'] = pd.to_datetime(consensus['reports_at'], utc = True)
consensus['date'] = pd.to_datetime(consensus['date'], utc = True)
consensus['datediff'] = consensus['reports_at'] - consensus['date'] 

consensus['datediff_in_days'] = consensus['datediff'].dt.days 

#Difference between values in estimize vs. wallstreet
#EPS
consensus['eps_diff_abs'] = consensus['estimize.eps.weighted'] - consensus['actual.eps']
consensus['eps_diff_percent'] = (consensus['estimize.eps.weighted'] - consensus['actual.eps'])/consensus['actual.eps']
consensus['eps_diff_percent_abs'] = consensus['eps_diff_percent'].abs

#Revenue
consensus['rev_diff_abs'] = consensus['estimize.revenue.weighted'] - consensus['actual.revenue']
consensus['rev_diff_percent'] = (consensus['estimize.revenue.weighted'] - consensus['actual.revenue'])/consensus['actual.revenue']
consensus['rev_diff_percent_abs'] = consensus['eps_diff_percent'].abs



#%%Exploratory data analysis
#Number of unique tickers
    consensus.groupby(['ticker'])['ticker'].nunique()

#Min date
    consensus['date'].min() ##--> 2010-09-29
    consensus['date'].max() ##--> 2020-12-30

#Get some graphics going
  #Look at ranges in lags in diff between report dates & 
     consensus.boxplot(column=['datediff_in_days']) #--> seems like there is a large range
     plt.boxplot(x=consensus['datediff_in_days']) #--> just trying out another metho dof plotting
  
  #Look at differences in EPS, revenue  
      consensus.boxplot(column=['eps_diff_percent', 'rev_diff_percent']) #--> really large differences... need to figure out why the differences are so high

      consensus.boxplot(column=['eps_diff_abs', 'rev_diff_abs']) #--> really large differences... need to figure out why the differences are so high

    consensus.boxplot(column=['eps_diff_abs']) #--> we have some bad data; really extreme values that are outliers

#%%Analysis
    #Create a conditional column that has the standard error based on whether the ws_flag is true or flase
    #Then we can run a t-test to see if the differences in error values are stastically significant
    #First need to clean up data though - there ar esome large outliers on EPS, Revenue
