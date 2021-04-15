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

#%% Addressing Outliers

consensus = consensus[consensus['datediff_in_days'] < 500] #removes long-term estimates
consensus = consensus[consensus['eps_diff_percent_abs'] < 5] #removes some outliers (what should be the right cutoffs)
consensus = consensus[consensus['rev_diff_percent_abs'] < 5] # same as above


#%% Extract List of Stocks of Interest (had no wall street est at some stage)

cons_tickers = consensus[consensus['ws_flag'] != True]
tickers = cons_tickers.groupby(['ticker'])['ticker'].nunique()
tickers = tickers.index
tickers = tickers.to_series()

#%% Extract rows with these tickers from the original dataset into new dataset

new_cons = consensus[consensus['ticker'].isin(tickers)]

#%% Initial Analysis to see if erros vary between our ws flag

new_cons.groupby(['ws_flag'])['eps_diff_percent_abs'].mean() # shows a significant difference in average absolute error in expected direction
new_cons.groupby(['ws_flag'])['rev_diff_percent_abs'].mean() # same as above

#%% Run regression to test for size and significance of the dummy coefficient

# First set up dummy variable for ws flag that can be interpreted by regression

cons_dummy = pd.get_dummies(new_cons['ws_flag']) # create dummy variables for the wall street flag
new_cons['ws_dummy'] = cons_dummy.iloc[:,1] #add dummy variables to the main dataset

# Set up the regression
eps_ols1 = smf.ols('eps_diff_percent_abs ~ C(ws_dummy) + datediff_in_days', data = new_cons).fit()
eps_ols1.summary() #R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change

eps_ols2 = smf.ols('rev_diff_percent_abs ~ C(ws_dummy) + datediff_in_days', data = new_cons).fit()
eps_ols2.summary() #similar takeaways as above

#%% Next Steps

# The special tickers still includes stocks that definitely had WS coverage but the estimize dataset doesn't include consensus
# We also don't have a clear time-series definied (i.e. starts off false and turns true) for a before and after analysis
# Need to figure out how to modify dataset for the second scenario
# We can then expand this analysis to look at trends in overestimates vs. underestimates
