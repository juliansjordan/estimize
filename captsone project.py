# -*- coding: utf-8 -*-

"""
Created on Mon Apr 12 07:05:17 2021
@author: full team
"""

# %%Import packages

from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # this is a version that allows us to write the model 
import statsmodels.api as sm
import seaborn as sns 
import scipy.stats as stats

# %%Import files

file_path = '/Users/laurendavis/Desktop/Wharton/Spring 2021/Data Science for Finance/Final Project/Raw Data/estimize-equities-2020q4/combined_consensus_new.csv'  # 'C:/Users/jjordan/Documents/Personal/Wharton/FNCE 737/Data/'

consensus = pd.read_csv(file_path)  # +'combined_unadjusted_consensus_new.csv')
# Columns
# Consensus: ticker, date, estimize stats (weighted avg. high, low, SD, count, for EPS, revenue), actuals

# %%Clean up data and add new data elements

# Update data types
# Add binary flag to EPS / Revenue

consensus['ws_flag'] = consensus['wallstreet.eps'].notnull()

# Diff between reporting date
consensus['reports_at'] = pd.to_datetime(consensus['reports_at'], utc=True)
consensus['date'] = pd.to_datetime(consensus['date'], utc=True)
consensus['datediff'] = consensus['reports_at'] - consensus['date']

consensus['datediff_in_days'] = consensus['datediff'].dt.days

# Difference between values in estimize vs. actuals
# EPS
consensus['eps_diff'] = consensus['estimize.eps.weighted'] - consensus['actual.eps']
consensus['eps_diff_abs'] = consensus['eps_diff'].abs()
consensus['eps_diff_percent'] = (consensus['estimize.eps.weighted'] - consensus['actual.eps'])/consensus['actual.eps']
consensus['eps_diff_percent_abs'] = consensus[['eps_diff_percent']].abs()

# Revenue
consensus['rev_diff_abs'] = consensus['estimize.revenue.weighted'] - consensus['actual.revenue']
consensus['rev_diff_percent'] = (consensus['estimize.revenue.weighted'] - consensus['actual.revenue'])/consensus['actual.revenue']
consensus['rev_diff_percent_abs'] = consensus[['rev_diff_percent']].abs()

# WS Figures, difference between WS consensus and actuals
# EPS

consensus['eps_diff_ws'] = consensus['wallstreet.eps'] - consensus['actual.eps']
consensus['eps_diff_abs'] = consensus['eps_diff'].abs()
consensus['eps_diff_percent_ws'] = (consensus['wallstreet.eps'] - consensus['actual.eps'])/consensus['actual.eps']
consensus['eps_diff_percent_ws_abs'] = consensus[['eps_diff_percent_ws']].abs()

# Revenue
consensus['rev_diff_ws'] = consensus['wallstreet.revenue'] - consensus['actual.revenue']
consensus['rev_diff_percent_ws'] = (consensus['wallstreet.revenue'] - consensus['actual.revenue'])/consensus['actual.revenue']
consensus['rev_diff_percent_ws_abs'] = consensus[['rev_diff_percent_ws']].abs()


# %% Get key values from consensus

# create cutoff value
cutoff = 180   # required days on either side of the WS flag

# Find first instane a ticker has no wall street estimate
FirstEST = consensus[['ticker', 'date', 'ws_flag']]
FirstEST = FirstEST[FirstEST['ws_flag'] == False]
FirstEST = FirstEST.groupby('ticker')['date'].min()
FirstEST = FirstEST.to_frame()

# Find last instance of estimize consensus with no wallstreet estimate
LastEST = consensus[['ticker', 'date', 'ws_flag']]
LastEST = LastEST[LastEST['ws_flag'] == False]
LastEST = LastEST.groupby('ticker')['date'].max()
LastEST = LastEST.to_frame()

# Get All data with a wall street estimate
FirstWS = consensus[['ticker', 'date', 'ws_flag']]
FirstWS = FirstWS[FirstWS['ws_flag'] == True]

# Find last instane of wall street estimate
LastWS = FirstWS.groupby('ticker')['date'].max()
LastWS = LastWS.to_frame()

# Find first instance of wall street estimate
FirstWS = FirstWS.groupby('ticker')['date'].min()
FirstWS = FirstWS.to_frame()

# Merge all wallstreet info
WSData = FirstWS.merge(LastWS, how='outer', left_on='ticker', right_on='ticker', suffixes=("_Min", "_Max"))
WSData['WS_Duration'] = WSData['date_Max'] - WSData['date_Min']
WSData['WS_Duration'] = WSData['WS_Duration'].astype("timedelta64[D]")


# %% Determine instances of WS Dropping Data

# Merge wall street data with estimize estimate data
WSDrop = LastEST.merge(WSData, how='outer', left_on='ticker', right_on='ticker', suffixes=("_False", "_True"))

# Find instances where Wallstreet dropped coverage
WSDrop = WSDrop[(WSDrop['date']) > WSDrop['date_Max']] 
WSDrop['Pre_Drop_Period'] = WSDrop['date_Max'] - WSDrop['date_Min']
WSDrop['Pre_Drop_Period'] = WSDrop['Pre_Drop_Period'].dt.days
WSDrop['Post_Drop_Period'] = WSDrop['date'] - WSDrop['date_Max']
WSDrop['Post_Drop_Period']= WSDrop['Post_Drop_Period'].dt.days

# Limit to instances where there are 90 days of coverage on either side of drop
WSDrop = WSDrop[WSDrop['Pre_Drop_Period'] > 90]
WSDrop = WSDrop[WSDrop['Post_Drop_Period'] > 90]
    ## FINDING --> Only 29 tickers satisfy this criteria


#%% Create Key Frame for analysis on WS picking up variables

# Get earliest, latest reporting dates from consensus data
estimize_dateRange = consensus.groupby('ticker')['date'].min()
estimize_dateRange = estimize_dateRange.to_frame()
estimize_dateRange['max_date'] = consensus.groupby('ticker')['date'].max()

estimize_dateRange = estimize_dateRange.rename(columns={"date": "est_min_date", "max_date": "est_max_date"})

ticker_analysis_pd = estimize_dateRange.merge(FirstWS, left_on='ticker', right_on='ticker')
ticker_analysis_pd = ticker_analysis_pd.rename(columns={"date": "ws_min_date"})

# Pre-period
ticker_analysis_pd['pre_period_length'] = ticker_analysis_pd['ws_min_date'] - ticker_analysis_pd['est_min_date']
ticker_analysis_pd['pre_period_length'] = ticker_analysis_pd['pre_period_length'].dt.days

# Post-period
ticker_analysis_pd['post_period_length'] = ticker_analysis_pd['est_max_date'] - ticker_analysis_pd['ws_min_date']
ticker_analysis_pd['post_period_length'] = ticker_analysis_pd['post_period_length'].dt.days

# Create 6 mo. pre and post period
tick_analysis_6mo_pd = ticker_analysis_pd.loc[(ticker_analysis_pd['pre_period_length']) > cutoff]
tick_analysis_6mo_pd = tick_analysis_6mo_pd.loc[(tick_analysis_6mo_pd['post_period_length']) > cutoff]

# Create 3 mo. pre and post period
tick_analysis_3mo_pd = ticker_analysis_pd.loc[(ticker_analysis_pd['pre_period_length']) > (cutoff/2)]
tick_analysis_3mo_pd = tick_analysis_3mo_pd.loc[(tick_analysis_3mo_pd['post_period_length']) > (cutoff/2)]

tick_analysis_3mo_pd.groupby('ticker').nunique()


# %% Look into WS picking up coverage

consensus = consensus[consensus['eps_diff_percent_abs'] < 2]  # removes some outliers
consensus = consensus[consensus['rev_diff_percent_abs'] < 2]  # same as above

# Pull consensus data for tickers with 6 month pre & post
consensus_6mo_analysis_pd = consensus.merge(tick_analysis_6mo_pd, left_on = 'ticker', right_on = 'ticker')
consensus_6mo_analysis_pd['days_since_ws_covg'] = consensus_6mo_analysis_pd['date'] - consensus_6mo_analysis_pd['ws_min_date']
consensus_6mo_analysis_pd['days_since_ws_covg'] = consensus_6mo_analysis_pd['days_since_ws_covg'].dt.days
consensus_6mo_analysis_pd['days_since_est_covg'] = consensus_6mo_analysis_pd['date'] - consensus_6mo_analysis_pd['est_min_date']
consensus_6mo_analysis_pd['days_since_est_covg'] = consensus_6mo_analysis_pd['days_since_est_covg'].dt.days

consensus_6mo_analysis_pd_clean = consensus_6mo_analysis_pd.loc[(consensus_6mo_analysis_pd['days_since_ws_covg']) < cutoff]

# Plot as first pass analysis
plt.plot(consensus_6mo_analysis_pd_clean['days_since_ws_covg'], consensus_6mo_analysis_pd_clean['rev_diff_percent'], '.', color='black')
plt.xlim(-cutoff, cutoff)
plt.show()

plt.plot(consensus_6mo_analysis_pd_clean['days_since_ws_covg'], consensus_6mo_analysis_pd_clean['eps_diff_percent'], '.', color='black')
plt.xlim(-cutoff, cutoff)
plt.show()

# Add flag for pre-post
consensus_6mo_analysis_pd_clean['pre_period'] = np.where(consensus_6mo_analysis_pd_clean['days_since_ws_covg']<0, True, False)

# Look at Mean & Standard Deviation of Revenue and EPS for 6 mo (Exhibit 1)
print(consensus_6mo_analysis_pd_clean.groupby(['pre_period'])['rev_diff_percent_abs'].mean())
print(consensus_6mo_analysis_pd_clean.groupby(['pre_period'])['rev_diff_percent_abs'].std())

print(consensus_6mo_analysis_pd_clean.groupby(['pre_period'])['eps_diff_percent_abs'].mean())
print(consensus_6mo_analysis_pd_clean.groupby(['pre_period'])['eps_diff_percent_abs'].std())

consensus_6mo_analysis_pd_clean['num_predictors'] = consensus_6mo_analysis_pd_clean['estimize.eps.count']

# Run regression on 6 mo data (Exhibit 3a)
eps_ols_6mo = smf.ols('eps_diff_percent ~ C(pre_period) + days_since_ws_covg + datediff_in_days', data=consensus_6mo_analysis_pd_clean).fit()
eps_ols_6mo.summary()
    ## FINDING --> it seems that EPS prediciton errors are higher when we are in the pre-period (i.e. before WS Covg)

rev_ols_6mo = smf.ols('rev_diff_percent ~ C(pre_period) + days_since_ws_covg + datediff_in_days', data=consensus_6mo_analysis_pd_clean).fit()
rev_ols_6mo.summary()  # R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change
    ## FINDING --> it seems that Rev prediciton errors are higher when we are in the pre-period (i.e. before WS Covg)


# Pull consensus data for tickers with 3 month pre &  to test on a slightly smaller sample
consensus_3mo_analysis_pd = consensus.merge(tick_analysis_3mo_pd, left_on='ticker', right_on='ticker')
consensus_3mo_analysis_pd['days_since_ws_covg'] = consensus_3mo_analysis_pd['date'] - consensus_3mo_analysis_pd['ws_min_date']
consensus_3mo_analysis_pd['days_since_ws_covg'] = consensus_3mo_analysis_pd['days_since_ws_covg'].dt.days


# a lot of tickers don't have a 3 mo. pre-period, so just limiting the post-period to 3 mo.
consensus_3mo_analysis_pd_clean = consensus_3mo_analysis_pd.loc[(consensus_3mo_analysis_pd['days_since_ws_covg'] < 90)]
consensus_3mo_analysis_pd_clean['pre_period'] = np.where(consensus_3mo_analysis_pd_clean['days_since_ws_covg'] < 0, True, False)

# Start basic mean & standard deviation analysis on 3 mo. data
print(consensus_3mo_analysis_pd_clean.groupby(['pre_period'])['rev_diff_percent', 'rev_diff_percent_abs'].mean())
print(consensus_3mo_analysis_pd_clean.groupby(['pre_period'])['rev_diff_percent', 'rev_diff_percent_abs'].var())

print(consensus_3mo_analysis_pd_clean.groupby(['pre_period'])['eps_diff_percent', 'eps_diff_percent_abs'].mean())
print(consensus_3mo_analysis_pd_clean.groupby(['pre_period'])['eps_diff_percent', 'eps_diff_percent_abs'].var())

consensus_3mo_analysis_pd_clean['num_predictors'] = consensus_3mo_analysis_pd_clean['estimize.eps.count']

# Run regression on 3 mo data (Exhibit 3a)
eps_ols_3mo = smf.ols('eps_diff_percent ~ C(ws_flag) + C(pre_period) + days_since_ws_covg + datediff_in_days', data=consensus_3mo_analysis_pd_clean).fit()
eps_ols_3mo.summary()  # R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change

rev_ols_3mo = smf.ols('rev_diff_percent ~ C(ws_flag) + C(pre_period) + days_since_ws_covg + datediff_in_days', data=consensus_3mo_analysis_pd_clean).fit()
rev_ols_3mo.summary()  # R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change

eps_ols_6mo = smf.ols('eps_diff_percent ~ C(ws_flag) + C(pre_period) + days_since_ws_covg + datediff_in_days + num_predictors', data=consensus_6mo_analysis_pd_clean).fit()
eps_ols_6mo.summary()  # R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change

rev_ols_6mo = smf.ols('rev_diff_percent ~ C(ws_flag) + C(pre_period) + days_since_ws_covg + datediff_in_days + num_predictors', data=consensus_6mo_analysis_pd_clean).fit()
rev_ols_6mo.summary()  # R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change

# Run a basic T-test on all data
eps_3mo_pre = consensus_3mo_analysis_pd_clean['eps_diff_percent'].loc[(consensus_3mo_analysis_pd_clean['pre_period'] == True)]
eps_3mo_post = consensus_3mo_analysis_pd_clean['eps_diff_percent'].loc[(consensus_3mo_analysis_pd_clean['pre_period'] == False)]

rev_3mo_pre = consensus_3mo_analysis_pd_clean['rev_diff_percent'].loc[(consensus_3mo_analysis_pd_clean['pre_period'] == True)]
rev_3mo_post = consensus_3mo_analysis_pd_clean['rev_diff_percent'].loc[(consensus_3mo_analysis_pd_clean['pre_period'] == False)]

eps_6mo_pre = consensus_6mo_analysis_pd_clean['eps_diff_percent'].loc[(consensus_6mo_analysis_pd_clean['pre_period'] == True)]
eps_6mo_post = consensus_6mo_analysis_pd_clean['eps_diff_percent'].loc[(consensus_6mo_analysis_pd_clean['pre_period'] == False)]

rev_6mo_pre = consensus_6mo_analysis_pd_clean['rev_diff_percent'].loc[(consensus_6mo_analysis_pd_clean['pre_period'] == True)]
rev_6mo_post = consensus_6mo_analysis_pd_clean['rev_diff_percent'].loc[(consensus_6mo_analysis_pd_clean['pre_period'] == False)]

# T-test shows that there is a difference
stats.ttest_ind(eps_3mo_pre, eps_3mo_post)
stats.ttest_ind(rev_3mo_pre, rev_3mo_post)


# %% Visualize Data


# Visualize 3 Mo. data with box lot
# Exhibit 2a
eps_boxplot_data = consensus_3mo_analysis_pd_clean[['eps_diff_percent', 'pre_period']]
eps_boxplot_data['pre_period'] = np.where(eps_boxplot_data['pre_period'] == True, "Pre-Period", "Post-Period")
eps_boxplot_data.boxplot(by='pre_period')

# Exhibit 2b
rev_boxplot_data = consensus_3mo_analysis_pd_clean[['rev_diff_percent', 'pre_period']]
rev_boxplot_data['pre_period'] = np.where(rev_boxplot_data['pre_period'] == True, "Pre-Period", "Post-Period")
rev_boxplot_data.boxplot(by='pre_period')

# Visualize 6 MO. data with box plot
eps_boxplot_data_6mo = consensus_6mo_analysis_pd_clean[['eps_diff_percent', 'pre_period']]
eps_boxplot_data_6mo['pre_period'] = np.where(eps_boxplot_data_6mo['pre_period'] == True, "Pre-Period", "Post-Period")
eps_boxplot_data_6mo.boxplot(by='pre_period')
stats.ttest_ind(eps_6mo_pre, eps_6mo_post)

rev_boxplot_data_6mo = consensus_6mo_analysis_pd_clean[['rev_diff_percent', 'pre_period']]
rev_boxplot_data_6mo['pre_period'] = np.where(rev_boxplot_data_6mo['pre_period'] == True, "Pre-Period", "Post-Period")
rev_boxplot_data_6mo.boxplot(by='pre_period')
stats.ttest_ind(rev_6mo_pre, rev_6mo_post)

# Try histograms
bins = np.linspace(-.5, .5, 20)
plt.hist(rev_3mo_pre, bins, alpha=1, label='pre')
plt.hist(rev_3mo_post, bins, alpha=1, label='post')
plt.legend(loc='upper right')
plt.show()

# Try histograms
bins = np.linspace(-.5, .5, 20)
plt.hist(eps_3mo_pre, bins, alpha=1, label='pre')
plt.hist(eps_3mo_post, bins, alpha=1, label='post')
plt.legend(loc='upper right')
plt.show()

# Try histograms
bins = np.linspace(-.5, .5, 20)
plt.hist(rev_6mo_pre, bins, alpha=1, label='pre')
plt.hist(rev_6mo_post, bins, alpha=1, label='post')
plt.legend(loc='upper right')
plt.show()

# Try histograms
bins = np.linspace(-.5, .5, 20)
plt.hist(eps_6mo_pre, bins, alpha=0.5, label='pre')
plt.hist(eps_6mo_post, bins, alpha=0.5, label='post')
plt.legend(loc='upper right')
plt.show()