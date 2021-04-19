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
consensus['eps_diff'] = consensus['estimize.eps.weighted'] - consensus['actual.eps']
#consensus['eps_diff_abs'] = consensus['eps_diff'].abs()
consensus['eps_diff_percent'] = (consensus['estimize.eps.weighted'] - consensus['actual.eps'])/consensus['actual.eps']
consensus['eps_diff_percent_abs'] = consensus[['eps_diff_percent']].abs()

#Revenue
consensus['rev_diff_abs'] = consensus['estimize.revenue.weighted'] - consensus['actual.revenue']
consensus['rev_diff_percent'] = (consensus['estimize.revenue.weighted'] - consensus['actual.revenue'])/consensus['actual.revenue']
consensus['rev_diff_percent_abs'] = consensus[['rev_diff_percent']].abs()



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
cons_tickers = new_cons[new_cons['ws_flag'] == True]
tickers = cons_tickers.groupby(['ticker'])['ticker'].nunique()
tickers = tickers.index
tickers = tickers.to_series()
new_cons = consensus[consensus['ticker'].isin(tickers)]

#%% Initial Analysis to see if erros vary between our ws flag

new_cons.groupby(['ws_flag'])['eps_diff_percent_abs'].mean() # shows a significant difference in average absolute error in expected direction
new_cons.groupby(['ws_flag'])['rev_diff_percent_abs'].mean() # same as above

###box plot with true false for each

#%% Run regression to test for size and significance of the dummy coefficient

# First set up dummy variable for ws flag that can be interpreted by regression

cons_dummy = pd.get_dummies(new_cons['ws_flag']) # create dummy variables for the wall street flag
new_cons['ws_dummy'] = cons_dummy.iloc[:,1] #add dummy variables to the main dataset

# Set up the regression
eps_ols1 = smf.ols('eps_diff_percent_abs ~ C(ws_dummy) + datediff_in_days', data = new_cons).fit()
eps_ols1.summary() #R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change

rev_ols1 = smf.ols('rev_diff_percent_abs ~ C(ws_dummy) + datediff_in_days', data = new_cons).fit()
rev_ols1.summary() #similar takeaways as above

#%% Create Sector Segments to view Errors

sector_errors = new_cons.groupby(['instrument_sector', 'ws_flag'])['eps_diff_percent_abs'].mean() # calculate average eps errors by industry
sector_errors = sector_errors.to_frame() # convert to dataframe
sector_errors = sector_errors.reset_index() 
sector_errors = sector_errors.pivot(index = 'instrument_sector', columns = 'ws_flag') #create pivot table to compare errors by industry / flag
sector_errors['error_diff'] = sector_errors['eps_diff_percent_abs',False] - sector_errors['eps_diff_percent_abs',True] # create new column for the difference in average errors

# Make Charts to show dispersion of errors by sector


#%% Identify ratio missing wallstreet cons to listed wall street cons per stock

est_missing = []
for i in tickers:
    df = new_cons[new_cons['ticker'].isin([i])]
    a = df['wallstreet.eps'].isna().sum() / df['wallstreet.eps'].count()
    est_missing.append(a)

est_missing = pd.DataFrame(est_missing, columns = ['est_missing'])
tickers_list = tickers.tolist()
est_missing['ticker'] = tickers_list

#%% Create new tickers list with stocks missing >50% of wall street cons

new_tickers = est_missing[est_missing['est_missing'] > 0.999]
new_tickers = new_tickers.drop(['est_missing'], axis = 1)
new_tickers = new_tickers.reset_index()
new_tickers = new_tickers.drop(['index'], axis = 1)

#%% Create database using filtered ticker list and re-run analysis

small_cons = consensus[consensus['ticker'].isin(new_tickers['ticker'])]
small_cons.groupby(['ws_flag'])['eps_diff_percent_abs'].mean() # shows a significant difference in average absolute error in unexpected direction
small_cons.groupby(['ws_flag'])['rev_diff_percent_abs'].mean() # not a significant difference

small_dummy = pd.get_dummies(small_cons['ws_flag']) # create dummy variables for the wall street flag
small_cons['ws_dummy'] = small_dummy.iloc[:,1] #add dummy variables to the main dataset

eps_ols2 = smf.ols('eps_diff_percent_abs ~ C(ws_dummy) + datediff_in_days', data = small_cons).fit()
eps_ols2.summary() #R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change

rev_ols2 = smf.ols('rev_diff_percent_abs ~ C(ws_dummy) + datediff_in_days', data = small_cons).fit()
rev_ols2.summary() #similar takeaways as above

#%%Sector analysis on filtered ticker list

small_sector_errors = small_cons.groupby(['instrument_sector', 'ws_flag'])['eps_diff_percent_abs'].mean() # calculate average eps errors by industry
small_sector_errors = small_sector_errors.to_frame() # convert to dataframe
small_sector_errors = small_sector_errors.reset_index() 
small_sector_errors = small_sector_errors.pivot(index = 'instrument_sector', columns = 'ws_flag') #create pivot table to compare errors by industry / flag
small_sector_errors['error_diff'] = small_sector_errors['eps_diff_percent_abs',False] - small_sector_errors['eps_diff_percent_abs',True] # create new column for the difference in average errors

#add some charts to show dispersion of errors by sector


#%% Custom stock list - focused on small/microcap names in the dataset with minimal coverage

final_stocks = ['XXII','BCPC','CALM','CKP','DLX','FDP','FRO','GIVN','GSC','HMTV','HOV','IOSP','ISCA','JBSS','KELY','LNCR','MHLD','MFB','MM','NVEC','OTTR','PDLI','PKE','SAFT','SWM','SPOK','TUEM','TOUR','VCLK','VMEM']
final_cons = consensus[consensus['ticker'].isin(final_stocks)]

#%% Repeat analysis from above

final_cons.groupby(['ws_flag'])['eps_diff_percent_abs'].mean() # shows a significant difference in average absolute error in expected direction
final_cons.groupby(['ws_flag'])['rev_diff_percent_abs'].mean() # 

final_dummy = pd.get_dummies(final_cons['ws_flag']) # create dummy variables for the wall street flag
final_cons['ws_dummy'] = final_dummy.iloc[:,1] #add dummy variables to the main dataset

eps_ols3 = smf.ols('eps_diff_percent_abs ~ C(ws_dummy) + datediff_in_days', data = final_cons).fit() #ws coverage seems to improve estimize user accuracy
eps_ols3.summary() 

rev_ols3 = smf.ols('rev_diff_percent_abs ~ C(ws_dummy) + datediff_in_days', data = final_cons).fit()
rev_ols3.summary() #same as above

#%% Next Steps
# Add charts for visual representation of the analysis
# Expand this analysis to look at trends in overestimates vs. underestimates
