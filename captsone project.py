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
file_path = '/Users/laurendavis/Desktop/Wharton/Spring 2021/Data Science for Finance/Final Project/Raw Data/estimize-equities-2020q4/combined_consensus_new.csv'

consensus = pd.read_csv(file_path)

consensus.head()

estimates = pd.read_csv(file_path)

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

#WS Figures
#EPS

consensus['eps_diff_ws'] = consensus['wallstreet.eps'] - consensus['actual.eps']
#consensus['eps_diff_abs'] = consensus['eps_diff'].abs()
consensus['eps_diff_percent_ws'] = (consensus['wallstreet.eps'] - consensus['actual.eps'])/consensus['actual.eps']
consensus['eps_diff_percent_ws_abs'] = consensus[['eps_diff_percent_ws']].abs()

#Revenue
consensus['rev_diff_ws'] = consensus['wallstreet.revenue'] - consensus['actual.revenue']
consensus['rev_diff_percent_ws'] = (consensus['wallstreet.revenue'] - consensus['actual.revenue'])/consensus['actual.revenue']
consensus['rev_diff_percent_ws_abs'] = consensus[['rev_diff_percent_ws']].abs()


# %% Get only first instances
# Create a consensus_2 df that includes only estimates within 6 months of WS first flag

cutoff = 180   # required days on either side of teh WS flag

# Find first instane a ticker has no wall street estimate
FirstEST = consensus[['ticker', 'date', 'ws_flag']]
FirstEST = FirstEST[FirstEST['ws_flag'] == False]
FirstEST = FirstEST.groupby('ticker')['date'].min()
FirstEST = FirstEST.to_frame()

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
WSData = FirstWS.merge(LastWS, how='outer', left_on='ticker', right_on='ticker', suffixes=("_Min", " _Max"))
WSData['WS_Duration'] = WSData['date_Max'] - WSData['date_Min']
WSData['WS_Duration'] = WSData['WS_Duration'].astype("timedelta64[D]")

# Keep only instanes with a wallstreet coverage duration longer than cutoff
WSData = WSData.loc[(WSData['WS_Duration']) >= cutoff]

# Merge wall street data with estimize estimate data
TickerPull = FirstEST.merge(WSData, how='outer', left_on='ticker', right_on='ticker', suffixes=("_False", "_True"))
TickerPull['EstDur'] = TickerPull['date_Min'] - TickerPull['date']
TickerPull['EstDur'] = TickerPull['EstDur'].astype("timedelta64[D]")

# Keep only instances with an estimize only duration greater than cutoff
TickerPull = TickerPull.loc[(TickerPull['EstDur']) >= cutoff]
         
                   
#%% Create DataFrame that has estimates in relation to First instance of WS <-- NEEDS UPDATING BASED ON ABOVE

consensus_2 = consensus.merge(FirstWS, how='left', left_on='ticker', right_on='ticker', suffixes=(None, "_WS"))
consensus_2['date_to_ws'] = consensus_2['date']-consensus_2['date_WS']
consensus_2['date_to_ws'] = consensus_2['date_to_ws'].astype("timedelta64[D]")
consensus_2 = consensus_2.loc[(consensus_2['date_to_ws'].abs() <= 180)]

#Below not what I wanted but an example
consensus_2.plot.scatter(x='date_to_ws', y = 'eps_diff_percent').set_ylim(-100,100)

#%%Cleaning up data more

# Get earliest, latest reporting dates from consensus data
estimize_dateRange = consensus.groupby('ticker')['date'].min()
estimize_dateRange =estimize_dateRange.to_frame()
estimize_dateRange['max_date'] = consensus.groupby('ticker')['date'].max()

estimize_dateRange = estimize_dateRange.rename(columns={"date": "est_min_date", "max_date": "est_max_date"})

ticker_analysis_pd = estimize_dateRange.merge(FirstWS, left_on='ticker', right_on='ticker')
ticker_analysis_pd = ticker_analysis_pd.rename(columns={"date": "ws_min_date"})

#Pre-period

ticker_analysis_pd['pre_period_length'] = ticker_analysis_pd['ws_min_date'] - ticker_min_preperiod['est_min_date']
ticker_analysis_pd['pre_period_length'] = ticker_analysis_pd['pre_period_length'].dt.days

#Post-period
ticker_analysis_pd['post_period_length'] = ticker_analysis_pd['est_max_date'] - ticker_analysis_pd['ws_min_date']
ticker_analysis_pd['post_period_length'] = ticker_analysis_pd['post_period_length'].dt.days

#Create 6 mo. pre and post period
tick_analysis_6mo_pd = ticker_analysis_pd.loc[(ticker_analysis_pd['pre_period_length'])>= cutoff]
tick_analysis_6mo_pd = tick_analysis_6mo_pd.loc[(tick_analysis_6mo_pd['post_period_length'])>= cutoff]

#Create 3 mo. pre and post period

tick_analysis_3mo_pd = ticker_analysis_pd.loc[(ticker_analysis_pd['pre_period_length'])>= (cutoff/2)]
tick_analysis_3mo_pd = tick_analysis_3mo_pd.loc[(tick_analysis_3mo_pd['post_period_length'])>= (cutoff/2)]

#%%JJ to share with Lauren
#Check to see how many times WS drops coverage

consensus = consensus[consensus['eps_diff_percent_abs'] < 5] #removes some outliers (what should be the right cutoffs)
consensus = consensus[consensus['rev_diff_percent_abs'] < 5] # same as above

#Pull consensus data for tickers with 6 month pre & post
consensus_6mo_analysis_pd = consensus.merge(tick_analysis_6mo_pd, left_on = 'ticker', right_on = 'ticker')
consensus_6mo_analysis_pd['days_since_ws_covg'] = consensus_6mo_analysis_pd['date'] - consensus_6mo_analysis_pd['ws_min_date']
consensus_6mo_analysis_pd['days_since_ws_covg'] = consensus_6mo_analysis_pd['days_since_ws_covg'].dt.days

consensus_6mo_analysis_pd_clean = consensus_6mo_analysis_pd.loc[(abs(consensus_6mo_analysis_pd['days_since_ws_covg']) < 700)]

#Plot
plt.plot(consensus_6mo_analysis_pd_clean['days_since_ws_covg'], consensus_6mo_analysis_pd_clean['rev_diff_percent_abs'], 'o', color='black')
plt.show()

plt.plot(consensus_6mo_analysis_pd_clean['days_since_ws_covg'], consensus_6mo_analysis_pd_clean['eps_diff_percent_abs'], 'o', color='black')
plt.show()

#Add flag for pre-post
consensus_6mo_analysis_pd_clean['pre_period'] = np.where(consensus_6mo_analysis_pd_clean['days_since_ws_covg']<0,True,False)

print(consensus_6mo_analysis_pd_clean.groupby(['pre_period'])['rev_diff_percent', 'rev_diff_percent_abs'].mean())
print(consensus_6mo_analysis_pd_clean.groupby(['pre_period'])['rev_diff_percent', 'rev_diff_percent_abs'].var())

print(consensus_6mo_analysis_pd_clean.groupby(['pre_period'])['eps_diff_percent', 'eps_diff_percent_abs'].mean())
print(consensus_6mo_analysis_pd_clean.groupby(['pre_period'])['eps_diff_percent', 'eps_diff_percent_abs'].var())

eps_ols_6mo = smf.ols('eps_diff_percent_abs ~ C(ws_flag) + C(pre_period) + days_since_ws_covg + datediff_in_days', data = consensus_6mo_analysis_pd_clean).fit()
eps_ols_6mo.summary()
    ##FINDING --> it seems that EPS prediciton errors are higher when we are in the pre-period (i.e. before WS Covg)

rev_ols_6mo = smf.ols('rev_diff_percent_abs ~ C(ws_flag) + C(pre_period) + days_since_ws_covg + datediff_in_days', data = consensus_6mo_analysis_pd_clean).fit()
rev_ols_6mo.summary() #R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change
    ##FINDING --> it seems that Rev prediciton errors are higher when we are in the pre-period (i.e. before WS Covg)



#Pull consensus data for tickers with 3 month pre &  to test on a slightly smaller sample
consensus_3mo_analysis_pd = consensus.merge(tick_analysis_3mo_pd, left_on = 'ticker', right_on = 'ticker')
consensus_3mo_analysis_pd['days_since_ws_covg'] = consensus_3mo_analysis_pd['date'] - consensus_6mo_analysis_pd['ws_min_date']
consensus_3mo_analysis_pd['days_since_ws_covg'] = consensus_3mo_analysis_pd['days_since_ws_covg'].dt.days

#a lot of tickers don't have a 3 mo. pre-period, so just limiting the post-period to 3 mo.
consensus_3mo_analysis_pd_clean = consensus_3mo_analysis_pd.loc[(consensus_3mo_analysis_pd['days_since_ws_covg'] < 90)]
consensus_3mo_analysis_pd_clean['pre_period'] = np.where(consensus_3mo_analysis_pd_clean['days_since_ws_covg']<0,True,False)


#Start analysis on 3 mo. data
print(consensus_3mo_analysis_pd_clean.groupby(['pre_period'])['rev_diff_percent', 'rev_diff_percent_abs'].mean())
print(consensus_3mo_analysis_pd_clean.groupby(['pre_period'])['rev_diff_percent', 'rev_diff_percent_abs'].var())

print(consensus_3mo_analysis_pd_clean.groupby(['pre_period'])['eps_diff_percent', 'eps_diff_percent_abs'].mean())
print(consensus_3mo_analysis_pd_clean.groupby(['pre_period'])['eps_diff_percent', 'eps_diff_percent_abs'].var())

eps_ols_3mo = smf.ols('eps_diff_percent ~ C(ws_flag) + C(pre_period) + days_since_ws_covg + datediff_in_days', data = consensus_3mo_analysis_pd_clean).fit()
eps_ols_3mo.summary() #R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change

rev_ols_3mo = smf.ols('rev_diff_percent ~ C(ws_flag) + C(pre_period) + days_since_ws_covg + datediff_in_days', data = consensus_3mo_analysis_pd_clean).fit()
rev_ols_3mo.summary() #R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change


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

#%%Boxplots galore
cons_ws_yes = consensus[consensus['ws_flag'] == True]
cons_ws_no = consensus[consensus['ws_flag'] != True]

#EPS Flag
   
eps = pd.DataFrame(cons_ws_yes['eps_diff_percent'])
eps['eps_diff_percent_no'] = cons_ws_no['eps_diff_percent']
    
eps.boxplot(column=['eps_diff_percent','eps_diff_percent_no'])
                                   


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

new_cons.groupby(['ws_flag'])['eps_diff_percent_abs','eps_diff_percent_ws_abs'].mean() # shows a significant difference in average absolute error in expected direction
new_cons.groupby(['ws_flag'])['rev_diff_percent_abs','rev_diff_percent_ws_abs'].mean() # same as above

###box plot with true false for each

#%% Run regression to test for size and significance of the dummy coefficient

# First set up dummy variable for ws flag that can be interpreted by regression

cons_dummy = pd.get_dummies(new_cons['ws_flag']) # create dummy variables for the wall street flag
new_cons['ws_dummy'] = cons_dummy.iloc[:,1] #add dummy variables to the main dataset
new_cons.rename(columns={"estimize.eps.count":"estimize_eps_count"})
new_cons['estimize_eps_count'] = new_cons['estimize.eps.count']

new_cons.dtypes

# Set up the regression
eps_ols1 = smf.ols('eps_diff_percent_abs ~ C(ws_dummy) + datediff_in_days + estimize_eps_count', data = new_cons).fit()
eps_ols1.summary() #R2 is low (but not a big deal bc we aren't predicting), P-value is significant and coefficient suggests meaningful change

rev_ols1 = smf.ols('rev_diff_percent_abs ~ C(ws_dummy) + datediff_in_days + estimize_eps_count', data = new_cons).fit()
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
