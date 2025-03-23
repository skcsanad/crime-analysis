# %%
import itertools
import numpy as np
import pandas as pd
import os
import glob
import re
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# %%
state = 'ND-'
year = '2021'
file_list = os.listdir('Data/NIBRS_ND_2021/original')
csv_list = []
# list only csv files
for file in file_list:
    if re.search('.+csv', file):
        csv_list.append(file)
# dictionary of dataframes corresponding to each csv file, where the key is the
# dataframe name and the value is the actual data frame
df_dict = {file[:-4].lower(): pd.read_csv(os.path.join('..', 'Data', 'NIBRS_ND_2021', 'original', file))
           for file in csv_list}
# Since we chose the state and year, data_year and state columns don't give any extra information
# so we drop them
for key, val in df_dict.items():
    val=val[val.columns.drop(list(val.filter(regex='data_year')))].copy()
    val=val[val.columns.drop(list(val.filter(regex='state')))].copy()
    df_dict[key]=val
# Now we create variables for all the dictionary keys and assign to them
# the value from the respective value of the dictionary according to their key
locals().update(df_dict)

# %%
# Combining the different dataframes
df_combined = pd.merge(nibrs_victim_offense, nibrs_offense,
                       how='outer', on='offense_id')
df_combined = pd.merge(df_combined, nibrs_suspect_using,
                       how='outer', on='offense_id')
df_combined = pd.merge(df_combined, nibrs_bias_motivation,
                       how='outer', on='offense_id')
nibrs_offender = pd.merge(nibrs_offender, nibrs_age,
                       how='outer',
                          on='age_id')
nibrs_offender = pd.merge(nibrs_offender, ref_race,
                          on='race_id')
nibrs_offender = pd.merge(nibrs_offender, nibrs_ethnicity,
                          on='ethnicity_id')
nibrs_victim = pd.merge(nibrs_victim, nibrs_age,
                       how='outer',
                        on='age_id')
nibrs_victim = pd.merge(nibrs_victim, ref_race, on='race_id')
nibrs_victim = pd.merge(nibrs_victim, nibrs_ethnicity,
                        on='ethnicity_id')
df_victim_offender = pd.merge(nibrs_victim, nibrs_offender,
                              on='incident_id', how='outer', suffixes=('_victim', '_offender'))
df_combined = pd.merge(df_combined, df_victim_offender,
                       how='outer', left_on=['incident_id', 'victim_id'], right_on=['incident_id',
                                                                                    'victim_id'])
df_combined = pd.merge(df_combined, nibrs_incident,
                       how='outer', on='incident_id')
df_combined = pd.merge(df_combined, nibrs_property,
                       how='outer', on='incident_id')
df_combined = pd.merge(df_combined, nibrs_property_desc,
                       how='outer', on='property_id')
df_combined = pd.merge(df_combined, nibrs_victim_offender_rel, how='outer', left_on=[
    'victim_id', 'offender_id'], right_on=[
        'victim_id', 'offender_id'])
df_combined = pd.merge(df_combined, nibrs_criminal_act,
                       how='outer', on='offense_id')
df_combined = pd.merge(df_combined, nibrs_weapon,
                       how='outer', on='offense_id')
df_combined = pd.merge(df_combined, nibrs_victim_injury,
                       how='outer', on='victim_id')
df_combined = pd.merge(df_combined, agencies,
                       how='outer', on='agency_id')
df_combined = pd.merge(df_combined, nibrs_offense_type, how='outer',
                       on='offense_code')
df_combined = pd.merge(df_combined, nibrs_prop_desc_type,
                       how='outer', on='prop_desc_id')
df_combined = pd.merge(df_combined, nibrs_location_type,
                       how='outer', on='location_id')
df_combined = pd.merge(df_combined, nibrs_bias_list,
                       how='outer', on='bias_id')
df_combined = pd.merge(df_combined, nibrs_weapon_type,
                       how='outer', on='weapon_id')
df_combined = pd.merge(df_combined, nibrs_relationship,
                       how='outer', on='relationship_id')
df_combined = pd.merge(df_combined, nibrs_victim_circumstances,
                       how='outer', on='victim_id')
df_combined = pd.merge(df_combined, nibrs_circumstances,
                       how='outer', on='circumstances_id')
df_combined = pd.merge(df_combined, nibrs_criminal_act_type,
                       how='outer', on='criminal_act_id')
df_combined = pd.merge(df_combined, nibrs_activity_type,
                       how='outer', on='activity_type_id')
df_combined = pd.merge(df_combined, nibrs_victim_type,
                       how='outer', on='victim_type_id')
df_combined = pd.merge(df_combined, nibrs_prop_loss_type,
                       how='outer', on='prop_loss_id')
df_combined = pd.merge(df_combined, nibrs_injury,
                       how='outer', on='injury_id')
df_combined = pd.merge(df_combined, nibrs_using_list, on='suspect_using_id')

# %%
# we create a copy of the combined dataframe to work on
df_cleaned=df_combined.copy()
# Let's see the complete dataframe
pd.set_option('display.max_columns', None)
df_cleaned

# %%
# Transform date to datetime format
df_cleaned['incident_date'] = pd.to_datetime(df_cleaned['incident_date'], format="%Y-%m-%d")

# Since crime is done by humans and we know that the date has an effect on human behavior,
# we do some feature engineering extracting new variables from the date data could be useful
# for our predictions. We extract day of the week,if it is weekend or a holiday, month,
# day of the month, and week of the year
df_cleaned['incident_day'] = df_cleaned['incident_date'].dt.dayofweek
df_cleaned['incident_isweekend'] = df_cleaned['incident_day'] > 4
cal = USFederalHolidayCalendar()
holidays = cal.holidays()
df_cleaned['incident_is_holiday'] = df_cleaned['incident_date'].isin(holidays)
df_cleaned['incident_month'] = df_cleaned['incident_date'].dt.month
df_cleaned['incident_dayofmonth'] = df_cleaned['incident_date'].dt.day
df_cleaned['incident_weekofyear'] =df_cleaned['incident_date'].dt.isocalendar().week

# %%
# Splitting crimes to different categories (crimes againts people, society and property)
grouped = df_cleaned.groupby(df_cleaned.crime_against)
df_person = grouped.get_group("Person")
df_society = grouped.get_group("Society")
df_property = grouped.get_group("Property")

# %%
# Indicating whether the incident happened on a weekend or a US national holiday
weekends=sorted(df_person[df_person['incident_isweekend']==True].incident_date.unique())
holidays=sorted(df_person[df_person['incident_is_holiday']==True].incident_date.unique())
df_person.groupby('incident_date')['incident_id'].nunique().plot(title='incident_date'+' time series',figsize=(12,6))
i = 0
while i < len(weekends)-1:
    plt.axvspan(weekends[i], weekends[i+1]+np.timedelta64(1,'D'), facecolor='green', edgecolor='none', alpha=.2)
    i += 2
for i in range(len(holidays)):
    plt.axvspan(holidays[i], holidays[i]+np.timedelta64(1,'D'), facecolor='red', edgecolor='none', alpha=.5)

# %%
# Remove columns with only null values
for df in [df_person,df_property,df_society]:
    df.dropna(thresh=2, axis=1,inplace=True)

# Remove all columns where there is a class that represents more than 90% of the data
for df in [df_person,df_property,df_society]:
    very_imbalanced_columns=[]
    for col in tqdm(df.columns):
        if df[col].value_counts(normalize=True,dropna=False).max() >= 0.90:
            very_imbalanced_columns.append(col)
    df.drop(very_imbalanced_columns, axis=1,inplace=True)

# Remove columns with more than 90% null values
for df in [df_person,df_property,df_society]:
    # Remove all columns where there is more than 90% null values
    to_remove=[]
    for column,null_percent in enumerate((df.isna().sum()/df.shape[0]).to_frame().values):
        if null_percent >0.90:
            to_remove.append(column)
            #print(null_percent,column)
    df.drop(columns=df.columns[to_remove], inplace=True)

# %%
# Remove irrelevant columns
for df in [df_person,df_property,df_society]:
    df.drop(columns=['num_premises_entered',
                          'age_range_low_num_victim',
                         'notes_victim'
                         ,'population_group_desc','county_name',
                         'age_range_low_num_offender',
                         'notes_offender', 'submission_date',
                         'cleared_except_date','offense_category_name',
                         'incident_status', 'did','suspected_drug_name','drug_measure_name',
                         'date_recovered', 'ori', 'legacy_ori',
                         'ucr_agency_name', 'ncic_agency_name',
                         'pub_agency_unit','prop_loss_desc',
                         'suburban_area_flag', 'parent_pop_group_desc',
                         'mip_flag', 'pe_reported_flag',
                         'nibrs_start_date', 'nibrs_leoka_start_date',
                         'nibrs_ct_start_date', 'msa_name', 'ct_flag',
                         'age_range_high_num', 'est_drug_qty',
                         'male_officer','male_civilian','criminal_act_desc',
                         'male_officer+male_civilian', 'female_officer',
                         'female_civilian', 'female_officer+female_civilian',
                         'officer_rate', 'hc_flag', 'shr_flag', 'employee_rate',
                         ], inplace=True, axis=1,errors='ignore')

# %%
# Looking at nibrs_age csv we see that there are a minority of ages that don't correspond to a specific age. We'll remove this for simplicity
df_person=df_person.loc[~df_person['age_id_victim'].isin([1,2,3,103,104])]
df_person=df_person.loc[~df_person['age_id_offender'].isin([1,2,3,103,104])]

# Let's remove columns that don't give valuable information like id and code columns
for df in [df_person,df_property,df_society]:
    (df.drop(columns=[i for i in list(df.filter(regex='_id'))],errors='ignore',inplace=True))
    (df.drop(columns=[i for i in list(df.filter(regex='_code')) if i!='resident_status_code'],errors='ignore',inplace=True))

# Transform age to numeric
df_person["age_num_victim"] = pd.to_numeric(df_person["age_num_victim"])
df_person["age_num_offender"] = pd.to_numeric(df_person["age_num_offender"])

# %%
inputFeature = df_person[['victim_seq_num','age_num_victim','resident_status_code','race_desc_victim',
'ethnicity_name_victim','incident_hour','pub_agency_name','offense_name','location_name','weapon_name','population'
,'injury_name','incident_day','incident_isweekend','incident_month','incident_dayofmonth','incident_weekofyear','relationship_name']]
numerical_output = df_person[['age_num_offender']]
categorial_output = df_person[['race_desc_offender']]
inputFeature

# %%
if not os.path.exists('Data/NIBRS_ND_2021/processed'):
    os.mkdir('Data/NIBRS_ND_2021/processed')
inputFeature.to_csv('Data/NIBRS_ND_2021/processed/input.csv'), numerical_output.to_csv('Data/NIBRS_ND_2021/processed/num_out.csv'), categorial_output.to_csv('Data/NIBRS_ND_2021/processed/cat_out.csv')

# %%



