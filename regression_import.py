import csv
import re
import pandas as pd



pd.options.mode.chained_assignment = None  # default='warn'
input_file = csv.DictReader(open('out.csv'))
input_file_1 = csv.DictReader(open('page.csv'))

#=================================================#
#+++++++++++++++++Load Data++++++++++++++++++++++#
#=================================================#

input_file_list = []
for row0 in input_file:
    input_file_list.append(row0) 

input_file1_list=[]
for row1 in input_file_1:
    input_file1_list.append(row1)


new_list = []
s =0
for item1 in input_file1_list:
    for item0 in input_file_list:
        if item1['Address'] == item0['Address']:
            crime_level = item1['Crime_level']
            #print crime_level
            item0['Crime_level'] = crime_level
            new_list.append(item0)
#=================================================#
#+++++++++++++++++Build DataFrame+++++++++++++++++#
#=================================================#

df = pd.DataFrame(input_file_list)
df1 = pd.DataFrame(input_file1_list)
df1_1 = df1[(df1['Housing_price'] !="NA")&(df1['Housing_price']!="")]
df_1 = df[(df['Housing_area']!="NA") & (df['Housing_area']!="")&(df['Housing_price']!="NA")&(df['Housing_price']!="")]
df1_1

df1_select = df1_1[['Address','Postal_code','Baths','Beds','Built_year','Crime_level','Housing_price', 'Housing_type']]
df_duplicates = pd.merge(df1_select,df_1[['Location','Address']],how="inner",on="Address")
#df_duplicates1 = pd.merge(df[['Housing_area','Address']],df_duplicates,how="inner",on="Address")
df_final = df_duplicates.drop_duplicates()
Baths_list = df_final['Baths'].values.tolist()
Beds_list = df_final['Beds'].values.tolist()
#df_final

#=================================================#
#+++++++++++++++++Preprocess Data+++++++++++++++++#
#=================================================#
Beds1 = [item.replace('Bedroom','').strip() for item in Beds_list]
Beds2 = []
for item in Beds1:
    if item !="NA":
        item = int(item)
    elif item == "NA":
        item = "NaN"
    Beds2.append(item)
Beds2
Bath1 = [item.replace('full Bathroom','').strip() for item in Baths_list]
Bath2 = []
for item in Bath1:
    if item=="" or item=="NA":
        item="NaN"
    else: item=int(item)  
    Bath2.append(item)

built_year_list = [item.replace('Built in','') for item in df_final['Built_year'].values.tolist()]
built_year_list2 = []
for item in built_year_list:
    if item !="NA":
        item = int(item)
    elif item == "NA":
        item = "NaN"
    built_year_list2.append(item)

    
crime_list = []  
for item in df_final['Crime_level'].values.tolist():
    if item=="NA":
        item = "High"
    crime_list.append(item)

housing_price_list = [float(re.sub(r'[/sqft/,/+/,/$]','',item)) for item in df_final['Housing_price'].values.tolist()]
#housing_area_list = [re.sub(r'[/sqft/,/+]','',item) for item in df_final['Housing_area'].values.tolist()]
latitude_list = [float(item.split(',')[0]) for item in df_final['Location'].values.tolist()]
longitude_list = [float(item.split(',')[1]) for item in df_final['Location'].values.tolist()]
df_final['Latitude'] = latitude_list
df_final['Longitude'] = longitude_list
#df_final['Housing_area'] = housing_area_list
df_final['Crime_level']=crime_list
df_final['Housing_price'] = housing_price_list
df_final['Built_year'] = built_year_list2
df_final['Baths'] = Bath2
df_final['Beds'] = Beds2
#del df_final['Location']



# deal with the categorical features
housing_type_dict = {'Condo':1,'Coop':2,'Single-Family Home':3,'Loft':4,'Townhome':5,'Multi-Family Home':6,'Apt/Condo/Twnhm':7,
                     'Apartment':8,'Unspecified property type':9}
housing_type_list_new = []
for item in df_final['Housing_type'].values.tolist():
    cat = housing_type_dict[item]
    housing_type_list_new.append(cat)
df_final['Housing_type'] = housing_type_list_new

crime_level_dict ={'Lowest':1,'Low':2,'Moderate':3,'High':4}
crime_level_list_new = []
for item in df_final['Crime_level'].values.tolist():
    cat = crime_level_dict[item]
    crime_level_list_new.append(cat)
df_final['Crime_level'] = crime_level_list_new


#=================================================#
#+++++++++++++++++Missing Value Imputation+++++++#
#=================================================#

import numpy as np
from sklearn.preprocessing import Imputer
bed_bath = [list(item) for item in zip(Beds2,Bath2)]
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(bed_bath)
transformed_list = imp.transform(bed_bath).tolist()
complete_beds_list= [item[0] for item in transformed_list]
complete_baths_list = [item[1] for item in transformed_list]
df_final['Beds'] = complete_beds_list
df_final['Baths'] = complete_baths_list
built_year_list = [item for item in df_final['Built_year'].values.tolist()]
housing_price_list = [item for item in df_final['Housing_price'].values.tolist()]
built_year_price = [list(item) for item in zip(built_year_list,housing_price_list)]
imp1 = Imputer(missing_values='NaN',strategy='median',axis=0)
imp1.fit(built_year_price)
transformed_list_builtyear = imp1.transform(built_year_price).tolist()
complete_builtyear_list = [item[0] for item in transformed_list_builtyear]
df_final['Built_year'] = complete_builtyear_list



from bokeh.charts import Bar, output_file, show


#p = Bar(df_final,label = 'Beds',values = 'Housing_price',agg= 'median')

#output_file("bar.html")

#show(p)



