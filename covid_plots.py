import pandas as pd
csvFile_qa = pd.read_csv('subset_all1.csv') # 2 qa
csvFile_site = pd.read_csv("subset_all2.csv") #site
csvFile = pd.read_csv("subset_all3.csv") #pre cont like csvFile2

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

import datetime
import seaborn as sns
import glob, zarr



csvFile['ch_pt'] = csvFile_site['ch_pt']
csvFile['ISO2'] = csvFile_site['ISO2']
csvFile['Date_reported'] = csvFile_qa['Date_reported']
csvFile['percent_ch'] = csvFile_qa['percent_ch']
csvFile = csvFile[csvFile['SITE'].notna()]
csvFile = csvFile[csvFile['SITE']!='AN']
csvFile = csvFile[(csvFile['qa_allPix']>=50) & (csvFile['qa_roll_avg_allPix']>=65)]
country_to_continent = {"BD": "AS", "BE": "EU", "BF": "AF", "BG": "EU", "BA": "EU", "BB": "NA", "WF": "OC", "BL": "NA", "BM": "NA", "BN": "AS", "BO": "SA", "BH": "AS", "BI": "AF", "BJ": "AF", "BT": "AS", "JM": "NA", "BV": "AN", "BW": "AF", "WS": "OC", "BQ": "NA", "BR": "SA", "BS": "NA", "JE": "EU", "BY": "EU", "BZ": "NA", "RU": "EU", "RW": "AF", "RS": "EU", "TL": "OC", "RE": "AF", "TM": "AS", "TJ": "AS", "RO": "EU", "TK": "OC", "GW": "AF", "GU": "OC", "GT": "NA", "GS": "AN", "GR": "EU", "GQ": "AF", "GP": "NA", "JP": "AS", "GY": "SA", "GG": "EU", "GF": "SA", "GE": "AS", "GD": "NA", "GB": "EU", "GA": "AF", "SV": "NA", "GN": "AF", "GM": "AF", "GL": "NA", "GI": "EU", "GH": "AF", "OM": "AS", "TN": "AF", "JO": "AS", "HR": "EU", "HT": "NA", "HU": "EU", "HK": "AS", "HN": "NA", "HM": "AN", "VE": "SA", "PR": "NA", "PS": "AS", "PW": "OC", "PT": "EU", "SJ": "EU", "PY": "SA", "IQ": "AS", "PA": "NA", "PF": "OC", "PG": "OC", "PE": "SA", "PK": "AS", "PH": "AS", "PN": "OC", "PL": "EU", "PM": "NA", "ZM": "AF", "EH": "AF", "EE": "EU", "EG": "AF", "ZA": "AF", "EC": "SA", "IT": "EU", "VN": "AS", "SB": "OC", "ET": "AF", "SO": "AF", "ZW": "AF", "SA": "AS", "ES": "EU", "ER": "AF", "ME": "EU", "MD": "EU", "MG": "AF", "MF": "NA", "MA": "AF", "MC": "EU", "UZ": "AS", "MM": "AS", "ML": "AF", "MO": "AS", "MN": "AS", "MH": "OC", "MK": "EU", "MU": "AF", "MT": "EU", "MW": "AF", "MV": "AS", "MQ": "NA", "MP": "OC", "MS": "NA", "MR": "AF", "IM": "EU", "UG": "AF", "TZ": "AF", "MY": "AS", "MX": "NA", "IL": "AS", "FR": "EU", "IO": "AS", "SH": "AF", "FI": "EU", "FJ": "OC", "FK": "SA", "FM": "OC", "FO": "EU", "NI": "NA", "NL": "EU", "NO": "EU", "NA": "AF", "VU": "OC", "NC": "OC", "NE": "AF", "NF": "OC", "NG": "AF", "NZ": "OC", "NP": "AS", "NR": "OC", "NU": "OC", "CK": "OC", "XK": "EU", "CI": "AF", "CH": "EU", "CO": "SA", "CN": "AS", "CM": "AF", "CL": "SA", "CC": "AS", "CA": "NA", "CG": "AF", "CF": "AF", "CD": "AF", "CZ": "EU", "CY": "EU", "CX": "AS", "CR": "NA", "CW": "NA", "CV": "AF", "CU": "NA", "SZ": "AF", "SY": "AS", "SX": "NA", "KG": "AS", "KE": "AF", "SS": "AF", "SR": "SA", "KI": "OC", "KH": "AS", "KN": "NA", "KM": "AF", "ST": "AF", "SK": "EU", "KR": "AS", "SI": "EU", "KP": "AS", "KW": "AS", "SN": "AF", "SM": "EU", "SL": "AF", "SC": "AF", "KZ": "AS", "KY": "NA", "SG": "AS", "SE": "EU", "SD": "AF", "DO": "NA", "DM": "NA", "DJ": "AF", "DK": "EU", "VG": "NA", "DE": "EU", "YE": "AS", "DZ": "AF", "US": "NA", "UY": "SA", "YT": "AF", "UM": "OC", "LB": "AS", "LC": "NA", "LA": "AS", "TV": "OC", "TW": "AS", "TT": "NA", "TR": "AS", "LK": "AS", "LI": "EU", "LV": "EU", "TO": "OC", "LT": "EU", "LU": "EU", "LR": "AF", "LS": "AF", "TH": "AS", "TF": "AN", "TG": "AF", "TD": "AF", "TC": "NA", "LY": "AF", "VA": "EU", "VC": "NA", "AE": "AS", "AD": "EU", "AG": "NA", "AF": "AS", "AI": "NA", "VI": "NA", "IS": "EU", "IR": "AS", "AM": "AS", "AL": "EU", "AO": "AF", "AQ": "AN", "AS": "OC", "AR": "SA", "AU": "OC", "AT": "EU", "AW": "NA", "IN": "AS", "AX": "EU", "AZ": "AS", "IE": "EU", "ID": "AS", "UA": "EU", "QA": "AS", "MZ": "AF"}
csvFile = csvFile[csvFile['ISO2'].notna()]
csvFile = csvFile[csvFile['ISO2']!='AN']
csvFile['Continent'] = csvFile.apply(lambda row: country_to_continent[row.ISO2], axis = 1)
csvFile[['Continent']] = csvFile[['Continent']].fillna(value='NA')
csvFile2 = csvFile
csvFile2 = csvFile2[(csvFile2.percent_ch != -100)]


low = ['2019-01-22', '2021-05-14', '2021-08-04', '2021-08-05'] 
csvFile2 = csvFile2[~csvFile2.Date_reported.isin(low)]       


df_new_mean = csvFile2.groupby(csvFile2['Date_reported'], as_index=False).mean()
df_new_median = csvFile2.groupby(csvFile2['Date_reported'], as_index=False).median()



fig = plt.figure(figsize=(20,12), dpi=1000)

df_new_mean.Date_reported = np.asarray(df_new_mean.Date_reported, dtype='datetime64[s]')
 
plt.plot(df_new_mean.Date_reported, df_new_mean.ch_pt, lw=6)

plt.xticks(fontsize=48)
plt.yticks([0.1,0.3,0.5,0.7], fontsize=48)
 
# Providing x and y label to the chart
plt.xlabel('Date',fontsize=48)
plt.ylabel('Proportion of Change Points',fontsize=48, labelpad=15)

fig.savefig("global_proportion_changepoints.pdf")

csvFile2 = csvFile
csvFile2 = csvFile2[(csvFile2.percent_ch != -100)]

eu = csvFile2[csvFile2['Continent']=='EU']
af = csvFile2[csvFile2['Continent']=='AF']
as1 = csvFile2[csvFile2['Continent']=='AS']
na = csvFile2[csvFile2['Continent']=='NA']
oc = csvFile2[csvFile2['Continent']=='OC']
sa = csvFile2[csvFile2['Continent']=='SA']

#low = ['2021-08-04','2021-08-05']
low=['2017-03-02', '2017-03-06', '2017-05-01', '2017-05-02', '2017-05-05', '2017-05-06', '2017-05-07', '2017-05-12', '2017-05-13', '2017-11-05', '2017-12-10', '2017-12-12', '2017-12-13', '2017-12-14', '2017-12-15', '2017-12-16', '2017-12-19', '2017-12-20', '2017-12-21', '2017-12-22', '2017-12-23', '2017-12-24', '2017-12-25', '2017-12-26', '2017-12-27', '2017-12-28', '2017-12-29', '2017-12-30', '2017-12-31', '2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08', '2018-01-09', '2018-01-10', '2018-01-11', '2018-01-12', '2018-01-13', '2018-01-14', '2018-01-15', '2018-01-16', '2018-01-17', '2018-01-18', '2018-01-19', '2018-01-20', '2018-01-21', '2018-01-22', '2018-01-23', '2018-01-24', '2018-01-25', '2018-01-26', '2018-01-27', '2018-01-29', '2018-01-30', '2018-01-31', '2018-02-01', '2018-02-05', '2018-02-07', '2018-02-13', '2018-02-15', '2018-02-16', '2018-02-18', '2018-02-19', '2018-02-20', '2018-02-21', '2018-02-22', '2018-02-23', '2018-02-24', '2018-02-25', '2018-02-26', '2018-02-27', '2018-02-28', '2018-03-01', '2018-03-02', '2018-03-03', '2018-03-04', '2018-03-05', '2018-03-06', '2018-03-07', '2018-03-08', '2018-03-09', '2018-03-10', '2018-03-11', '2018-03-12', '2018-03-13', '2018-03-14', '2018-03-15', '2018-03-16', '2018-03-17', '2018-03-18', '2018-03-19', '2018-03-20', '2018-03-23', '2018-03-24', '2018-03-25', '2018-03-26', '2018-03-27', '2018-03-28', '2018-03-29', '2018-03-30', '2018-03-31', '2018-04-08', '2018-04-11', '2018-05-14', '2018-05-16', '2018-05-17', '2018-05-19', '2018-05-20', '2018-05-21', '2018-05-22', '2018-05-23', '2018-05-24', '2018-05-25', '2018-06-06', '2018-11-02', '2018-11-03', '2018-11-04', '2018-11-05', '2018-11-06', '2018-11-07', '2018-11-08', '2018-11-09', '2018-11-10', '2018-11-11', '2018-11-12', '2018-11-13', '2018-11-14', '2018-11-15', '2018-11-16', '2018-11-17', '2018-11-18', '2018-11-19', '2018-11-20', '2018-11-21', '2018-11-22', '2018-11-23', '2018-11-24', '2018-11-25', '2018-11-26', '2018-11-27', '2018-11-28', '2018-11-29', '2018-11-30', '2018-12-01', '2018-12-02', '2018-12-03', '2018-12-04', '2018-12-05', '2018-12-14', '2018-12-15', '2018-12-16', '2018-12-20', '2018-12-21', '2019-01-31', '2019-04-21', '2019-04-22', '2019-04-23', '2019-04-24', '2019-04-25', '2019-04-26', '2019-04-27', '2019-04-28', '2019-04-29', '2019-04-30', '2019-05-01', '2019-05-02', '2019-05-03', '2019-05-04', '2019-05-05', '2019-05-09', '2019-05-10', '2019-05-11', '2019-05-19', '2019-05-20', '2019-05-25', '2019-11-13', '2019-11-14', '2019-11-15', '2019-11-16', '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22', '2019-11-25', '2019-11-26', '2019-11-27', '2019-11-28', '2019-11-29', '2019-11-30', '2019-12-01', '2019-12-02', '2019-12-03', '2019-12-04', '2019-12-05', '2019-12-06', '2019-12-07', '2019-12-08', '2019-12-09', '2019-12-10', '2019-12-11', '2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15', '2019-12-16', '2019-12-17', '2019-12-18', '2019-12-19', '2019-12-20', '2019-12-21', '2019-12-22', '2019-12-23', '2019-12-24', '2019-12-25', '2019-12-26', '2019-12-27', '2019-12-28', '2020-01-30', '2020-03-03', '2020-03-04', '2020-03-09', '2020-03-10', '2020-03-11', '2020-03-13', '2020-03-14', '2020-04-30', '2020-05-01', '2020-05-02', '2020-05-03', '2020-05-04', '2020-05-05', '2020-05-06', '2020-05-07', '2020-05-08', '2020-05-09', '2020-05-10', '2020-05-11', '2020-05-12', '2020-05-13', '2020-05-14', '2020-05-15', '2020-05-16', '2020-05-17', '2020-05-18', '2020-05-19', '2020-05-20', '2020-05-21', '2020-05-22', '2020-05-23', '2020-11-06', '2020-11-19', '2020-11-24', '2020-11-26', '2020-11-27', '2020-11-28', '2020-11-29', '2020-11-30', '2020-12-01', '2020-12-02', '2020-12-03', '2020-12-04', '2020-12-05', '2020-12-06', '2020-12-07', '2020-12-08', '2020-12-09', '2020-12-10', '2020-12-15', '2020-12-28', '2021-01-11', '2021-01-12', '2021-01-13', '2021-01-14', '2021-01-15', '2021-01-16', '2021-01-17', '2021-01-18', '2021-01-19', '2021-01-20', '2021-01-21', '2021-01-22', '2021-01-23', '2021-01-24', '2021-01-25', '2021-01-26', '2021-01-27', '2021-01-28', '2021-01-29', '2021-01-30', '2021-01-31', '2021-02-01', '2021-02-02', '2021-02-03', '2021-02-04', '2021-02-05', '2021-02-06', '2021-02-07', '2021-02-08', '2021-02-09', '2021-02-10', '2021-02-11', '2021-02-12', '2021-02-13', '2021-02-14', '2021-02-15', '2021-02-16', '2021-02-17', '2021-02-18', '2021-02-21', '2021-04-29', '2021-04-30', '2021-05-01', '2021-05-02', '2021-05-03', '2021-05-04', '2021-05-07', '2021-05-09', '2021-05-10', '2021-08-05', '2021-11-21', '2021-11-22', '2021-12-01', '2021-12-03', '2021-12-08', '2021-12-09', '2021-12-10', '2021-12-11', '2021-12-12', '2021-12-13', '2021-12-14', '2021-12-15', '2021-12-16', '2021-12-17', '2021-12-18', '2021-12-19', '2021-12-21', '2021-12-22', '2021-12-26']
['2018-02-14', '2018-04-12', '2021-08-05']
eu = eu[~eu.Date_reported.isin(low)] 

low=['2018-02-14', '2018-04-12', '2021-08-05']
af = af[~af.Date_reported.isin(low)] 

low=['2017-09-30', '2018-05-12', '2019-01-22', '2021-04-28', '2021-05-02', '2021-05-03', '2021-05-04', '2021-05-05', '2021-05-06', '2021-05-07', '2021-05-08', '2021-05-14', '2021-07-25', '2021-07-29', '2021-08-04', '2021-08-05']
as1 = as1[~as1.Date_reported.isin(low)] 

low=['2017-09-30', '2019-11-01', '2021-08-04', '2021-08-05']
na = na[~na.Date_reported.isin(low)] 

low=['2017-04-08', '2017-04-09', '2017-04-16', '2017-04-28', '2017-04-30', '2017-05-08', '2017-05-09', '2017-05-10', '2017-05-11', '2017-05-12', '2017-05-13', '2017-05-14', '2017-05-17', '2017-05-20', '2017-05-21', '2017-05-22', '2017-05-23', '2017-09-30', '2017-10-17', '2017-10-22', '2017-10-24', '2017-10-25', '2017-10-27', '2017-10-28', '2017-10-29', '2017-10-30', '2017-10-31', '2017-11-03', '2017-11-04', '2017-11-05', '2017-11-06', '2017-11-07', '2017-11-09', '2018-01-14', '2018-11-24', '2019-04-01', '2019-04-16', '2019-04-19', '2019-04-20', '2019-04-21', '2019-04-22', '2019-04-24', '2019-04-25', '2019-04-27', '2019-05-03', '2021-08-04', '2021-08-05']
sa = sa[~sa.Date_reported.isin(low)] 

low=['2017-03-13', '2017-05-16', '2017-06-28', '2017-09-26', '2017-09-27', '2017-09-28', '2017-09-29', '2017-09-30', '2017-10-01', '2017-10-02', '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07', '2017-10-08', '2017-10-09', '2017-10-10', '2017-10-14', '2017-10-15', '2017-10-16', '2017-10-19', '2017-10-20', '2017-10-21', '2017-10-28', '2017-11-03', '2017-11-17', '2017-11-18', '2017-11-24', '2017-11-28', '2017-12-04', '2017-12-08', '2018-01-15', '2018-01-18', '2018-02-02', '2018-02-03', '2018-02-17', '2018-02-27', '2018-03-10', '2018-03-11', '2018-03-12', '2018-03-13', '2018-03-14', '2018-03-15', '2018-03-16', '2018-03-17', '2018-03-18', '2018-03-19', '2018-03-20', '2018-03-21', '2018-03-22', '2018-03-23', '2018-03-24', '2018-03-25', '2018-03-26', '2018-03-27', '2018-03-28', '2018-03-29', '2018-03-30', '2018-03-31', '2018-04-01', '2018-04-02', '2018-04-03', '2018-04-04', '2018-04-05', '2018-06-28', '2018-07-02', '2018-10-28', '2018-11-19', '2018-11-22', '2018-11-23', '2018-12-04', '2018-12-06', '2018-12-15', '2019-02-21', '2019-03-04', '2019-03-06', '2019-03-08', '2019-03-14', '2019-03-15', '2019-03-16', '2019-03-17', '2019-03-18', '2019-03-19', '2019-03-20', '2019-03-21', '2019-03-22', '2019-03-23', '2019-05-31', '2019-06-02', '2019-06-03', '2019-06-10', '2019-06-17', '2019-12-16', '2020-02-04', '2020-02-14', '2020-02-15', '2020-02-16', '2020-02-17', '2020-02-18', '2020-02-19', '2020-02-22', '2020-02-23', '2020-02-24', '2020-02-25', '2020-02-26', '2020-02-27', '2020-02-28', '2020-02-29', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-05', '2020-03-08', '2020-03-09', '2020-03-10', '2020-03-11', '2020-03-12', '2020-03-13', '2020-03-14', '2020-03-15', '2020-03-16', '2020-03-17', '2020-03-18', '2020-03-20', '2020-04-07', '2020-04-18', '2020-05-20', '2020-05-21', '2020-05-22', '2020-05-23', '2020-05-24', '2020-05-25', '2020-05-27', '2020-05-28', '2020-05-29', '2020-05-30', '2020-05-31', '2020-06-01', '2020-06-02', '2020-06-03', '2020-06-04', '2020-06-05', '2020-06-06', '2020-06-07', '2020-07-11', '2020-07-13', '2020-08-05', '2020-08-07', '2020-11-29', '2020-12-02', '2021-01-03', '2021-02-10', '2021-02-14', '2021-02-16', '2021-02-17', '2021-02-18', '2021-02-19', '2021-02-20', '2021-02-21', '2021-02-22', '2021-02-23', '2021-02-24', '2021-02-25', '2021-02-26', '2021-02-27', '2021-02-28', '2021-03-01', '2021-03-02', '2021-03-03', '2021-03-04', '2021-03-07', '2021-03-11', '2021-03-13', '2021-03-26', '2021-06-23', '2021-06-24', '2021-06-25', '2021-06-26', '2021-06-29', '2021-06-30', '2021-07-01', '2021-07-02', '2021-07-08', '2021-07-09', '2021-07-10', '2021-07-13', '2021-07-14', '2021-07-15', '2021-08-04', '2021-08-05', '2021-10-02', '2021-10-06', '2021-10-07', '2021-10-08', '2021-10-09', '2021-10-10', '2021-10-11', '2021-10-12', '2021-10-13', '2021-10-14', '2021-10-15', '2021-10-18', '2021-10-19', '2021-10-20', '2021-10-21', '2021-10-22', '2021-10-23', '2021-10-24', '2021-10-25', '2021-10-26', '2021-10-27', '2021-10-28', '2021-10-29', '2021-10-30', '2021-10-31', '2021-11-01', '2021-11-02', '2021-11-03', '2021-11-05', '2021-11-09', '2021-11-22', '2021-11-28', '2021-11-30', '2021-12-10', '2021-12-18', '2022-02-03', '2022-02-12']
oc = oc[~oc.Date_reported.isin(low)] 

df_new_eu = eu.groupby(eu['Date_reported'], as_index=False).aggregate({'ch_pt': 'sum'})
df_new_af = af.groupby(af['Date_reported'], as_index=False).aggregate({'ch_pt': 'sum'})
df_new_as1 = as1.groupby(as1['Date_reported'], as_index=False).aggregate({'ch_pt': 'sum'})
df_new_na = na.groupby(na['Date_reported'], as_index=False).aggregate({'ch_pt': 'sum'})
df_new_oc = oc.groupby(oc['Date_reported'], as_index=False).aggregate({'ch_pt': 'sum'})
df_new_sa = sa.groupby(sa['Date_reported'], as_index=False).aggregate({'ch_pt': 'sum'})

df_new_eu_mean = eu.groupby(eu['Date_reported'], as_index=False).mean()
df_new_af_mean = af.groupby(af['Date_reported'], as_index=False).mean()
df_new_as1_mean = as1.groupby(as1['Date_reported'], as_index=False).mean()
df_new_na_mean = na.groupby(na['Date_reported'], as_index=False).mean()
df_new_oc_mean = oc.groupby(oc['Date_reported'], as_index=False).mean()
df_new_sa_mean = sa.groupby(sa['Date_reported'], as_index=False).mean()

df_new_eu_med = eu.groupby(eu['Date_reported'], as_index=False).median()
df_new_af_med = af.groupby(af['Date_reported'], as_index=False).median()
df_new_as1_med = as1.groupby(as1['Date_reported'], as_index=False).median()
df_new_na_med = na.groupby(na['Date_reported'], as_index=False).median()
df_new_oc_med = oc.groupby(oc['Date_reported'], as_index=False).median()
df_new_sa_med = sa.groupby(sa['Date_reported'], as_index=False).median()

labs = ['eu','af','as','na','oc','sa']
conts = [eu,af,as1,na,oc,sa]
for each,lab in zip(conts,labs):
    db2020 = each[pd.DatetimeIndex(each['Date_reported']).year == 2020]

    cnt, min_pc, date = [], [], []
    for j,i in enumerate(db2020.SITE.unique()):
        cnt.append(db2020[(db2020.SITE == i) & (db2020.percent_ch == db2020[db2020.SITE == i].percent_ch.min())].ISO2.tolist())
        min_pc.append(db2020[(db2020.SITE == i) & (db2020.percent_ch == db2020[db2020.SITE == i].percent_ch.min())].percent_ch.tolist())
        date.append(db2020[(db2020.SITE == i) & (db2020.percent_ch == db2020[db2020.SITE == i].percent_ch.min())].Date_reported.tolist())

    cnt = [item[0] for item in cnt]
    min_pc = [item[0] for item in min_pc]
    date = [item[0] for item in date]

    df = pd.DataFrame({'ISO2':cnt,'min_pc':min_pc,'Date_reported':date})

    #max drop 
    df['doy'] = pd.to_datetime(df['Date_reported']).dt.dayofyear

    df.to_csv("max_drop.csv")

    #change points 
    ch_pt = db2020[(db2020.ch_pt == 1)]
    ch_pt['doy'] = pd.to_datetime(ch_pt['Date_reported']).dt.dayofyear

    #change points negative  
    ch_pt_neg = db2020[(db2020.ch_pt == 1) & (db2020.percent_ch < 0)]
    ch_pt_neg['doy'] = pd.to_datetime(ch_pt_neg['Date_reported']).dt.dayofyear

    ch_pt.to_csv("ch_pt_neg.csv")

    import random
    chars = '0123456789ABCDEF'
    countries = [x for x in np.unique(df.ISO2)]
    colors = ['#'+''.join(random.sample(chars,6)) for i in range(len(countries))]

    gs = grid_spec.GridSpec(len(countries),1)
    if lab == 'oc' or lab == 'sa':
        fig = plt.figure(figsize=(16,10))
    else:
        fig = plt.figure(figsize=(16,20))

    i = 0

    ax_objs = []
    for iso in countries:
        iso = countries[i]
        x = np.array(df[df.ISO2 == iso].doy)
        x_d = np.array(range(1,367))

        kde = KernelDensity()
        kde.fit(x[:, None])

        logprob = kde.score_samples(x_d[:, None])

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

        # plotting the distribution
        ax_objs[-1].plot(x_d, np.exp(logprob),color="#f0f0f0",lw=1)
        ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1,color=colors[i])


        # setting uniform x and y lims
        ax_objs[-1].set_xlim(1,367)
        #ax_objs[-1].set_ylim(0,1)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_yticks([])


        if i == len(countries)-1:
            ax_objs[-1].set_xlabel("Month (2020)", fontsize=16,fontweight="bold")
            ax_objs[-1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'], fontsize=14)
            ax_objs[-1].set_xticks([i for i in range(2,365,30)])

        else:
            ax_objs[-1].set_xticklabels([])

        if i == round(len(countries)/2):
            if lab == 'oc' or lab == 'sa':
                ax_objs[-1].set_ylabel("Distribution of maximum NTL" "\n" "drop per country over time", fontsize=16,fontweight="bold", labelpad=30)

            else:
                ax_objs[-1].set_ylabel("Distribution of maximum NTL" "\n" "drop per country over time", fontsize=16,fontweight="bold", labelpad=30)

            

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        adj_country = iso.replace(" ","\n")
        ax_objs[-1].text(-0.02,0,adj_country,fontweight="bold",fontsize=14,ha="right")

        i += 1

    plt.tight_layout()
    

    fig.savefig(f"max_drop_{lab}.pdf",dpi=2400)



    import random
    chars = '0123456789ABCDEF'
    countries = [x for x in np.unique(ch_pt_neg.ISO2)]
    colors = ['#'+''.join(random.sample(chars,6)) for i in range(len(countries))]


    if lab == 'oc':
        gs = grid_spec.GridSpec(len(countries),1)
        fig = plt.figure(figsize=(16,4.5))

    i = 0

    ax_objs = []
    for iso in countries:
        iso = countries[i]
        x = np.array(ch_pt_neg[ch_pt_neg.ISO2 == iso].doy)
        x_d = np.array(range(1,367))

        kde = KernelDensity()
        kde.fit(x[:, None])

        logprob = kde.score_samples(x_d[:, None])

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

        # plotting the distribution
        ax_objs[-1].plot(x_d, np.exp(logprob),color="#f0f0f0",lw=1)
        ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1,color=colors[i])


        # setting uniform x and y lims
        ax_objs[-1].set_xlim(1,367)
        #ax_objs[-1].set_ylim(0,1)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_yticks([])

        if i == len(countries)-1:
            ax_objs[-1].set_xlabel("Month (2020)", fontsize=16,fontweight="bold")
            ax_objs[-1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'], fontsize=14)
            ax_objs[-1].set_xticks([i for i in range(2,365,30)])
        else:
            ax_objs[-1].set_xticklabels([])

        if i == round(len(countries)/2):
            ax_objs[-1].set_ylabel("Distribution of number of" "\n" "urban areas in each country" "\n" "where a decline in" "\n" "NTL was detected", fontsize=16,fontweight="bold", labelpad=30)

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        adj_country = iso.replace(" ","\n")
        ax_objs[-1].text(-0.02,0,adj_country,fontweight="bold",fontsize=14,ha="right")


        i += 1

    plt.tight_layout()
    

    fig.savefig(f"ch_pt_neg_{lab}.pdf",dpi=2400)


import matplotlib.pyplot as plt
plt.rc('font', size=56)

df_new_eu_mean.Date_reported = np.asarray(df_new_eu_mean.Date_reported, dtype='datetime64[s]')
df_new_as1_mean.Date_reported = np.asarray(df_new_as1_mean.Date_reported, dtype='datetime64[s]')
df_new_af_mean.Date_reported = np.asarray(df_new_af_mean.Date_reported, dtype='datetime64[s]')
df_new_na_mean.Date_reported = np.asarray(df_new_na_mean.Date_reported, dtype='datetime64[s]')
df_new_sa_mean.Date_reported = np.asarray(df_new_sa_mean.Date_reported, dtype='datetime64[s]')
df_new_oc_mean.Date_reported = np.asarray(df_new_oc_mean.Date_reported, dtype='datetime64[s]')


fig, axs = plt.subplots(2, 3, figsize=(35,18), dpi=1000, sharex=True)
axs[0,0].plot(df_new_eu_mean.Date_reported, df_new_eu_mean.ch_pt, lw=3)
axs[0,1].plot(df_new_as1_mean.Date_reported, df_new_as1_mean.ch_pt, lw=3)
axs[0,2].plot(df_new_af_mean.Date_reported, df_new_af_mean.ch_pt, lw=3)
axs[1,0].plot(df_new_na_mean.Date_reported, df_new_na_mean.ch_pt, lw=3)
axs[1,1].plot(df_new_sa_mean.Date_reported, df_new_sa_mean.ch_pt, lw=3)
axs[1,2].plot(df_new_oc_mean.Date_reported, df_new_oc_mean.ch_pt, lw=3)
fig.supxlabel('Date', fontsize=72)
fig.supylabel('Proportion of Change Points', x=0.01, fontsize=72)
plt.xticks(fontsize=56)
plt.yticks(fontsize=56)
start, end = axs[0,0].get_xlim()
axs[0,0].xaxis.set_ticks(['2018','2020','2022'],['2018','2020','2022'])
axs[0,1].xaxis.set_ticks(['2018','2020','2022'],['2018','2020','2022'])
axs[0,2].xaxis.set_ticks(['2018','2020','2022'],['2018','2020','2022'])
axs[1,0].xaxis.set_ticks(['2018','2020','2022'],['2018','2020','2022'])
axs[1,1].xaxis.set_ticks(['2018','2020','2022'],['2018','2020','2022'])
axs[1,2].xaxis.set_ticks(['2018','2020','2022'],['2018','2020','2022'])

axs[0,0].yaxis.set_ticks([0,0.5,1])
axs[0,1].yaxis.set_ticks([0.2,0.5,0.8])
axs[0,2].yaxis.set_ticks([0.2,0.5,0.8])
axs[1,0].yaxis.set_ticks([0.2,0.5,0.8])
axs[1,1].yaxis.set_ticks([0.2,0.5,0.8])
axs[1,2].yaxis.set_ticks([0,0.5,1])
#plt.rc('font', size=36) #controls default text size
plt.tight_layout()
fig.savefig("continental_proportion_changepoints_row1_europe_asia_africa_row2_na_sa_oc_25percent.pdf")



df_new_mean.to_csv("global_mean_percentch.csv")
df_new_median.to_csv("global_median_percentch.csv")

df_new_eu_med.to_csv("eu_med_percentch.csv")
df_new_af_med.to_csv("af_med_percentch.csv")
df_new_as1_med.to_csv("as1_med_percentch.csv")
df_new_na_med.to_csv("na_med_percentch.csv")
df_new_oc_med.to_csv("oc_med_percentch.csv")
df_new_sa_med.to_csv("sa_med_percentch.csv")

df_new_eu_mean.to_csv("eu_avg_percentch_and_chpts.csv")
df_new_af_mean.to_csv("af_avg_percentch_and_chpts.csv")
df_new_as1_mean.to_csv("as1_avg_percentch_and_chpts.csv")
df_new_na_mean.to_csv("na_avg_percentch_and_chpts.csv")
df_new_oc_mean.to_csv("oc_avg_percentch_and_chpts.csv")
df_new_sa_mean.to_csv("sa_avg_percentch_and_chpts.csv")















import pandas
csvFile3 = pandas.read_csv("subset_stl.csv") #iso2 date

csvFile = csvFile3

csvFile = csvFile[csvFile['SITE'].notna()]
csvFile = csvFile[csvFile['SITE']!='AN']
csvFile = csvFile[(csvFile['qa_allPix']>=50) & (csvFile['qa_roll_avg_allPix']>=65)]
country_to_continent = {"BD": "AS", "BE": "EU", "BF": "AF", "BG": "EU", "BA": "EU", "BB": "NA", "WF": "OC", "BL": "NA", "BM": "NA", "BN": "AS", "BO": "SA", "BH": "AS", "BI": "AF", "BJ": "AF", "BT": "AS", "JM": "NA", "BV": "AN", "BW": "AF", "WS": "OC", "BQ": "NA", "BR": "SA", "BS": "NA", "JE": "EU", "BY": "EU", "BZ": "NA", "RU": "EU", "RW": "AF", "RS": "EU", "TL": "OC", "RE": "AF", "TM": "AS", "TJ": "AS", "RO": "EU", "TK": "OC", "GW": "AF", "GU": "OC", "GT": "NA", "GS": "AN", "GR": "EU", "GQ": "AF", "GP": "NA", "JP": "AS", "GY": "SA", "GG": "EU", "GF": "SA", "GE": "AS", "GD": "NA", "GB": "EU", "GA": "AF", "SV": "NA", "GN": "AF", "GM": "AF", "GL": "NA", "GI": "EU", "GH": "AF", "OM": "AS", "TN": "AF", "JO": "AS", "HR": "EU", "HT": "NA", "HU": "EU", "HK": "AS", "HN": "NA", "HM": "AN", "VE": "SA", "PR": "NA", "PS": "AS", "PW": "OC", "PT": "EU", "SJ": "EU", "PY": "SA", "IQ": "AS", "PA": "NA", "PF": "OC", "PG": "OC", "PE": "SA", "PK": "AS", "PH": "AS", "PN": "OC", "PL": "EU", "PM": "NA", "ZM": "AF", "EH": "AF", "EE": "EU", "EG": "AF", "ZA": "AF", "EC": "SA", "IT": "EU", "VN": "AS", "SB": "OC", "ET": "AF", "SO": "AF", "ZW": "AF", "SA": "AS", "ES": "EU", "ER": "AF", "ME": "EU", "MD": "EU", "MG": "AF", "MF": "NA", "MA": "AF", "MC": "EU", "UZ": "AS", "MM": "AS", "ML": "AF", "MO": "AS", "MN": "AS", "MH": "OC", "MK": "EU", "MU": "AF", "MT": "EU", "MW": "AF", "MV": "AS", "MQ": "NA", "MP": "OC", "MS": "NA", "MR": "AF", "IM": "EU", "UG": "AF", "TZ": "AF", "MY": "AS", "MX": "NA", "IL": "AS", "FR": "EU", "IO": "AS", "SH": "AF", "FI": "EU", "FJ": "OC", "FK": "SA", "FM": "OC", "FO": "EU", "NI": "NA", "NL": "EU", "NO": "EU", "NA": "AF", "VU": "OC", "NC": "OC", "NE": "AF", "NF": "OC", "NG": "AF", "NZ": "OC", "NP": "AS", "NR": "OC", "NU": "OC", "CK": "OC", "XK": "EU", "CI": "AF", "CH": "EU", "CO": "SA", "CN": "AS", "CM": "AF", "CL": "SA", "CC": "AS", "CA": "NA", "CG": "AF", "CF": "AF", "CD": "AF", "CZ": "EU", "CY": "EU", "CX": "AS", "CR": "NA", "CW": "NA", "CV": "AF", "CU": "NA", "SZ": "AF", "SY": "AS", "SX": "NA", "KG": "AS", "KE": "AF", "SS": "AF", "SR": "SA", "KI": "OC", "KH": "AS", "KN": "NA", "KM": "AF", "ST": "AF", "SK": "EU", "KR": "AS", "SI": "EU", "KP": "AS", "KW": "AS", "SN": "AF", "SM": "EU", "SL": "AF", "SC": "AF", "KZ": "AS", "KY": "NA", "SG": "AS", "SE": "EU", "SD": "AF", "DO": "NA", "DM": "NA", "DJ": "AF", "DK": "EU", "VG": "NA", "DE": "EU", "YE": "AS", "DZ": "AF", "US": "NA", "UY": "SA", "YT": "AF", "UM": "OC", "LB": "AS", "LC": "NA", "LA": "AS", "TV": "OC", "TW": "AS", "TT": "NA", "TR": "AS", "LK": "AS", "LI": "EU", "LV": "EU", "TO": "OC", "LT": "EU", "LU": "EU", "LR": "AF", "LS": "AF", "TH": "AS", "TF": "AN", "TG": "AF", "TD": "AF", "TC": "NA", "LY": "AF", "VA": "EU", "VC": "NA", "AE": "AS", "AD": "EU", "AG": "NA", "AF": "AS", "AI": "NA", "VI": "NA", "IS": "EU", "IR": "AS", "AM": "AS", "AL": "EU", "AO": "AF", "AQ": "AN", "AS": "OC", "AR": "SA", "AU": "OC", "AT": "EU", "AW": "NA", "IN": "AS", "AX": "EU", "AZ": "AS", "IE": "EU", "ID": "AS", "UA": "EU", "QA": "AS", "MZ": "AF"}

csvFile = csvFile[csvFile['ISO2'].notna()]
csvFile = csvFile[csvFile['ISO2']!='AN']

csvFile['Continent'] = csvFile.apply(lambda row: country_to_continent[row.ISO2], axis = 1)

csvFile[['Continent']] = csvFile[['Continent']].fillna(value='NA')

csvFile['percent_ch'] = ((csvFile['diff']*100)/csvFile['Forecast'])

csvFile2 = csvFile
csvFile2 = csvFile2[(csvFile2.percent_ch != -100)]


low = ['2021-05-06', '2021-08-04', '2021-08-05']
csvFile2 = csvFile2[~csvFile2.Date_reported.isin(low)]   

csvFile2['rec'] = 0
csvFile2['rec'][csvFile2['diff'] > 0] = 1


df_new_mean = csvFile2.groupby(csvFile2['Date_reported'], as_index=False).mean()
df_new_median = csvFile2.groupby(csvFile2['Date_reported'], as_index=False).median()


df_new_mean.to_csv("global_mean_recovery_stl.csv")

eu = csvFile2[csvFile2['Continent']=='EU']
af = csvFile2[csvFile2['Continent']=='AF']
as1 = csvFile2[csvFile2['Continent']=='AS']
na = csvFile2[csvFile2['Continent']=='NA']
oc = csvFile2[csvFile2['Continent']=='OC']
sa = csvFile2[csvFile2['Continent']=='SA']

low=['2020-05-12', '2020-05-18', '2020-06-03', '2020-07-06', '2020-07-22', '2021-04-14', '2021-05-01', '2021-05-05', '2021-05-06', '2021-05-07', '2021-05-08', '2021-05-31', '2021-06-23', '2021-07-05', '2021-07-06', '2021-08-04', '2021-08-05', '2021-08-30', '2021-08-31', '2022-02-08']
eu = eu[~eu.Date_reported.isin(low)] 

low=['2020-06-03', '2020-07-06', '2020-07-22', '2020-07-23', '2021-05-05', '2021-05-06', '2021-06-23', '2021-07-05', '2021-07-06', '2021-08-04', '2021-08-05']
af = af[~af.Date_reported.isin(low)] 

low=['2021-08-04', '2021-08-05']
as1 = as1[~as1.Date_reported.isin(low)] 

low=['2020-03-14', '2020-05-05', '2020-05-15', '2020-05-22', '2020-05-29', '2020-05-30', '2020-05-31', '2020-06-02', '2020-06-03', '2020-06-13', '2020-06-14', '2020-06-20', '2020-06-24', '2020-07-06', '2020-07-10', '2020-07-15', '2020-07-16', '2020-07-22', '2020-07-23', '2020-07-29', '2020-07-30', '2020-08-26', '2020-08-28', '2020-09-11', '2020-09-12', '2020-09-13', '2020-09-14', '2021-04-13', '2021-04-18', '2021-04-27', '2021-04-28', '2021-05-02', '2021-05-05', '2021-05-06', '2021-05-15', '2021-05-16', '2021-05-20', '2021-05-21', '2021-05-28', '2021-05-31', '2021-06-07', '2021-06-22', '2021-06-30', '2021-07-01', '2021-07-05', '2021-07-06', '2021-07-15', '2021-08-04', '2022-02-08']
na = na[~na.Date_reported.isin(low)] 

low=['2020-04-18', '2020-04-27', '2020-05-15', '2020-05-22', '2020-05-29', '2020-06-01', '2020-06-11', '2020-06-12', '2020-06-14', '2020-06-21', '2020-06-24', '2020-07-03', '2020-07-06', '2020-07-07', '2020-07-17', '2020-07-18', '2020-07-22', '2020-09-18', '2020-09-25', '2020-09-26', '2020-09-27', '2020-09-28', '2020-09-29', '2021-04-14', '2021-04-15', '2021-04-16', '2021-04-22', '2021-04-30', '2021-05-01', '2021-05-02', '2021-05-03', '2021-05-17', '2021-05-21', '2021-05-29', '2021-05-30', '2021-06-01', '2021-06-08', '2021-06-09', '2021-06-12', '2021-06-19', '2021-06-20', '2021-06-22', '2021-07-02', '2021-07-31', '2021-08-02', '2021-08-03', '2021-08-05']
sa = sa[~sa.Date_reported.isin(low)] 

low=['2020-01-12', '2020-01-13', '2020-01-15', '2020-01-16', '2020-01-22', '2020-02-28', '2020-03-11', '2020-03-19', '2020-03-28', '2020-04-03', '2020-04-06', '2020-04-19', '2020-04-24', '2020-04-27', '2020-04-28', '2020-04-30', '2020-05-02', '2020-05-03', '2020-05-07', '2020-05-08', '2020-05-11', '2020-05-15', '2020-05-17', '2020-05-19', '2020-05-21', '2020-05-24', '2020-06-03', '2020-06-04', '2020-06-06', '2020-06-08', '2020-06-14', '2020-06-15', '2020-06-19', '2020-06-23', '2020-06-24', '2020-06-27', '2020-06-28', '2020-06-30', '2020-07-08', '2020-07-23', '2020-07-26', '2020-07-30', '2020-08-02', '2020-08-05', '2020-08-06', '2020-08-09', '2020-08-17', '2020-08-20', '2020-09-12', '2020-09-18', '2020-09-27', '2020-10-05', '2020-10-08', '2020-10-21', '2020-10-28', '2020-11-04', '2020-11-23', '2020-11-25', '2020-12-13', '2021-01-13', '2021-04-16', '2021-04-17', '2021-04-18', '2021-04-22', '2021-04-24', '2021-05-01', '2021-05-03', '2021-05-06', '2021-05-07', '2021-05-09', '2021-05-12', '2021-05-19', '2021-05-20', '2021-05-21', '2021-05-24', '2021-05-25', '2021-06-01', '2021-06-03', '2021-06-04', '2021-06-06', '2021-06-12', '2021-06-13', '2021-06-22', '2021-06-23', '2021-06-26', '2021-06-27', '2021-06-29', '2021-07-02', '2021-07-05', '2021-07-07', '2021-07-08', '2021-07-12', '2021-07-13', '2021-07-17', '2021-07-22', '2021-08-02', '2021-08-04', '2021-08-05', '2021-08-07', '2021-08-08', '2021-08-09', '2021-08-10', '2021-08-12', '2021-08-15', '2021-08-17', '2021-08-19', '2021-08-20', '2021-08-21', '2021-08-22', '2021-08-26', '2021-08-28', '2021-08-30', '2021-09-03', '2021-09-05', '2021-09-08', '2021-09-09', '2021-09-10', '2021-09-11', '2021-09-12', '2021-09-13', '2021-09-14', '2021-09-16', '2021-09-20', '2021-09-22', '2021-09-23', '2021-09-25', '2021-09-28', '2021-09-29', '2021-11-03', '2021-11-12', '2021-12-02', '2021-12-15', '2021-12-17', '2022-01-05', '2022-01-06', '2022-01-14', '2022-01-16', '2022-01-24', '2022-02-02']
oc = oc[~oc.Date_reported.isin(low)] 

df_new_eu_mean = eu.groupby(eu['Date_reported'], as_index=False).mean()
df_new_af_mean = af.groupby(af['Date_reported'], as_index=False).mean()
df_new_as1_mean = as1.groupby(as1['Date_reported'], as_index=False).mean()
df_new_na_mean = na.groupby(na['Date_reported'], as_index=False).mean()
df_new_oc_mean = oc.groupby(oc['Date_reported'], as_index=False).mean()
df_new_sa_mean = sa.groupby(sa['Date_reported'], as_index=False).mean()

df_new_eu_med = eu.groupby(eu['Date_reported'], as_index=False).median()
df_new_af_med = af.groupby(af['Date_reported'], as_index=False).median()
df_new_as1_med = as1.groupby(as1['Date_reported'], as_index=False).median()
df_new_na_med = na.groupby(na['Date_reported'], as_index=False).median()
df_new_oc_med = oc.groupby(oc['Date_reported'], as_index=False).median()
df_new_sa_med = sa.groupby(sa['Date_reported'], as_index=False).median()


fig = plt.figure(figsize=(20,12), dpi=1000)

df_new_median.Date_reported = np.asarray(df_new_median.Date_reported, dtype='datetime64[s]')
 
plt.plot(df_new_median.Date_reported, df_new_median['percent_ch'], lw=6)

plt.xticks(['2020-01','2020-07','2021-01','2021-07','2022-01'],['2020-01','2020-07','2021-01','2021-07','2022-01'],fontsize=48)
plt.yticks(fontsize=48)
 
# Providing x and y label to the chart
plt.xlabel('Date',fontsize=48)
plt.ylabel('Median Percent Change',fontsize=48, x=0.1)
fig.savefig("global_median_percentchange_stl.pdf")


fig = plt.figure(figsize=(24,12), dpi=1000)

df_new_mean.Date_reported = np.asarray(df_new_mean.Date_reported, dtype='datetime64[s]')
 
plt.plot(df_new_mean.Date_reported, df_new_mean['rec'], lw=6)

plt.xticks(['2020-01','2020-07','2021-01','2021-07','2022-01'],['2020-01','2020-07','2021-01','2021-07','2022-01'],fontsize=48)
plt.yticks(fontsize=48)
 
# Providing x and y label to the chart
plt.xlabel('Date',fontsize=48)
plt.ylabel("Proportion of urban" "\n" "areas showing recovery",fontsize=48, x=0.1)
fig.savefig("global_recovery_stl.pdf")


import matplotlib.pyplot as plt
plt.rc('font', size=56)

df_new_eu_mean.Date_reported = np.asarray(df_new_eu_mean.Date_reported, dtype='datetime64[s]')
df_new_as1_mean.Date_reported = np.asarray(df_new_as1_mean.Date_reported, dtype='datetime64[s]')
df_new_af_mean.Date_reported = np.asarray(df_new_af_mean.Date_reported, dtype='datetime64[s]')
df_new_na_mean.Date_reported = np.asarray(df_new_na_mean.Date_reported, dtype='datetime64[s]')
df_new_sa_mean.Date_reported = np.asarray(df_new_sa_mean.Date_reported, dtype='datetime64[s]')
df_new_oc_mean.Date_reported = np.asarray(df_new_oc_mean.Date_reported, dtype='datetime64[s]')

fig, axs = plt.subplots(2, 3, figsize=(35,18), dpi=1000, sharex=True)
axs[0,0].plot(df_new_eu_mean.Date_reported, df_new_eu_mean.rec, lw=3)
axs[0,1].plot(df_new_as1_mean.Date_reported, df_new_as1_mean.rec, lw=3)
axs[0,2].plot(df_new_af_mean.Date_reported, df_new_af_mean.rec, lw=3)
axs[1,0].plot(df_new_na_mean.Date_reported, df_new_na_mean.rec, lw=3)
axs[1,1].plot(df_new_sa_mean.Date_reported, df_new_sa_mean.rec, lw=3)
axs[1,2].plot(df_new_oc_mean.Date_reported, df_new_oc_mean.rec, lw=3)
fig.supxlabel('Date', fontsize=56)
fig.supylabel("Proportion of urban" "\n" "areas showing recovery", x=0.01, fontsize=56)
start, end = axs[0,0].get_xlim()
axs[0,0].set_xticks(['2020','2021','2022'])
axs[0,1].set_xticks(['2020','2021','2022'])
axs[0,2].set_xticks(['2020','2021','2022'])
axs[1,0].set_xticks(['2020','2021','2022'])
axs[1,1].set_xticks(['2020','2021','2022'])
axs[1,2].set_xticks(['2020','2021','2022'])

axs[0,0].set_xticklabels(['2020','2021','2022'])
axs[0,1].set_xticklabels(['2020','2021','2022'])
axs[0,2].set_xticklabels(['2020','2021','2022'])
axs[1,0].set_xticklabels(['2020','2021','2022'])
axs[1,1].set_xticklabels(['2020','2021','2022'])
axs[1,2].set_xticklabels(['2020','2021','2022'])

axs[0,0].set_yticks([0,.3,.6])
axs[0,1].set_yticks([0.2,0.5,0.7])
axs[0,2].set_yticks([0.2,0.5,0.8])
axs[1,0].set_yticks([0.2,0.5,0.8])
axs[1,1].set_yticks([0.2,0.5,0.8])
axs[1,2].set_yticks([0,0.5,1])

axs[0,0].set_yticklabels(['0','.3','.6'])
axs[0,1].set_yticklabels([0.2,0.5,0.7])
axs[0,2].set_yticklabels([0.2,0.5,0.8])
axs[1,0].set_yticklabels([0.2,0.5,0.8])
axs[1,1].set_yticklabels([0.2,0.5,0.8])
axs[1,2].set_yticklabels([0,0.5,1])

plt.tight_layout()
fig.savefig("continental_proportion_recovery_row1_europe_asia_africa_row2_na_sa_oc_stl.pdf")


df_new_eu_mean.to_csv("eu_avg_rec_stl.csv")
df_new_af_mean.to_csv("af_avg_rec_stl.csv")
df_new_as1_mean.to_csv("as1_avg_rec_stl.csv")
df_new_na_mean.to_csv("na_avg_rec_stl.csv")
df_new_oc_mean.to_csv("oc_avg_rec_stl.csv")
df_new_sa_mean.to_csv("sa_avg_rec_stl.csv")

df_new_eu_med.to_csv("eu_med_percentch_stl.csv")
df_new_af_med.to_csv("af_med_percentch_stl.csv")
df_new_as1_med.to_csv("as1_med_percentch_stl.csv")
df_new_na_med.to_csv("na_med_percentch_stl.csv")
df_new_oc_med.to_csv("oc_med_percentch_stl.csv")
df_new_sa_med.to_csv("sa_med_percentch_stl.csv")