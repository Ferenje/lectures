"""
Created on Tue Nov  24 14:5:10 2021

@author: Tobias Weirowski
"""

import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import math
import datetime as dt
import random
import os
import csv
import seaborn as sb
import statsmodels.api as sm
import datetime

mac_dir = "/Users/tobiasweirowski/Library/Mobile Documents/com~apple~CloudDocs/LS-EmpMakro/Vorlesungen/01 AIE"
win_dir = "C:/Users/Tobias Weirowski/iCloudDrive/LS-EmpMakro/Vorlesungen/01 AIE"
cur_dir = mac_dir

''' FOREX '''


'''0: Functions'''


def convert(list):
    '''Converts a list to a tuple'''
    return tuple(list)

gentimeSerM(startyear, startmonth,  endyear, endmonth)
def gentimeSerQ(ystart, yend):
    startyear = int(ystart[:4])
    startquarter = int(ystart[5:7])
    endyear = int(yend[:4])
    endquarter = int(yend[5:7])
    ls = [ystart]
    if endyear - startyear >= 0:
        u = 4 - startquarter
        for i in range(1, u+1):
            string = str(startyear) + "Q" + str(startquarter+i)
            ls.append(string)
        for year in range(startyear+1 , endyear):
            for q in range(1, 5):
                string2 = str(year)  + "Q"+ str(q)
                ls.append(string2)
        for j in range(1, endquarter+1):
            string3 = str(endyear) + "Q" + str(j)
            ls.append(string3)
    else:
        print("End date before start date")
        return ls


def timeSerCountry(Dataframe, country_code, series_code, ystart, yend):
    ''' Get list of: country and subject in range of years (ystart, yend). Enter arguments: country, subject, ystart, yend as strings'''
    df = Dataframe
    varnames = list(df.columns.values)
    df_country = df[df.Country_Code == country_code]
    df_country_series = df_country[df_country.Series_Code == series_code]
    aa = df_country_series.iloc[0, varnames.index(ystart): varnames.index(yend) + 1]
    y = (aa)
    #x = list(range(int(ystart), int(yend) + 1, 1))
    x = varnames[varnames.index(ystart):varnames.index(yend) + 1]
    y = [float(ele) for ele in y]
    return x, y

def plottimeSerCountry(Dataframe, country_code, series_code, ystart, yend):
    ''' Plot the values of a subject for a country in range of years (ystart, yend). Enter arguments as strings'''
    x, y = timeSerCountry(Dataframe, country_code, series_code, ystart, yend)
    plt.plot(x, y)
    plt.axhline(0, color='k', linestyle='solid')
    plt.xlabel('year')
    plt.xticks(rotation=90)
    plt.ylabel('values')
    plt.legend(loc="upper left", )
    plt.title(series_code + ' ' + dictCountry[country_code] + ' ' + ystart + '-' + yend)
    plt.show()


'''A: Data read '''
# Data set from https://www.imf.org/external/np/fin/ert/GUI/Pages/ReportOptions.aspx

dfERIMF = pd.read_excel(cur_dir + "/FOREX/IMF_Exchange_Rate_Report.xls", header = 2)
dfERIMF = dfERIMF.loc[1:7127, :]
dfERIMF['Date'] = dfERIMF['Date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d')) # changes display (w/o hours and minutes)

# spaces in header to be removed
dfERIMF.columns = [c.replace('   (', '_(') for c in dfERIMF.columns]
dfERIMF.columns = [c.replace(') ', ')#') for c in dfERIMF.columns]
dfERIMF.columns = [c.replace(' ', '_') for c in dfERIMF.columns]
header = list(dfERIMF.columns.values)

newheader= []
for ele in header:
    a_string = ele
    split_string = a_string.split("#", 1)
    newheader.append(split_string[0])

dfERIMF.columns= newheader

del header,  a_string , split_string, newheader, ele

# Data set  for Turkey IFS - https://data.imf.org/?sk=4c514d48-b6ba-49ed-8ab9-52b0c1a0179b&sId=1409151240976
#dfERIMFTL = pd.read_excel(cur_dir + "/FOREX/Turkey_Exchange_Rates_incl_Effective.xls", sheet_name= "Monthly")
dfERIMFTL = pd.read_excel(cur_dir + "/FOREX/Turkey_selection_IFS.xlsx", sheet_name= "Turkey")

ls = ["Labor", "TLEUR" , "TLEURa", "TLUSD", "TLUSDa"]
dfERIMFTL["Series_Code"]= ls

del ls
dfERIMFTL.columns = [c.replace(' ', '_') for c in dfERIMFTL.columns] # Space --> _

def timeSerCountryTUR01(Dataframe, country_code, series_code, ystart, yend):
    ''' Get list of: country and subject in range of years (ystart, yend). Enter arguments: country, subject, ystart, yend as strings'''
    df = Dataframe
    varnames = list(df.columns.values)
    #df_country = df[df.Country_Code == country_code]
    df_series = df[df.Series_Code == series_code]
    aa = df_series.iloc[0, varnames.index(ystart): varnames.index(yend) + 1]
    y = (aa)
    #x = list(range(int(ystart), int(yend) + 1, 1))
    x = varnames[varnames.index(ystart):varnames.index(yend) + 1]
    y = [float(ele) for ele in y]
    return x, y

def plottimeSerCountryTUR01(Dataframe, country_code, series_code, ystart, yend):
    ''' Plot the values of a subject for a country in range of years (ystart, yend). Enter arguments as strings'''
    x, y = timeSerCountryTUR01(Dataframe, country_code, series_code, ystart, yend)
    plt.plot(x, y)
    plt.axhline(0, color='k', linestyle='solid')
    plt.xlabel('year')
    plt.ylabel('values')
    plt.legend(loc="upper left", )
    plt.title(series_code + ' ' + dictCountry[country_code] + ' ' + ystart + '-' + yend)
    plt.show()

plottimeSerCountryTUR01(dfERIMFTL, "TUR", "TLUSDa", "Jan_2010", "Oct_2021")
plottimeSerCountryTUR01(dfERIMFTL, "TUR", "TLUSDa", "Jan_2010", "Oct_2015")
plottimeSerCountryTUR01(dfERIMFTL, "TUR", "TLUSDa", "Nov_2015", "Oct_2021")


# Data set from https://databank.worldbank.org/source/global-economic-monitor-(gem)

dfER2 = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-all20211116_Data-02.csv", sep=",", na_values= ['..'], error_bad_lines=False)
dfER1 = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-all20211116_Data-01.csv", sep=",", na_values= ['..'], error_bad_lines=False)
# splicing
ls = list(dfER1.columns.values)
ls2 = list(dfER2.columns.values)
ls1 = ls[4:]

for ele in ls1:
    val = dfER1[ele]
    dfER2 = dfER2.join(val)

dfER= dfER2

del ls, ls1, ls2, dfER1, dfER2

dfER2 = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-all20211116_Data-02.csv", sep=",", na_values= ['..'], error_bad_lines=False)
dfER1 = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-all20211116_Data-01.csv", sep=",", na_values= ['..'], error_bad_lines=False)
# splicing
ls = list(dfER1.columns.values)
ls2 = list(dfER2.columns.values)
ls1 = ls[4:]

for ele in ls1:
    val = dfER1[ele]
    dfER2 = dfER2.join(val)

dfERY= dfER2

del ls, ls1, ls2, dfER1, dfER2

dfER2 = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-all20211116_Data-02.csv", sep=",", na_values= ['..'], error_bad_lines=False)
dfER1 = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-all20211116_Data-01.csv", sep=",", na_values= ['..'], error_bad_lines=False)
# splicing
ls = list(dfER1.columns.values)
ls2 = list(dfER2.columns.values)
ls1 = ls[4:]

for ele in ls1:
    val = dfER1[ele]
    dfER2 = dfER2.join(val)

dfERQ= dfER2

del ls, ls1, ls2, dfER1, dfER2

dfER2 = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-all20211116_Data-02.csv", sep=",", na_values= ['..'], error_bad_lines=False)
dfER1 = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-all20211116_Data-01.csv", sep=",", na_values= ['..'], error_bad_lines=False)
# splicing
ls = list(dfER1.columns.values)
ls2 = list(dfER2.columns.values)
ls1 = ls[4:]

for ele in ls1:
    val = dfER1[ele]
    dfER2 = dfER2.join(val)

dfERM= dfER2

del ls, ls1, ls2, dfER1, dfER2



#dfER = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-FOREX-CPI_Data.csv", sep=",", na_values= ['..'], error_bad_lines=False)
#dfERY = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-FOREX-CPI_Data.csv", sep=",", na_values= ['..'], error_bad_lines=False)
#dfERQ = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-FOREX-CPI_Data.csv", sep=",", na_values= ['..'], error_bad_lines=False)
#dfERM = pd.read_csv(cur_dir + "/FOREX/GlobalEconomicMonitor-FOREX-CPI_Data.csv", sep=",", na_values= ['..'], error_bad_lines=False)

dfER.columns = [c.replace(' ', '_') for c in dfER.columns] # Space --> _
dfERY.columns = [c.replace(' ', '_') for c in dfER.columns] # Space --> _
dfERQ.columns = [c.replace(' ', '_') for c in dfER.columns] # Space --> _
dfERM.columns = [c.replace(' ', '_') for c in dfER.columns] # Space --> _
varnamesER = list(dfER.columns.values)

lsM=[] # column names monthly values
for i in range(3, len(varnamesER)):
    if "M" in varnamesER[i]:
        lsM.append(varnamesER[i])

lsQ=[] # column names quarterly values
for i in range(3, len(varnamesER)):
    if "Q" in varnamesER[i]:
        lsQ.append(varnamesER[i])
lsQM = lsQ + lsM

for ele in lsQM: # clean dataframe from monthly and Quarterly data
    del dfERY[ele]

varnamesERY = list(dfERY.columns.values)
lsY = varnamesERY[4:] # Column names of yearly values
lsYM  = lsY + lsM
lsYQ  = lsY + lsQ

for ele in lsYM: # clean dataframe from monthly and yearly data
        del dfERQ[ele]

for ele in lsYQ: # clean dataframe from yearly and Quarterly data
        del dfERM[ele]

# cleaning the header, e.g. 1987_[1987] --> 1987
header = list(dfERM.columns.values)
newheader= []
for ele in header:
    a_string = ele
    split_string = a_string.split("_[", 1)
    newheader.append(split_string[0])
dfERM.columns= newheader

header = list(dfERQ.columns.values)
newheader= []
for ele in header:
    a_string = ele
    split_string = a_string.split("_[", 1)
    newheader.append(split_string[0])
dfERQ.columns= newheader

header = list(dfERY.columns.values)
newheader= []
for ele in header:
    a_string = ele
    split_string = a_string.split("_[", 1)
    newheader.append(split_string[0])
dfERY.columns= newheader

del  lsQ,  lsM, lsY, lsQM, lsYM, lsYQ, header, a_string, newheader

country_names = list(pd.unique(dfERY.Country))
country_code = list(pd.unique(dfERY.Country_Code))
Series_code = list(pd.unique(dfERY.Series_Code))
Series_desc = list(pd.unique(dfERY.Series))

dictCountryCode = dict(zip(country_names, country_code))  # Which country_code for country name
dictCountry = dict(zip(country_code, country_names,))  # Which country for country_code
dictSeries = dict(zip(Series_code, Series_desc))  # What does the Series code stands for

del country_code, country_names, Series_code, Series_desc

print(dfER.shape)
print(dfERY.shape)
print(dfERQ.shape)
print(dfERM.shape)

plottimeSerCountry(dfERM, "AUS", "NEER", "2011M01", "2012M12")

plottimeSerCountry(dfERM, "TUR", "IMPCOV", "2015M01", "2020M07")
plottimeSerCountry(dfERM, "TUR", "NEER", "2015M01", "2020M07")
plottimeSerCountry(dfERQ, "TUR", "IMPCOV", "2015Q1", "2020Q2")
timeSerCountry(dfERM, "DZA", "DPANUSLCU", "1987M01", "1990M12")
timeSerCountry(dfERM, "DZA", "DPANUSLCU", "2000M01", "2020M07")
plottimeSerCountry(dfERY, "TUR", "DMGSRMRCHSACD", "2015", "2019")
plottimeSerCountry(dfERY, "DEU", "DMGSRMRCHSACD", "2015", "2019")
plottimeSerCountry(dfERY, "FRA", "DMGSRMRCHSACD", "2015", "2019")

plottimeSerCountry(dfERQ, "TUR", "DMGSRMRCHSACD", "2015Q1", "2019Q4")
plottimeSerCountry(dfERQ, "DEU", "DMGSRMRCHSACD", "2015Q1", "2019Q4")
plottimeSerCountry(dfERQ, "FRA", "DMGSRMRCHSACD", "2015Q1", "2019Q4")

plottimeSerCountry(dfERQ, "TUR", "DXGSRMRCHSACD", "2015Q1", "2019Q4")
plottimeSerCountry(dfERQ, "TUR", "DMGSRMRCHSACD", "2015Q1", "2019Q4")
plottimeSerCountry(dfERQ, "TUR", "CPTOTNSXN", "2015Q1", "2019Q4")
plottimeSerCountry(dfERQ, "TUR", "NYGDPMKTPSAKD", "2015Q1", "2019Q4")
plottimeSerCountry(dfERQ, "TUR", "NYGDPMKTPSACD", "2015Q1", "2019Q4")
plottimeSerCountry(dfERM, "TUR", "TOT", "2015M01", "2020M07")

plottimeSerCountry(dfERM, "TUR", "DPANUSLCU", "2015M01", "2020M07")
plottimeSerCountry(dfERM, "TUR", "TOTRESV", "2015M01", "2020M07")
plottimeSerCountry(dfERM, "TUR", "CPTOTSAXN", "2015M01", "2020M07")
plottimeSerCountry(dfERM, "TUR", "CPTOTNSXN", "2015M01", "2020M07")


plottimeSerCountry(dfERY, "TUR", "CPTOTNSXN", "2015", "2019")
plottimeSerCountry(dfERM, "TUR", "CPTOTNSXN", "2015M01", "2019M12")
plottimeSerCountry(dfERM, "TUR", "CPTOTNSXN", "2015M01", "2020M07")

plottimeSerCountry(dfERQ, "DEU", "DXGSRMRCHSACD", "2015Q1", "2019Q4")
plottimeSerCountry(dfERQ, "FRA", "DXGSRMRCHSACD", "2015Q1", "2019Q4")

x, y = timeSerCountry(Dataframe, country_code, series_code, ystart, yend)

# Export/import (merchandize) ratio Turkey
x1, y1 = timeSerCountry(dfERQ, "TUR", "DXGSRMRCHSACD", "2015Q1", "2019Q4")
x2, y2 = timeSerCountry(dfERQ, "TUR", "DMGSRMRCHSACD", "2015Q1", "2019Q4")
y3= [y1/y2 for y1,y2 in zip(y1,y2)]


plt.plot(x1, y1)
plt.axhline(0, color='k', linestyle='solid')
plt.xlabel('year')
plt.xticks(rotation=90)
plt.ylabel('values')
plt.legend(loc="upper left", )
plt.title("VAR"+ ' ' + "Country" + ' ' + "start" + '-' + "end")
plt.show()

country_sing = list(pd.unique(dfGS.sanctioning_state))  # list of sanctioning countries
country_sed = list(pd.unique(dfGS.sanctioned_state))  # list of sanctioned countries

# Datensatz WEO einlesen
df = pd.read_csv(cur_dir + "/WEOOct2020countries2.csv", sep=";", encoding='latin-1', error_bad_lines=False)
#df = pd.read_csv(cur_dir + "LS-EmpMakro/Vorlesungen/01 AIE/WEOOct2020countries3.csv", sep=";", encoding='latin-1', error_bad_lines=False) # anders als in WEOOct2020countries2.csv sind "," als Tausendertrennzeichen entfernt
df.columns = [c.replace(' ', '_') for c in df.columns]
varnames = list(df.columns.values)
country_names = list(pd.unique(df.Country))

#### Dictionaries for WEO Dataset on Countrylevel
country_iso = list(pd.unique(df.ISO))
WEOSubjectCode = list(pd.unique(df.WEO_Subject_Code))
df_deu = df[df.ISO == "DEU"]
WEOSubjectDesc = list(
    df_deu.Subject_Descriptor)  # Subject_Decriptor ist oft identisch, daher keine pd.unique() und nur für ein Land
del WEOSubjectCode[-1]  # last item was "nan", so has to be delated
WEOSubjectNote = list(df_deu.Subject_Notes)

dictionaryISO = dict(zip(country_names, country_iso))  # Welche ISOverbirgt sich hinter  Land
dictionarySubjects = dict(zip(WEOSubjectCode, WEOSubjectDesc))  # Welches Subject verbirgt sich hinter SubjectCode
dictionarySubjectNote = dict(zip(WEOSubjectCode, WEOSubjectNote))  # Bemerkung zum Subject hinter SubjectCode
dictionaryCountryIso = dict(zip(country_iso, country_names))  # Welches Land verbirgt sich hinter ISO

del df_deu , country_names , varnames

def timeSerCountryWeo(country, subject, ystart, yend):
    ''' Get list of: country and subject in range of years (ystart, yend). Enter arguments as strings'''
    varnames = list(df.columns.values)
    df_country = df[df.ISO == country]
    df_country_subject = df_country[df_country.WEO_Subject_Code == subject]
    aa = df_country_subject.iloc[0, varnames.index(ystart): varnames.index(yend) + 1]
    y = (aa)
    x = list(range(int(ystart), int(yend) + 1, 1))
    y = [float(ele) for ele in y]
    return x, y
def plottimeSerCountryWeo(country, subject, ystart, yend):
    ''' Plot the values of a subject for a country in range of years (ystart, yend). Enter arguments as strings'''
    x, y = timeSerCountryWeo(country, subject, ystart, yend)
    plt.plot(x, y)
    plt.axhline(0, color='k', linestyle='solid')
    plt.xlabel('year')
    plt.xticks(rotation=90)
    plt.ylabel('values')
    plt.legend(loc="upper left", )
    plt.title(subject + ' ' + country + ' ' + ystart + '-' + yend)
    plt.show()

plottimeSerCountryWeo("TUR", "GGXCNL_NGDP", "2015", "2020")
plottimeSerCountryWeo("TUR", "GGXWDG_NGDP", "2015", "2020")
plottimeSerCountryWeo("TUR", "GGXWDN_NGDP", "2015", "2020")

'''some calculations
Problem: time dimension is in row, not column'''
# Example TOTRESV = total reserves
# build a data frame with the variables columnwise

ls = ['TUR', 'FRA', 'ITA', 'ESP' ,'GRC',  "PRT" ] # List of countries to analyse/relate
df_calc = pd.DataFrame(columns=[ls])
ls_res = ['TUR_res', 'FRA_res', 'ITA_res', 'ESP_res' ,'GRC_res',  "PRT_res" ] # List of countries to analyse/relate

for ele in ls:
    x,y = timeSerCountry(dfERM, ele, "TOTRESV", "2015M01", "2019M12")
    df_calc[ele]=y

df_calc.columns= ls_res
# how to plot the changes
df_calc_change = df_calc.pct_change(fill_method ='ffill') # calculate the percentage changes to previews row; option is for nan
df_calc_change.plot( y=['TUR_res', 'FRA_res', 'ITA_res', 'ESP_res' ,'GRC_res',  "PRT_res"])
plt.show()

for ele in ls_res:
    df_calc[ele + "_c"]=""
    df_calc[ele + "_c"][0] = np.float64(100)
    for i in range(1, len(df_calc[ele])):
        c= (df_calc[ele][i]-df_calc[ele][0])/df_calc[ele][0]
        df_calc[ele + "_c"][i]= 100+c*100


def SeriesChangesBase(dataf, basis, shift, annex ):
    '''calculates the relative change to a basis value.
    datdf is a pandas dataframe of all series to be caclulated;
    basis usually 100; shift parameter to shift along y-axis
    annex names the cal variable as "varname"+"annex"'''
    ls1 = list(dataf.columns.values)
    ls_var=[]
    for j in range(len(ls1)):
        ls_var.append(''.join(ls1[j]))
    print(ls_var)
    df =  dataf
    ls_var_annex = []
    for ele in ls_var:
        df[ele + annex] = ""
        df.at[0, ele + annex] = np.float64(basis)
        #df[ele + annex][0] = np.float64(basis)
        for i in range(1, len(df[ele])):
            c = (df[ele].values[i]-df[ele].values[0])/df[ele].values[0]
            df.at[i, ele + annex]= basis + c * basis - shift
    return df

df_test78 = SeriesChangesBase(df_calc, 1, 0, "_cal" )

df_test78.plot( y=['TUR_cal', 'FRA_cal', 'ITA_cal', 'ESP_cal' ,'GRC_cal',  "PRT_cal"])
plt.show()

df_calc.plot( y=['TUR_res', 'FRA_res', 'ITA_res', 'ESP_res' ,'GRC_res',  "PRT_res"])
plt.show()

df_calc.plot( y=['TUR_res_c', 'FRA_res_c', 'ITA_res_c', 'ESP_res_c' ,'GRC_res_c',  "PRT_res_c"])
plt.show()
# Summary stats of all numeric columns
df_calc.describe()

# Summary stats on precip column as dataframe
df_calc[["TUR"]].describe()

for i in df_calc["TUR_res_c"]:
    print(type(i))


'''WEO Data'''
#df.to_csv(cur_dir + '/Exercise/WEO_Exercise.csv')
#subjectsWEOExer = 'BCA_NGDPD', 'DS_NGDPD', 'GGXONLB_NGDP', 'GGXWDG_NGDP', 'NGSD_NGDP', 'PCPI', 'NGDPRPPPPC'
subjectsWEOExer = 'BCA_NGDPD', 'GGXONLB_NGDP', 'GGXWDG_NGDP', 'NGSD_NGDP', 'PCPI', 'NGDPRPPPPC'
for ele in subjectsWEOExer:
    print(dictionarySubjects[ele])
# List of countries
countryList = 'DEU', 'ESP', 'PRT', 'GRC', 'FRA', 'USA' # List of countries to analyse/relate
column_names=[]
for ele in countryList:
    column_names.append(ele + '-'  + 'BCA_NGDPD')

df_exer = pd.DataFrame(columns= ['DEU', 'ESP', 'PRT', 'GRC', 'FRA', 'USA' ])
df_exer = pd.DataFrame(columns= countryList)

for ele in countryList:
    x,y = timeSerCountryWeo(ele, "BCA_NGDPD", "2000", "2020")
    df_exer[ele]=y
df_exer.columns = column_names

df_exer['year'] = x
df_exer.set_index('year')

df_exer_c = SeriesChangesBase(df_exer, 100, 0, "_c" )
column_names_c=[]
for ele in column_names:
    column_names_c.append(ele + '_c')

dictionaryLabelsPlt = dict(zip(column_names_c, countryList))

df_exer_c.plot( y=column_names_c , x= 'year' , label=countryList)
plt.axhline(100, color='k', linestyle='solid')
plt.xlabel('year')
#plt.xticks(rotation=45)
plt.xticks( np.arange(min(x), max(x)+1, 2.0), rotation=45)
plt.ylabel('values')
plt.legend(loc="lower left", )
plt.title('BCA_NGDPD ' + '2000-' + '2020; ' + 'freq.: years' + ' rela. changes (2000=100)')
plt.show()

df_exer_c.plot( y=column_names, x= 'year' , label=countryList)
plt.axhline(0, color='k', linestyle='solid')
plt.xlabel('year')
#plt.xticks(rotation=45)
plt.xticks( np.arange(min(x), max(x)+1, 2.0), rotation=45)
plt.ylabel('values')
plt.legend(loc="lower left", )
plt.title('BCA_NGDPD ' + '2000-' + '2020; ' + 'freq.: years')
plt.show()

del df, df_exer
'''Trade: structural gravity 
from https://www.wto.org/english/res_e/reser_e/structural_gravity_e.htm
'''
import pyreadstat
directoryGr = cur_dir + "/Gravity-Trade/Structural.gravity.manufacturing.database.Ver1.dta"
dfgravity, meta = pyreadstat.read_dta(directoryGr)
dfgravity.sort_values(by=['year','exporter'], inplace=True)


'''Trade: structural gravity - From An Advanced Guide to Trade Policy Analysis: The Structural Gravity Model
https://vi.unctad.org/tpa/web/vol2/vol2home.html
'''

'''Excercise 1. Estimating the effects of WTO accession '''

''' (i) Histogram '''
import pyreadstat
directoryGr = cur_dir + "/Gravity-Trade/Chapter1/Datasets/Chapter1Exercise1.dta"
dfgravity, meta = pyreadstat.read_dta(directoryGr)

#extract WTO accession
acc = dfgravity['WTO']
acc.hist()
plt.show()

'''(ii) Benchmark gravity estimation '''
''' 
a. Generate exporter-time and importer-time
* Create exporter-time fixed effects
		egen exp_time = group(exporter year)
		quietly tabulate exp_time, gen(EXPORTER_TIME_FE)

	* Create importer-time fixed effects
		egen imp_time = group(importer year)
		quietly tabulate imp_time, gen(IMPORTER_TIME_FE)
'''
import econtools

exp_time = econtools.group_id(dfgravity, cols=['exporter' , 'year'])
imp_time= econtools.group_id(dfgravity, cols=['importer' , 'year'])

dfgravity_dummies = pd.get_dummies(dfgravity, prefix='', prefix_sep='',
                            columns=['exporter', 'year'])
dfgravity_dummies2 = pd.get_dummies(dfgravity, prefix='', prefix_sep='',
                            columns=['importer', 'year'])


'''
b. Estimate the following standard gravity specification by considering only 
*    international trade flows (i.e. for i=/j ) and applying the OLS estimator:
*    Xij = pi_it + xsi_jt + b*GRAVITY + b5*RTA + b6*WTO
* Create the symmetric pair id
		egen pair_id = group(DIST)

* Alternatively create asymmetric pair id
	*	egen pair_id = group(exporter importer)
		
* Create the log of distance
		gen ln_DIST = ln(DIST)
	
* Estimate the gravity model with the OLS estimator and store the results
 		regress trade EXPORTER_TIME_FE* IMPORTER_TIME_FE* ln_DIST CNTG LANG CLNY RTA WTO if exporter != importer, cluster(pair_id)
			estimates store ols
'''
pair_id = econtools.group_id(dfgravity, cols=['DIST'])
# Alternative
#pair_id = econtools.group_id(dfgravity, cols=['exporter', 'importer'])
ln_DIST  = np.log(dfgravity['DIST'])
