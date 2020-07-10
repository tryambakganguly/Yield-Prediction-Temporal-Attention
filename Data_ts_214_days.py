import sqlite3
import numpy as np
import pandas as pd
import os

rawdata_dir="Data"

data_dir="Data_Reframed"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Opening the txt file
data=open("%s/10252017AllPlotsWithWeather.txt" %(rawdata_dir),"r")
data=data.readlines()[1:103366]  # Read lines till 103365


reframed_1=np.zeros((103365, 3)) 
reframed_2=np.zeros((103365, 214, 8)) 

genotype_list=list() # Genotype
county_list=list() # City / County
state_list=list()  # State / Province


sqlite_file = 'soy_yield.sqlite'  #connect to database
conn=sqlite3.connect (sqlite_file)
c=conn.cursor()
c.execute("DROP TABLE IF EXISTS yield")
c.execute("CREATE table yield (id INTEGER NOT NULL PRIMARY KEY, dat_time INTEGER NOT NULL UNIQUE, Gentype TEXT, GenID float, location INTEGER, YIELD FLOAT)")


count=-1
for i in data:
    count=count+1
    da=i.strip().split(" ")
    c.execute("INSERT OR IGNORE INTO 'yield' ('dat_time', 'Gentype', 'GenID', 'location', 'YIELD')VALUES(?,?,?,?,?)", (da[5],da[3],da[15],da[8],da[2]))

   
       
    genotype=da[3] #Genotype
    genotype_list.append(genotype)
    
    county=da[13] # City / County
    county_list.append(county)
    
    state=da[14] # State / Province 
    state_list.append(state)
    
    reframed_1[count,0]=da[2]   #Yield
    reframed_1[count,1]=da[5]   #Year
    reframed_1[count,2]=da[8]   #Location
    
    reframed_2[count,:,0]=da[4] # MG
    reframed_2[count,:,1]=da[19:233] #Average Direct Normal Irradiance 
    reframed_2[count,:,2]=da[233:447] #Average Precipitation Previous Hour (inches)
    reframed_2[count,:,3]=da[447:661] #Average Relative Humidity 
    reframed_2[count,:,4]=da[875:1089] #Maximum Direct Normal Irradiance 
    reframed_2[count,:,5]=da[1517:1731] #Maximum Surface Temperature 
    reframed_2[count,:,6]=da[1945:2159] #Minimum Surface Temperature 
    
    # Index(873) of Avg. Surface Temp is NA :- Doesnt matter when we are doing weekly (Taking first 210 from 214 values)
    da[873]= (float(da[872]) + float(da[874]))/2 #Filling the NA value     
    reframed_2[count,:,7]=da[661:875] #Average Surface Temperature (Fahrenheit)  
    
    
    
    #print(da[2])
    #print(da[19])
    print(count)
conn.commit()
print('Completed')
c.close()
conn.close()


#Save Files 
genotype_list = pd.DataFrame(genotype_list).to_csv('%s/genotype_list_all_data.csv'%(data_dir))

county_list = pd.DataFrame(county_list).to_csv('%s/county_list_all_data.csv'%(data_dir))

state_list = pd.DataFrame(state_list).to_csv('%s/state_list_all_data.csv'%(data_dir))


np.save("%s/yield_year_location_all_data"%(data_dir), reframed_1)
np.save("%s/mg_weather_variables_all_data_TS_214_days"%(data_dir), reframed_2)

print("Saved files to disk")