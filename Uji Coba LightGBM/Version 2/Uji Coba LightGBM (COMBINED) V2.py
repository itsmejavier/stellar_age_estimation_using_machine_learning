#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install LightGBM
get_ipython().system('pip install lightgbm')


# In[2]:


# coding: utf-8
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb


# In[3]:


print('Data Gaia yang Berisi 626 Ribu Bintang (Data Release 3) V2')
# load or create your dataset
data = pd.read_csv('C:\\Users\\ACER\\OneDrive\\Documents\\Gaia626kStarsDR3V2\\dataGaia2.csv')
df = pd.DataFrame(data)
df


# In[4]:


df.info() # Tampilkan info data.


# In[5]:


logg = data['logg'].to_list()
Fe_H = data['[Fe/H]'].to_list()

PQSO = data['PQSO'].to_list()
PGal = data['PGal'].to_list()
Pstar = data['Pstar'].to_list()
PWD = data['PWD'].to_list()
Pbin = data['Pbin'].to_list()

Teff = data['Teff'].to_list()
Evol = data['Evol'].to_list()

Rad = data['Rad-Flame'].to_list()
Lum = data['Lum-Flame'].to_list()
Mass = data['Mass-Flame'].to_list()
Age = data['Age-Flame'].to_list()
Spectral_Type = data['SpType-ELS'].to_list()


# In[6]:


data_new = {'logg':logg,'Metallicity':Fe_H,'Teff':Teff,'Rad':Rad,'Lum':Lum,'Mass':Mass,'Age':Age,'SpectralType':Spectral_Type,
            'Evol':Evol,'PQSO':PQSO,'PGal':PGal,'Pstar':Pstar,'PWD':PWD,'Pbin':Pbin}
df_new = pd.DataFrame(data_new)
df_new


# In[7]:


# Cek apakah terdapat angka infinit atau tidak.
print()
print("Checking for NaN values")

# Tampilkan kalimat "printing the count of infinity values".
print()
print("Printing the count of NaN values")

# Hitung ada berapa angka yang memiliki nilai kosong pada baris logg.
check_logg_nan = np.isnan(df_new['logg']).values.sum()
print("logg contains " + str(check_logg_nan) + " NaN values.")

# Hitung ada berapa angka yang memiliki nilai kosong pada baris [Fe/H].
check_feh_nan = np.isnan(df_new['Metallicity']).values.sum()
print("[Fe/H] contains " + str(check_feh_nan) + " NaN values.")

# Hitung ada berapa angka yang memiliki nilai kosong pada baris Teff.
check_teff_nan = np.isnan(df_new['Teff']).values.sum()
print("Teff contains " + str(check_teff_nan) + " NaN values.")

# Hitung ada berapa angka yang memiliki nilai kosong pada baris Rad.
check_rad_nan = np.isnan(df_new['Rad']).values.sum()
print("Rad contains " + str(check_rad_nan) + " NaN values.")

# Hitung ada berapa angka yang memiliki nilai kosong pada baris Lum.
check_lum_nan = np.isnan(df_new['Lum']).values.sum()
print("Lum contains " + str(check_lum_nan) + " NaN values.")

# Hitung ada berapa angka yang memiliki nilai kosong pada baris Mass.
check_mass_nan = np.isnan(df_new['Mass']).values.sum()
print("Mass contains " + str(check_mass_nan) + " NaN values.")

# Hitung ada berapa angka yang memiliki nilai kosong pada baris Age.
check_age_nan = np.isnan(df_new['Age']).values.sum()
print("Age contains " + str(check_age_nan) + " NaN values.")

# Hitung ada berapa angka yang memiliki nilai kosong pada baris Evol.
check_Evol_nan = np.isnan(df_new['Evol']).values.sum()
print("Evol contains " + str(check_Evol_nan) + " NaN values.")


# In[8]:


# Cek apakah terdapat angka infinit atau tidak.
print()
print("Checking for infinite values")

# Tampilkan kalimat "printing the count of infinity values".
print()
print("Printing the count of infinite values")

# Hitung ada berapa angka yang memiliki nilai infinit pada baris logg.
check_logg_inf = np.isinf(df_new['logg']).values.sum()
print("logg contains " + str(check_logg_inf) + " infinite values.")

# Hitung ada berapa angka yang memiliki nilai infinit pada baris [Fe/H].
check_feh_inf = np.isinf(df_new['Metallicity']).values.sum()
print("[Fe/H] contains " + str(check_feh_inf) + " infinite values.")

# Hitung ada berapa angka yang memiliki nilai infinit pada baris Teff.
check_teff_inf = np.isinf(df_new['Teff']).values.sum()
print("Teff contains " + str(check_teff_inf) + " infinite values.")

# Hitung ada berapa angka yang memiliki nilai infinit pada baris Rad.
check_rad_inf = np.isinf(df_new['Rad']).values.sum()
print("Rad contains " + str(check_rad_inf) + " infinite values.")

# Hitung ada berapa angka yang memiliki nilai infinit pada baris Lum.
check_lum_inf = np.isinf(df_new['Lum']).values.sum()
print("Lum contains " + str(check_lum_inf) + " infinite values.")

# Hitung ada berapa angka yang memiliki nilai infinit pada baris Mass.
check_mass_inf = np.isinf(df_new['Mass']).values.sum()
print("Mass contains " + str(check_mass_inf) + " infinite values.")

# Hitung ada berapa angka yang memiliki nilai infinit pada baris Age.
check_age_inf = np.isinf(df_new['Age']).values.sum()
print("Age contains " + str(check_age_inf) + " infinite values.")

# Hitung ada berapa angka yang memiliki nilai infinit pada baris Evol.
check_Evol_inf = np.isinf(df_new['Evol']).values.sum()
print("Evol contains " + str(check_Evol_inf) + " infinite values.")


# In[9]:


data_fix = {'logg':logg,'Metallicity':Fe_H,'Teff':Teff,'Rad':Rad,'Lum':Lum,'Mass':Mass,'Age':Age,'SpectralType':Spectral_Type,
            'Evol':Evol,'PQSO':PQSO,'PGal':PGal,'Pstar':Pstar,'PWD':PWD,'Pbin':Pbin}
df_fix = pd.DataFrame(data_fix)

# Drop baris yang memiliki nilai 'NaN'.
df_fix.dropna(inplace=True)

df_fix = df_fix.reset_index(drop=True)

# Tampilkan data.
df_fix


# In[10]:


# Buat histogram logaritma dari gravitasi permukaan bintang.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Logaritma dari Gravitasi Permukaan Bintang', fontsize=20)
sns.histplot(data=df_new, x="logg",label='Data Mentah',color='blue')
sns.histplot(data=df_fix, x="logg",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[11]:


# Buat histogram metalisitas bintang.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Metalisitas Bintang', fontsize=20)
sns.histplot(data=df_new, x="Metallicity",label='Data Mentah',color='blue')
sns.histplot(data=df_fix, x="Metallicity",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[12]:


# Buat histogram radius bintang.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Radius Bintang', fontsize=20)
sns.histplot(data=df_new, x="Rad",label='Data Mentah',color='blue')
sns.histplot(data=df_fix, x="Rad",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[13]:


# Buat histogram temperatur efektif bintang.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Temperatur Efektif Bintang', fontsize=20)
sns.histplot(data=df_new, x="Teff",label='Data Mentah',color='blue')
sns.histplot(data=df_fix, x="Teff",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[14]:


# Buat histogram luminositas bintang.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Luminositas Bintang', fontsize=20)
sns.histplot(data=df_new, x="Lum",label='Data Mentah',color='blue')
sns.histplot(data=df_fix, x="Lum",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[15]:


# Buat histogram massa bintang.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Massa Bintang', fontsize=20)
sns.histplot(data=df_new, x="Mass",label='Data Mentah',color='blue')
sns.histplot(data=df_fix, x="Mass",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[16]:


df_fix_B = df_fix[(df_fix['SpectralType'] == 'B      ')]
df_fix_B = df_fix_B.reset_index(drop=True)
df_fix_B


# In[17]:


df_new_B = df_new[(df_new['SpectralType'] == 'B      ')]
df_new_B = df_new_B.reset_index(drop=True)
df_new_B


# In[18]:


df_fix_B_QSO = df_fix_B[(df_fix_B['PQSO'] > 0.5)]
df_fix_B_QSO = df_fix_B_QSO.reset_index(drop=True)
df_fix_B_QSO


# In[19]:


df_fix_B_Gal = df_fix_B[(df_fix_B['PGal'] > 0.5)]
df_fix_B_Gal = df_fix_B_Gal.reset_index(drop=True)
df_fix_B_Gal


# In[20]:


df_fix_B_Star = df_fix_B[(df_fix_B['Pstar'] > 0.5)]
df_fix_B_Star = df_fix_B_Star.reset_index(drop=True)
df_fix_B_Star


# In[21]:


df_fix_B_WD = df_fix_B[(df_fix_B['PWD'] > 0.5)]
df_fix_B_WD = df_fix_B_WD.reset_index(drop=True)
df_fix_B_WD


# In[22]:


df_fix_B_Bin = df_fix_B[(df_fix_B['Pbin'] > 0.5)]
df_fix_B_Bin = df_fix_B_Bin.reset_index(drop=True)
df_fix_B_Bin


# In[23]:


df_fix_A = df_fix[(df_fix['SpectralType'] == 'A      ')]
df_fix_A = df_fix_A.reset_index(drop=True)
df_fix_A


# In[24]:


df_new_A = df_new[(df_new['SpectralType'] == 'A      ')]
df_new_A = df_new_A.reset_index(drop=True)
df_new_A


# In[25]:


df_fix_A_QSO = df_fix_A[(df_fix_A['PQSO'] > 0.5)]
df_fix_A_QSO = df_fix_A_QSO.reset_index(drop=True)
df_fix_A_QSO


# In[26]:


df_fix_A_Gal = df_fix_A[(df_fix_A['PGal'] > 0.5)]
df_fix_A_Gal = df_fix_A_Gal.reset_index(drop=True)
df_fix_A_Gal


# In[27]:


df_fix_A_Star = df_fix_A[(df_fix_A['Pstar'] > 0.5)]
df_fix_A_Star = df_fix_A_Star.reset_index(drop=True)
df_fix_A_Star


# In[28]:


df_fix_A_WD = df_fix_A[(df_fix_A['PWD'] > 0.5)]
df_fix_A_WD = df_fix_A_WD.reset_index(drop=True)
df_fix_A_WD


# In[29]:


df_fix_A_Bin = df_fix_A[(df_fix_A['Pbin'] > 0.5)]
df_fix_A_Bin = df_fix_A_Bin.reset_index(drop=True)
df_fix_A_Bin


# In[30]:


df_fix_F = df_fix[(df_fix['SpectralType'] == 'F      ')]
df_fix_F = df_fix_F.reset_index(drop=True)
df_fix_F


# In[31]:


df_new_F = df_new[(df_new['SpectralType'] == 'F      ')]
df_new_F = df_new_F.reset_index(drop=True)
df_new_F


# In[32]:


df_fix_F_QSO = df_fix_F[(df_fix_F['PQSO'] > 0.5)]
df_fix_F_QSO = df_fix_F_QSO.reset_index(drop=True)
df_fix_F_QSO


# In[33]:


df_fix_F_Gal = df_fix_F[(df_fix_F['PGal'] > 0.5)]
df_fix_F_Gal = df_fix_F_Gal.reset_index(drop=True)
df_fix_F_Gal


# In[34]:


df_fix_F_Star = df_fix_F[(df_fix_F['Pstar'] > 0.5)]
df_fix_F_Star = df_fix_F_Star.reset_index(drop=True)
df_fix_F_Star


# In[35]:


df_fix_F_WD = df_fix_F[(df_fix_F['PWD'] > 0.5)]
df_fix_F_WD = df_fix_F_WD.reset_index(drop=True)
df_fix_F_WD


# In[36]:


df_fix_F_Bin = df_fix_F[(df_fix_F['Pbin'] > 0.5)]
df_fix_F_Bin = df_fix_F_Bin.reset_index(drop=True)
df_fix_F_Bin


# In[37]:


df_fix_G = df_fix[(df_fix['SpectralType'] == 'G      ')]
df_fix_G = df_fix_G.reset_index(drop=True)
df_fix_G


# In[38]:


df_new_G = df_new[(df_new['SpectralType'] == 'G      ')]
df_new_G = df_new_G.reset_index(drop=True)
df_new_G


# In[39]:


df_fix_G_QSO = df_fix_G[(df_fix_G['PQSO'] > 0.5)]
df_fix_G_QSO = df_fix_G_QSO.reset_index(drop=True)
df_fix_G_QSO


# In[40]:


df_fix_G_Gal = df_fix_G[(df_fix_G['PGal'] > 0.5)]
df_fix_G_Gal = df_fix_G_Gal.reset_index(drop=True)
df_fix_G_Gal


# In[41]:


df_fix_G_Star = df_fix_G[(df_fix_G['Pstar'] > 0.5)]
df_fix_G_Star = df_fix_G_Star.reset_index(drop=True)
df_fix_G_Star


# In[42]:


df_fix_G_WD = df_fix_G[(df_fix_G['PWD'] > 0.5)]
df_fix_G_WD = df_fix_G_WD.reset_index(drop=True)
df_fix_G_WD


# In[43]:


df_fix_G_Bin = df_fix_G[(df_fix_G['Pbin'] > 0.5)]
df_fix_G_Bin = df_fix_G_Bin.reset_index(drop=True)
df_fix_G_Bin


# In[44]:


df_fix_K = df_fix[(df_fix['SpectralType'] == 'K      ')]
df_fix_K = df_fix_K.reset_index(drop=True)
df_fix_K


# In[45]:


df_new_K = df_new[(df_new['SpectralType'] == 'K      ')]
df_new_K = df_new_K.reset_index(drop=True)
df_new_K


# In[46]:


df_fix_K_QSO = df_fix_K[(df_fix_K['PQSO'] > 0.5)]
df_fix_K_QSO = df_fix_K_QSO.reset_index(drop=True)
df_fix_K_QSO


# In[47]:


df_fix_K_Gal = df_fix_K[(df_fix_K['PGal'] > 0.5)]
df_fix_K_Gal = df_fix_K_Gal.reset_index(drop=True)
df_fix_K_Gal


# In[48]:


df_fix_K_Star = df_fix_K[(df_fix_K['Pstar'] > 0.5)]
df_fix_K_Star = df_fix_K_Star.reset_index(drop=True)
df_fix_K_Star


# In[49]:


df_fix_K_WD = df_fix_K[(df_fix_K['PWD'] > 0.5)]
df_fix_K_WD = df_fix_K_WD.reset_index(drop=True)
df_fix_K_WD


# In[50]:


df_fix_K_Bin = df_fix_K[(df_fix_K['Pbin'] > 0.5)]
df_fix_K_Bin = df_fix_K_Bin.reset_index(drop=True)
df_fix_K_Bin


# In[51]:


df_fix_M = df_fix[(df_fix['SpectralType'] == 'M      ')]
df_fix_M = df_fix_M.reset_index(drop=True)
df_fix_M


# In[52]:


df_new_M = df_new[(df_new['SpectralType'] == 'M      ')]
df_new_M = df_new_M.reset_index(drop=True)
df_new_M


# In[53]:


df_fix_M_QSO = df_fix_M[(df_fix_M['PQSO'] > 0.5)]
df_fix_M_QSO = df_fix_M_QSO.reset_index(drop=True)
df_fix_M_QSO


# In[54]:


df_fix_M_Gal = df_fix_M[(df_fix_M['PGal'] > 0.5)]
df_fix_M_Gal = df_fix_M_Gal.reset_index(drop=True)
df_fix_M_Gal


# In[55]:


df_fix_M_Star = df_fix_M[(df_fix_M['Pstar'] > 0.5)]
df_fix_M_Star = df_fix_M_Star.reset_index(drop=True)
df_fix_M_Star


# In[56]:


df_fix_M_WD = df_fix_M[(df_fix_M['PWD'] > 0.5)]
df_fix_M_WD = df_fix_M_WD.reset_index(drop=True)
df_fix_M_WD


# In[57]:


df_fix_M_Bin = df_fix_M[(df_fix_M['Pbin'] > 0.5)]
df_fix_M_Bin = df_fix_M_Bin.reset_index(drop=True)
df_fix_M_Bin


# In[58]:


df_fix_B_Star_MSS = df_fix_B_Star[(df_fix_B_Star['Evol'] <= 360)]
df_fix_B_Star_MSS = df_fix_B_Star_MSS.reset_index(drop=True)
df_fix_B_Star_MSS


# In[59]:


df_fix_B_Star_RGS = df_fix_B_Star[(df_fix_B_Star['Evol'] > 360)]
df_fix_B_Star_RGS = df_fix_B_Star_RGS.reset_index(drop=True)
df_fix_B_Star_RGS


# In[60]:


df_fix_A_Star_MSS = df_fix_A_Star[(df_fix_A_Star['Evol'] <= 360)]
df_fix_A_Star_MSS = df_fix_A_Star_MSS.reset_index(drop=True)
df_fix_A_Star_MSS


# In[61]:


df_fix_A_Star_RGS = df_fix_A_Star[(df_fix_A_Star['Evol'] > 360)]
df_fix_A_Star_RGS = df_fix_A_Star_RGS.reset_index(drop=True)
df_fix_A_Star_RGS


# In[62]:


df_fix_F_Star_MSS = df_fix_F_Star[(df_fix_F_Star['Evol'] <= 360)]
df_fix_F_Star_MSS = df_fix_F_Star_MSS.reset_index(drop=True)
df_fix_F_Star_MSS


# In[63]:


df_fix_F_Star_RGS = df_fix_F_Star[(df_fix_F_Star['Evol'] > 360)]
df_fix_F_Star_RGS = df_fix_F_Star_RGS.reset_index(drop=True)
df_fix_F_Star_RGS


# In[64]:


df_fix_G_Star_MSS = df_fix_G_Star[(df_fix_G_Star['Evol'] <= 360)]
df_fix_G_Star_MSS = df_fix_G_Star_MSS.reset_index(drop=True)
df_fix_G_Star_MSS


# In[65]:


df_fix_G_Star_RGS = df_fix_G_Star[(df_fix_G_Star['Evol'] > 360)]
df_fix_G_Star_RGS = df_fix_G_Star_RGS.reset_index(drop=True)
df_fix_G_Star_RGS


# In[66]:


df_fix_K_Star_MSS = df_fix_K_Star[(df_fix_K_Star['Evol'] <= 360)]
df_fix_K_Star_MSS = df_fix_K_Star_MSS.reset_index(drop=True)
df_fix_K_Star_MSS


# In[67]:


df_fix_K_Star_RGS = df_fix_K_Star[(df_fix_K_Star['Evol'] > 360)]
df_fix_K_Star_RGS = df_fix_K_Star_RGS.reset_index(drop=True)
df_fix_K_Star_RGS


# In[68]:


df_fix_M_Star_MSS = df_fix_M_Star[(df_fix_M_Star['Evol'] <= 360)]
df_fix_M_Star_MSS = df_fix_M_Star_MSS.reset_index(drop=True)
df_fix_M_Star_MSS


# In[69]:


df_fix_M_Star_RGS = df_fix_M_Star[(df_fix_M_Star['Evol'] > 360)]
df_fix_M_Star_RGS = df_fix_M_Star_RGS.reset_index(drop=True)
df_fix_M_Star_RGS


# In[70]:


df_fix_Star = df_fix[(df_fix['Pstar'] > 0.5)]
df_fix_Star = df_fix_Star.reset_index(drop=True)


df_fix_Star_MSS = df_fix_Star[(df_fix_Star['Evol'] <= 360)]
df_fix_Star_MSS = df_fix_Star_MSS.reset_index(drop=True)

df_fix_Star_RGS = df_fix_Star[(df_fix_Star['Evol'] > 360)]
df_fix_Star_RGS = df_fix_Star_RGS.reset_index(drop=True)


# In[71]:


# Buat korelasi antara variabel numerik.
corr_pearson = df_fix_Star_MSS.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_pearson, cmap='RdBu_r', annot=True)

plt.title('Korelasi Antara Variabel Numerik untuk Bintang Deret Utama') # Beri judul.
plt.show() # Tampilkan plot korelasi.


# In[72]:


# Buat korelasi antara variabel numerik.
corr_pearson = df_fix_Star_RGS.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_pearson, cmap='RdBu_r', annot=True)

plt.title('Korelasi Antara Variabel Numerik untuk Bintang Raksasa Merah') # Beri judul.
plt.show() # Tampilkan plot korelasi.


# In[73]:


# Buat histogram logaritma dari gravitasi permukaan.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Logaritma dari Gravitasi Permukaan Bintang Kelas Spektrum B', fontsize=20)
sns.histplot(data=df_new_B, x="logg",label='Data Mentah',color='blue')
sns.histplot(data=df_fix_B, x="logg",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[74]:


# Buat histogram metalisitas.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Metalisitas Bintang Kelas Spektrum B', fontsize=20)
sns.histplot(data=df_new_B, x="Metallicity",label='Data Mentah',color='blue')
sns.histplot(data=df_fix_B, x="Metallicity",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[75]:


# Buat histogram temperatur efektif.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Temperatur Efektif Bintang Kelas Spektrum B', fontsize=20)
sns.histplot(data=df_new_B, x="Teff",label='Data Mentah',color='blue')
sns.histplot(data=df_fix_B, x="Teff",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[76]:


# Buat histogram temperatur efektif.
fig, ax = plt.subplots(figsize = (10,7))
plt.title('Histogram Massa Bintang Kelas Spektrum B', fontsize=20)
sns.histplot(data=df_new_B, x="Mass",label='Data Mentah',color='blue')
sns.histplot(data=df_fix_B, x="Mass",label='Data Diolah',color='orange')
plt.legend()
plt.plot()


# In[77]:


# Cek info data.
df_fix.info()


# In[78]:


X_MSS = df_fix_Star_MSS.iloc[:,[0,2,3,5]]
y_MSS = df_fix_Star_MSS.iloc[:,6]

X_MSS_train, X_MSS_test, y_MSS_train, y_MSS_test = train_test_split(X_MSS, y_MSS, test_size=0.2, random_state=42)

# Definisi model LightGBM
lgb_model_MSS = lgb.LGBMRegressor()

# Buat rentang learning rate yang akan diuji
lr_arr = np.arange(0.04, 1.04, 0.04)
lr = (list(lr_arr))
for i in range(len(lr)):
    lr[i] = round(lr[i], 2)

# Buat rentang n_estimators yang akan diuji
ne_arr = np.arange(25, 1025, 25)
ne = list(ne_arr)

# Set hyperparameter yang akan diuji
param_grid = {
    'learning_rate': lr,
    'n_estimators': ne
}

# Gunakan GridSearchCV untuk mencari kombinasi hyperparameter terbaik
grid_search_MSS = GridSearchCV(lgb_model_MSS, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_MSS.fit(X_MSS_train, y_MSS_train)

# Print hyperparameter terbaik
best_params_MSS = grid_search_MSS.best_params_
print("Hyperparameter terbaik:", best_params_MSS)

# Latih ulang model dengan hyperparameter terbaik pada seluruh data pelatihan
best_lgb_model_MSS = lgb.LGBMRegressor(**best_params_MSS)
best_lgb_model_MSS.fit(X_MSS_train, y_MSS_train)

# Lakukan prediksi pada data pengujian
y_MSS_pred = best_lgb_model_MSS.predict(X_MSS_test)

# Evaluasi performa model
rmse_MSS_test = mean_squared_error(y_MSS_test, y_MSS_pred) ** 0.5

def rmsle(y_MSS_true, y_MSS_pred):
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_MSS_pred) - np.log1p(y_MSS_true), 2))), False

def rae(y_MSS_true, y_MSS_pred):
    return 'RAE', np.sum(np.abs(y_MSS_pred - y_MSS_true)) / np.sum(np.abs(np.mean(y_MSS_true) - y_MSS_true)), False

rmsle_MSS_test = rmsle(y_MSS_test, y_MSS_pred)[1]
rae_MSS_test = rae(y_MSS_test, y_MSS_pred)[1]
mae_MSS_test = mean_absolute_error(y_MSS_test, y_MSS_pred)
r2_MSS_test = r2_score(y_MSS_test, y_MSS_pred)

print(f'The RMSE of prediction is: {rmse_MSS_test}')
print(f'The RMSLE of prediction is: {rmsle_MSS_test}')
print(f'The RAE of prediction is: {rae_MSS_test}')
print(f'The MAE of prediction is: {mae_MSS_test}')
print(f'The R2 Score of prediction is: {r2_MSS_test}')


# In[79]:


X_RGS = df_fix_Star_RGS.iloc[:,[0,5]]
y_RGS = df_fix_Star_RGS.iloc[:,6]

X_RGS_train, X_RGS_test, y_RGS_train, y_RGS_test = train_test_split(X_RGS, y_RGS, test_size=0.2, random_state=42)

# Definisi model LightGBM
lgb_model_RGS = lgb.LGBMRegressor()

# Buat rentang learning rate yang akan diuji
lr_arr = np.arange(0.04, 1.04, 0.04)
lr = (list(lr_arr))
for i in range(len(lr)):
    lr[i] = round(lr[i], 2)

# Buat rentang n_estimators yang akan diuji
ne_arr = np.arange(25, 1025, 25)
ne = list(ne_arr)

# Set hyperparameter yang akan diuji
param_grid = {
    'learning_rate': lr,
    'n_estimators': ne
}

# Gunakan GridSearchCV untuk mencari kombinasi hyperparameter terbaik
grid_search_RGS = GridSearchCV(lgb_model_RGS, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_RGS.fit(X_RGS_train, y_RGS_train)

# Print hyperparameter terbaik
best_params_RGS = grid_search_RGS.best_params_
print("Hyperparameter terbaik:", best_params_RGS)

# Latih ulang model dengan hyperparameter terbaik pada seluruh data pelatihan
best_lgb_model_RGS = lgb.LGBMRegressor(**best_params_RGS)
best_lgb_model_RGS.fit(X_RGS_train, y_RGS_train)

# Lakukan prediksi pada data pengujian
y_RGS_pred = best_lgb_model_RGS.predict(X_RGS_test)

# Evaluasi performa model
rmse_RGS_test = mean_squared_error(y_RGS_test, y_RGS_pred) ** 0.5

def rmsle(y_RGS_true, y_RGS_pred):
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_RGS_pred) - np.log1p(y_RGS_true), 2))), False

def rae(y_RGS_true, y_RGS_pred):
    return 'RAE', np.sum(np.abs(y_RGS_pred - y_RGS_true)) / np.sum(np.abs(np.mean(y_RGS_true) - y_RGS_true)), False

rmsle_RGS_test = rmsle(y_RGS_test, y_RGS_pred)[1]
rae_RGS_test = rae(y_RGS_test, y_RGS_pred)[1]
mae_RGS_test = mean_absolute_error(y_RGS_test, y_RGS_pred)
r2_RGS_test = r2_score(y_RGS_test, y_RGS_pred)

print(f'The RMSE of prediction is: {rmse_RGS_test}')
print(f'The RMSLE of prediction is: {rmsle_RGS_test}')
print(f'The RAE of prediction is: {rae_RGS_test}')
print(f'The MAE of prediction is: {mae_RGS_test}')
print(f'The R2 Score of prediction is: {r2_RGS_test}')


# In[80]:


yequalx = np.linspace(0.2,5)

plt.plot(yequalx,yequalx,color='orange',label='y = x')
plt.scatter(y_MSS_test,y_MSS_pred,s=1,color='blue',label='Umur Bintang Deret Utama')
plt.scatter(y_RGS_test,y_RGS_pred,s=1,color='red',label='Umur Bintang Raksasa Merah')
plt.title("Scatter Plot Umur Bintang\n Menggunakan LightGBM") # Beri judul scatter plot.
plt.xlabel("Umur (Tes)") # Beri label x.
plt.ylabel("Umur (Prediksi)") # Beri label y.
plt.grid()
plt.legend()
plt.show() # Tampilkan scatter plot.


# In[81]:


y_residual_MSS = [0 for i in range(len(y_MSS_pred))]
y_residual_RGS = [0 for i in range(len(y_RGS_pred))]

y_MSS_test_arr = np.array(y_MSS_test)
y_RGS_test_arr = np.array(y_RGS_test)

for i in range(len(y_MSS_pred)):
    y_residual_MSS[i] = y_MSS_test_arr[i] - y_MSS_pred[i]
for i in range(len(y_RGS_pred)):
    y_residual_RGS[i] = y_RGS_test_arr[i] - y_RGS_pred[i]

plt.scatter(y_MSS_pred,y_residual_MSS,s=1,color='blue',label='Umur Bintang Deret Utama') # Buat scatter plot data.
plt.scatter(y_RGS_pred,y_residual_RGS,s=1,color='red',label='Umur Bintang Raksasa Merah') # Buat scatter plot data.
plt.title("Scatter Plot Umur Bintang\n Menggunakan LightGBM") # Beri judul scatter plot.
plt.xlabel("Nilai Prediksi") # Beri label x.
plt.ylabel("Nilai Residual") # Beri label y.
plt.grid()
plt.legend()
plt.show() # Tampilkan scatter plot.


# In[82]:


print('Tabel Perbandingan Usia Tes dan Prediksi Bintang Deret Utama')

# Buat tabel perbandingan nilai umur tes dan prediksi.
y_MSS_diff_table = {'Usia (Tes)':y_MSS_test_arr,'Usia (Prediksi)':y_MSS_pred}
df_y_MSS_diff_table = pd.DataFrame(y_MSS_diff_table)

df_y_MSS_diff_table


# In[83]:


print('Tabel Perbandingan Usia Tes dan Prediksi Bintang Raksasa Merah Tipe A')

# Buat tabel perbandingan nilai umur tes dan prediksi.
y_RGS_diff_table = {'Usia (Tes)':y_RGS_test_arr,'Usia (Prediksi)':y_RGS_pred}
df_y_RGS_diff_table = pd.DataFrame(y_RGS_diff_table)

df_y_RGS_diff_table


# In[84]:


# Simpan model untuk Bintang Deret Utama.
# Gunakan fungsi open().
modelMSS = 'C:\\Users\\ACER\\Downloads\\Tugas Akhir Astronomi\\Uji Coba LightGBM\\Version 2\\modelMSS.txt'

# Buka file di write mode
with open(modelMSS,'w') as file:
  # Write content to the file
  file.write(str(best_lgb_model_MSS))

    
# Simpan model untuk Bintang Raksasa Merah.
# Gunakan fungsi open().
modelRGS = 'C:\\Users\\ACER\\Downloads\\Tugas Akhir Astronomi\\Uji Coba LightGBM\\Version 2\\modelRGS.txt'

# Buka file di write mode
with open(modelRGS,'w') as file:
  # Write content to the file
  file.write(str(best_lgb_model_RGS))


# In[85]:


# Simpan data.
df_y_MSS_diff_table.to_csv('C:\\Users\\ACER\\Downloads\\Tugas Akhir Astronomi\\Uji Coba LightGBM\\Version 2\\MainSequenceStars.csv')
df_y_RGS_diff_table.to_csv('C:\\Users\\ACER\\Downloads\\Tugas Akhir Astronomi\\Uji Coba LightGBM\\Version 2\\RedGiantStars.csv')


# In[86]:


data_metric_MSS = {'RMSE':[rmse_MSS_test],'RMSLE':[rmsle_MSS_test],'RAE':[rae_MSS_test],
                  'MAE':[mae_MSS_test],'R2 Score':[r2_MSS_test]}
df_metric_MSS = pd.DataFrame(data_metric_MSS)

data_metric_RGS = {'RMSE':[rmse_RGS_test],'RMSLE':[rmsle_RGS_test],'RAE':[rae_RGS_test],
                  'MAE':[mae_RGS_test],'R2 Score':[r2_RGS_test]}
df_metric_RGS = pd.DataFrame(data_metric_RGS)

# Simpan data.
df_metric_MSS.to_csv('C:\\Users\\ACER\\Downloads\\Tugas Akhir Astronomi\\Uji Coba LightGBM\\Version 2\\Metric_MainSequenceStars.csv')
df_metric_RGS.to_csv('C:\\Users\\ACER\\Downloads\\Tugas Akhir Astronomi\\Uji Coba LightGBM\\Version 2\\Metric_RedGiantStars.csv')

