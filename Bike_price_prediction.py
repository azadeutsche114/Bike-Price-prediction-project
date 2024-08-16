#!/usr/bin/env python
# coding: utf-8

# # Seoul bike sharing demand prediction
# 

# ### Import packeges 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ### Load Data

# In[2]:


data_path = r"E:\Projects\ML Projects\001 Bike-Sharing-Demand-Prediction\data\BikeData.csv"
df = pd.read_csv(data_path, encoding="unicode_escape")
df.shape


# In[3]:


df.head()


# In[4]:


df.tail()


# ### Data information

# In[5]:


df.info()


# In[6]:


df.describe(include="all").T


# ### Check Null Value

# In[7]:


df.isnull().sum()


# In[8]:


df["Date"] = pd.to_datetime(df["Date"])

df["weekday"] = df["Date"].dt.day_name()
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

df.drop("Date", axis=1, inplace=True)


# In[9]:


df.info()


# In[10]:


df.head()


# ## EDA

# In[11]:


sns.pairplot(df)


# In[12]:


plt.figure(figsize=(10,7))
Month = df.groupby("Month").sum().reset_index()
sns.barplot(x="Month", y="Rented Bike Count", data=Month)


# In[13]:


plt.figure(figsize=(10,7))
Month = df.groupby("Day").sum().reset_index()
sns.barplot(x="Day", y="Rented Bike Count", data=Month)


# In[14]:


plt.figure(figsize=(10,7))
Hour = df.groupby("Hour").sum().reset_index()
sns.barplot(x="Hour", y="Rented Bike Count", data=Hour)


# In[15]:


plt.figure(figsize=(10,7))
sns.barplot(x="Holiday", y="Rented Bike Count", data=df)


# In[16]:


plt.figure(figsize=(10,7))
sns.barplot(x="Seasons", y="Rented Bike Count", data=df)


# In[17]:


plt.figure(figsize=(40,7))
sns.barplot(x="Rainfall(mm)", y="Rented Bike Count", data=df)


# In[18]:


plt.figure(figsize=(40,7))
sns.barplot(x="Snowfall (cm)", y="Rented Bike Count", data=df)


# In[19]:


plt.figure(figsize=(40,7))
sns.displot(df["Rented Bike Count"])


# In[20]:


sns.displot(np.sqrt(df["Rented Bike Count"]))


# ## Skewed Data

# In[21]:


df.skew().sort_values(ascending=True)


# ## Remove Multicollinearity

# In[22]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")


# In[23]:


def get_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [ variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    return vif


# In[24]:


not_for_vif = [ "Day", "Month", "Year", "Rented Bike Count"] 

get_vif(df[[i for i in df.describe().columns if i not in not_for_vif]])


# In[25]:


not_for_vif = [ "Day", "Month", "Year", "Rented Bike Count", "Dew point temperature(°C)"] 

get_vif(df[[i for i in df.describe().columns if i not in not_for_vif]])


# In[26]:


df.drop(["Dew point temperature(°C)"], axis=1, inplace=True)


# ## Encoding

# In[27]:


df.info()


# In[28]:


cat_features = ["Seasons", "Holiday", "Functioning Day", "weekday"]


# In[29]:


df["Holiday"].value_counts()


# In[30]:


df["Functioning Day"].value_counts()


# In[31]:


df["Seasons"].value_counts()


# In[32]:


df["weekday"].value_counts()


# In[33]:


df["Holiday"] = df["Holiday"].map({"No Holiday":0, "Holiday":1})
df["Functioning Day"] = df["Functioning Day"].map({"No":0, "Yes":1})


# In[34]:


df_season = pd.get_dummies(df["Seasons"], drop_first = True)
df_weekday = pd.get_dummies(df["weekday"], drop_first = True)


# In[35]:


df.info()


# In[36]:


df = pd.concat([df, df_season, df_weekday], axis=1)


# In[37]:


df.info()


# In[38]:


df.drop(["Seasons", "weekday"], axis=1, inplace=True)


# In[39]:


df.info()


# In[40]:


df.head()


# In[68]:


df.columns


# In[41]:


df.shape


# ## Split Data for Training & Tesing

# In[42]:


X = df.drop("Rented Bike Count", axis=1)
y = df["Rented Bike Count"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

print("Shape of X_train : ", X_train.shape)
print("Shape of y_train : ", y_train.shape)
print("Shape of X_test : ", X_test.shape)
print("Shape of y_test : ", y_test.shape)


# ## Scaling

# In[43]:


sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# In[44]:


X_train[:2]


# In[45]:


sc.mean_


# In[46]:


sc.scale_


# # Training ML Model

# ## Linear Regression Model

# In[47]:


from sklearn.linear_model import LinearRegression


# In[48]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[49]:


y_pred = lr.predict(X_test)


# In[50]:


y_pred


# # Model Evaluation

# In[51]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[52]:


MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print(f"MSE : {MSE}")
print(f"RMSE : {RMSE}")
print(f"MAE : {MAE}")
print(f"R2 : {R2}")


# In[53]:


def get_metrics(y_true, y_pred, model_name):
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} : ['MSE': {round(MSE,3)}, 'RMSE':{round(RMSE,3)}, 'MAE' :{round(MAE,3)}, 'R2':{round(R2,3)}]")


# In[54]:


get_metrics(y_test, y_pred, "LinearRegression")


# ## Train Multiple Models

# In[55]:


get_ipython().system('pip install xgboost')


# In[56]:


from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[57]:


rir = Ridge().fit(X_train, y_train)
y_pred_rir = rir.predict(X_test)

lar = Lasso().fit(X_train, y_train)
y_pred_lar = lar.predict(X_test)

poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

poly_r = LinearRegression().fit(X_train_poly, y_train)
y_pred_poly = poly_r.predict(X_test_poly)

svr = SVR().fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

knnr = KNeighborsRegressor().fit(X_train, y_train)
y_pred_knnr = knnr.predict(X_test)

dtr = DecisionTreeRegressor().fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_test)

rfr = RandomForestRegressor().fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)

xgbr = XGBRegressor().fit(X_train, y_train)
y_pred_xgbr = xgbr.predict(X_test)


# In[58]:


get_metrics(y_test, y_pred_rir, "Ridge")
get_metrics(y_test, y_pred_lar, "Lasso")
get_metrics(y_test, y_pred_poly, "PolynomialFeatures")
get_metrics(y_test, y_pred_svr, "SVR")
get_metrics(y_test, y_pred_knnr, "KNNR")
get_metrics(y_test, y_pred_dtr, "DecisionTreeRegressor")
get_metrics(y_test, y_pred_rfr, "RandomForestRegressor")
get_metrics(y_test, y_pred_xgbr, "XGBRegressor")


# ## Visualise Model Prediction

# In[59]:


plt.scatter(y_test, y_pred)
plt.title("Linear Regression Truth vs Prediction ")
plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.show()


# In[60]:


plt.scatter(y_test, y_pred_rfr)
plt.title("Random Forest Regressor Truth vs Prediction ")
plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.show()


# In[61]:


plt.scatter(y_test, y_pred_xgbr)
plt.title("XGB Regressor Truth vs Prediction ")
plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.show()


# ## Hyperparameter Tuning for Random Forest Regressor

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# maximum number of levels allowed in each decision tree
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


import time 
start_time = time.time()

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
y_pred_rf_random = rf_random.predict(X_test)

print("Time taken to training using randomize search : ", time.time()-start_time)


# In[ ]:


get_metrics(y_test, y_pred_rf_random, "RandomForestRegressor Fine Tuning")


# In[ ]:


rf_random.best_params_


# In[ ]:


rf_tuned= RandomForestRegressor(n_estimators=400,
                               min_samples_split=2,
                               min_samples_leaf=1,
                                max_features="sqrt",
                                max_depth=120,
                                bootstrap=False)
rf_tuned.fit(X_train, y_train)
y_pred_rf_tuned = rf_tuned.predict(X_test)

get_metrics(y_test, y_pred_rf_tuned, "RandomForestRegressor With Best Parameters")


# In[ ]:





# ## Hyperparameter Tuning for XGBoost Regressor# 

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

import time 
start_time = time.time()

params = { 'max_depth': [3, 5, 6, 10, 15, 20],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 500, 1000]}


xgbr = XGBRegressor(seed = 20)
rscv = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params,
                         scoring='neg_mean_squared_error',
                         n_iter=25,
                          cv=5,
                         verbose=1)

rscv.fit(X_train, y_train)

y_pred_xgb_random = rscv.predict(X_test)

get_metrics(y_test, y_pred_xgb_random, "XGBRegressor With Best Parameters")

print("Time taken to training using randomize search : ", time.time()-start_time)

print("Best parameters:", rscv.best_params_)


# In[ ]:


xgbr = XGBRegressor(subsample=0.6,
                   n_estimators=1000,
                   max_depth=6,
                   learning_rate=0.1,
                   colsample_bytree=0.7,
                   colsample_bylevel=0.4,
                   seed = 20)

xgbr.fit(X_train, y_train)

y_pred_tuned = xgbr.predict(X_test)

get_metrics(y_test, y_pred_tuned, "XGBRegressor With Best Parameters")


# In[62]:


xgbr = XGBRegressor(subsample=0.6,
                   n_estimators=1000,
                   max_depth=6,
                   learning_rate=0.09,
                   colsample_bytree=0.7,
                   colsample_bylevel=0.4,
                   seed = 20)

xgbr.fit(X_train, y_train)

y_pred_tuned = xgbr.predict(X_test)

get_metrics(y_test, y_pred_tuned, "XGBRegressor With Best Parameters")


# ## Save ML Best Model

# In[63]:


import pickle
import os

dir = r"E:\Projects\ML Projects\001 RG Seoul-Bike-Sharing-Demand-Prediction\models"
model_file_name = "xgboost_regressor_r2_0_928_v1.pkl"

model_file_path = os.path.join(dir, model_file_name)

pickle.dump(xgbr, open(model_file_path, "wb"))


# In[64]:


X_test[0,:]


# In[65]:


X_test[1,:]


# In[66]:


y_test


# ## Dump Scaling Parameters

# In[67]:


sc_dump_path = r"E:\Projects\ML Projects\001 RG Seoul-Bike-Sharing-Demand-Prediction\models\sc.pkl"

pickle.dump(sc, open(sc_dump_path, "wb"))


# In[ ]:





# In[ ]:





# In[ ]:





# ## Deployment

# In[ ]:





# In[80]:


df.head(1)


# ## Users Input Data

# In[81]:


date = "01/03/2023"
hour = 5
temperature_c = 0
humidity = 25
wind_speed = 5.2
visibility = 2
solar_radiation = 0.0
rainfall = 35
snowfall = 8
holiday = "Holiday"
functioning_day = "No"
seasons = "Winter" # ["Spring", "Summer", "Autumn", "Winter"]


# # Convert users input to models consumable format

# In[82]:


holiday_dic = {"No Holiday":0, "Holiday":1}
functioning_day_dic = {"Yes":1, "No":0} 


# In[83]:


from datetime import datetime

def get_string_to_date_time(date):
    dt = datetime.strptime(date, '%d/%m/%Y')
    return {"Day": dt.day, "Month":dt.month, "Year":dt.year, "day_name":dt.strftime("%A")}

date_time = get_string_to_date_time(date)


# In[84]:


input_for_pred = [hour, temperature_c, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall, 
                  holiday_dic[holiday], functioning_day_dic[functioning_day], 
                  date_time["Day"], date_time["Month"], date_time["Year"],
                 ]

input_for_pred


# In[85]:


input_cols = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)',
              'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Day','Month', 'Year',]
                 
df_inputs = pd.DataFrame([input_for_pred], columns= input_cols)
df_inputs


# ## Users Date Processing

# In[86]:


u_day_name = "Monday" #date_time["day_name"]

cols_day = ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
data = np.zeros((1,len(cols_day)), dtype=int)
df_days = pd.DataFrame(data, columns=cols_day)

if u_day_name in cols_day:
    df_days[u_day_name]= 1

df_days


# ## Users Season processing

# In[87]:


u_seasons = "Spring" #["Spring", "Summer", "Autumn", "Winter"]

cols_seasons = ['Spring', 'Summer', 'Winter']
data = np.zeros((1,len(cols_seasons)), dtype=int)
df_seasons = pd.DataFrame(data, columns=cols_seasons)

if u_seasons in cols_seasons:
    df_seasons[u_seasons]= 1

df_seasons


# In[88]:


df_pred = pd.concat([df_inputs, df_seasons, df_days], axis=1)
df_pred


# In[89]:


u_input_pred = sc.transform(df_pred)
u_input_pred


# In[90]:


xgbr.predict(u_input_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




