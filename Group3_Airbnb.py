data = []

f = open("AB_NYC_2019.csv")
import csv
rows = csv.reader(f)

next(rows)

class Airbnb_NYC:
    property_id                             =  0
    name                                    =  ""
    host_id                                 =  ""
    host_name                               =  0
    neighbourhood_group                     =  ""
    neighbourhood                           =  0
    latitude                                =  0
    longitude                               =  0
    room_type                               =  ""
    price                                   =  0
    minimum_nights                          =  0
    number_of_reviews                       =  0
#   last_review                             =  0
    reviews_per_month                       =  0
    calculated_host_listings_count          =  0
    availability_365                        =  0
    
for col in rows:
    if col[9] == "" or col[9] == "0":
        continue
    if col[11] == "":
        col[11] = 0
    if col[13] == "":
        col[13] = 0
    if int(col[15]) > 365:
        col[15] = 365
               
    a                                       =  Airbnb_NYC()
    a.property_id                           =  int(col[0])
    a.name                                  =  col[1]
    a.host_id                               =  int(col[2])
    a.host_name                             =  col[3]
    a.neighbourhood_group                   =  col[4]
    a.neighbourhood                         =  col[5]
    a.latitude                              =  float(col[6])
    a.longitude                             =  float(col[7])  
    a.room_type                             =  col[8]  
    a.price                                 =  float(col[9])
    a.minimum_nights                        =  int(col[10])  
    a.number_of_reviews                     =  int(col[11])  
#   a.last_review                           =  col[12]  
    a.reviews_per_month                     =  float(col[13])  
    a.calculated_host_listings_count        =  float(col[14])  
    a.availability_365                      =  int(col[15])
    
    data.append(a)

print("read", len(data), "Airbnb properties")
print("==========================================")

data2 =[]
boroughs_list       = list(set(map(lambda x: x.neighbourhood_group, data)))
neighbourhoods = list(set(map(lambda x: x.neighbourhood, data)))
#print(boroughs_list)

class Neighbourhood:
    name                                    =  ""
    borough                                 =  ""
    nei_price_min                           =  0
    price_avg                               =  0
    nei_price_max                           =  0 
    minimum_nights_avg                      =  0
    number_of_reviews_avg                   =  0
    reviews_per_month_avg                   =  0
    calculated_host_listings_avg            =  0
    availability_365_avg                    =  0
    entire_apt_count                        =  0
    privateRoom_count                       =  0
    sharedRoom_count                        =  0
    nei_properties_count                    =  0

for nei in neighbourhoods:
        
        
    filtered         = list(filter(lambda x: x.neighbourhood == nei, data))
    price_avg_list   = list(map( lambda x: x.price, filtered))
    price_avg_sorted = list(sorted(price_avg_list))
    nei_min          = price_avg_sorted[0]
    nei_max          = price_avg_sorted[len(price_avg_sorted)-1]
    boroughs         = list(map( lambda x: x.neighbourhood_group, filtered))
    avg_price        = sum(price_avg_list) / len(price_avg_list)
    
    min_nights_avg_list = list(map( lambda x: x.minimum_nights, filtered))
    nights_avg          = sum(min_nights_avg_list) / len(min_nights_avg_list)
    
    reviews_avg_list = list(map( lambda x: x.number_of_reviews, filtered))
    reviews_avg      = sum(reviews_avg_list) / len(reviews_avg_list)
    
    reviews_avg_month_list = list(map( lambda x: x.reviews_per_month, filtered))
    reviews_avg_month      = sum(reviews_avg_month_list) / len(reviews_avg_month_list) 

    listing_count_avg_list = list(map( lambda x: x.calculated_host_listings_count, filtered))
    listing_count_avg      = sum(listing_count_avg_list) / len(listing_count_avg_list)   
    
    availability_365_avg_list = list(map( lambda x: x.availability_365, filtered))
    availability_avg          = sum(availability_365_avg_list) / len(availability_365_avg_list)
    
    nei_entire_list   = list(filter(lambda x: x.room_type == "Entire home/apt", filtered))

    nei_private_list  = list(filter(lambda x: x.room_type == "Private room", filtered))

    nei_shared_list   = list(filter(lambda x: x.room_type == "Shared room", filtered))

    
    neighbourhood                              = Neighbourhood()
    neighbourhood.name                         = nei
    neighbourhood.borough                      = boroughs[0]
    neighbourhood.nei_price_min                =nei_min
    neighbourhood.price_avg                    = avg_price
    neighbourhood.nei_price_max                =nei_max
    neighbourhood.minimum_nights_avg           = nights_avg
    neighbourhood.number_of_reviews_avg        = reviews_avg
    neighbourhood.reviews_per_month_avg        = reviews_avg_month
    neighbourhood.calculated_host_listings_avg = listing_count_avg
    neighbourhood.availability_365_avg         = availability_avg
    neighbourhood.entire_apt_count             = len(nei_entire_list)
    neighbourhood.privateRoom_count            = len(nei_private_list)
    neighbourhood.sharedRoom_count             = len(nei_shared_list)
    neighbourhood.nei_properties_count         = len(filtered)
    
    data2.append(neighbourhood)

header2 = ["Borough" , "Neighbourhood", "Minimum Price", "Average Price","Maximum Price", "Avg Min Nights", "Avg Reviews Count", "Monthly Avg Reviews", "Avg Listings Count", "Avg Availability Days", "Entire Apt/House", "Private Rooms", "Shared Rooms", "All Properties Count"]   
for d in data2:
    print(d.borough,"-", d.name,"-", "min price: ",round(d.nei_price_min, 2),"avg price: ",round(d.price_avg, 2),
          "max price: ",round(d.nei_price_max, 2),round(d.minimum_nights_avg,2), round(d.number_of_reviews_avg,2), round(d.reviews_per_month_avg,2), round(d.calculated_host_listings_avg,2), round(d.availability_365_avg,2), d.entire_apt_count, d.privateRoom_count, d.sharedRoom_count, d.nei_properties_count,"\n")
print("==========================================")  
"""
# import hierarchical clustering libraries
#import scipy.cluster.hierarchy as sch
#from sklearn.cluster import AgglomerativeClustering

# create dendrogram
#dendrogram = sch.dendrogram(sch.linkage(data2, method='ward'))
# create clusters
#hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
#y_hc = hc.fit_predict(data2)
    
#plt.scatter(data2[y_hc ==0,0], data2[y_hc == 0,1], s=100, c='red')
#plt.scatter(data2[y_hc==1,0], data2[y_hc == 1,1], s=100, c='black')
#plt.scatter(data2[y_hc ==2,0], data2[y_hc == 2,1], s=100, c='blue')
#plt.scatter(data2[y_hc ==3,0], data2[y_hc == 3,1], s=100, c='cyan')   
"""    
#data.describe()
#data.info()
data3 =[]

class NYC_boroughs:
    borough_name                            =  ""
    bor_price_min                           =  0
    price_avg                               =  0
    bor_price_max                           =  0
    minimum_nights_avg                      =  0
    number_of_reviews_avg                   =  0
    reviews_per_month_avg                   =  0
    calculated_host_listings_avg            =  0
    availability_365_avg                    =  0
    bor_entire_count                        =  0
    bor_privateRoom_count                   =  0
    bor_sharedRoom_count                    =  0
    bor_properties_count                    =  0

for bor in boroughs_list:
        
        
    filtered_bor         = list(filter(lambda x: x.neighbourhood_group == bor, data))
    bor_p_avg_list   = list(map( lambda x: x.price, filtered_bor))
    bor_price_avg_sorted = list(sorted(bor_p_avg_list))
    bor_min          = bor_price_avg_sorted[0]
    bor_max          = bor_price_avg_sorted[len(bor_price_avg_sorted)-1]
    bor_avg_price        = sum(bor_p_avg_list) / len(bor_p_avg_list)

    
    bor_min_n_avg_list = list(map( lambda x: x.minimum_nights, filtered_bor))
    bor_nights_avg          = sum(bor_min_n_avg_list) / len(bor_min_n_avg_list)
    
    bor_rev_avg_list = list(map( lambda x: x.number_of_reviews, filtered_bor))
    bor_reviews_avg      = sum(bor_rev_avg_list) / len(bor_rev_avg_list)
    
    reviews_avg_month_list = list(map( lambda x: x.reviews_per_month, filtered_bor))
    bor_r_avg_month      = sum(reviews_avg_month_list) / len(reviews_avg_month_list) 

    bor_l_count_avg_list = list(map( lambda x: x.calculated_host_listings_count, filtered_bor))
    bor_l_count_avg      = sum(bor_l_count_avg_list) / len(bor_l_count_avg_list)   
    
    bor_avail_avg_list = list(map( lambda x: x.availability_365, filtered_bor))
    bor_avail_avg          = sum(bor_avail_avg_list) / len(bor_avail_avg_list)
    
    bor_entire_list   = list(filter(lambda x: x.room_type == "Entire home/apt", filtered_bor))
    bor_private_list  = list(filter(lambda x: x.room_type == "Private room", filtered_bor))
    bor_shared_list   = list(filter(lambda x: x.room_type == "Shared room", filtered_bor))

    
    nyc_boroughs                              = NYC_boroughs()
    nyc_boroughs.borough_name                 = bor
    nyc_boroughs.bor_price_min                = bor_min
    nyc_boroughs.price_avg                    = bor_avg_price
    nyc_boroughs.bor_price_max                = bor_max
    nyc_boroughs.minimum_nights_avg           = bor_nights_avg
    nyc_boroughs.number_of_reviews_avg        = bor_reviews_avg
    nyc_boroughs.reviews_per_month_avg        = bor_r_avg_month
    nyc_boroughs.calculated_host_listings_avg = bor_l_count_avg
    nyc_boroughs.availability_365_avg         = bor_avail_avg
    nyc_boroughs.bor_entire_count             = len(bor_entire_list)  #entire Apartment
    nyc_boroughs.bor_privateRoom_count        = len(bor_private_list) # private room
    nyc_boroughs.bor_sharedRoom_count         = len(bor_shared_list)  # shared room
    nyc_boroughs.bor_properties_count         = len(filtered_bor)
    
    data3.append(nyc_boroughs)

    
for dat in data3:
    print(dat.borough_name,":\n","min price: " ,round(dat.bor_price_min, 2),
          "price avg: ",round(dat.price_avg, 2),"max price: ", round(dat.bor_price_max, 2),
          "avg min nights: ", round(dat.minimum_nights_avg,2), "avg revies: ", round(dat.number_of_reviews_avg,2),"avg monthly reviews:", round(dat.reviews_per_month_avg,2), "avg listing: ",round(dat.calculated_host_listings_avg,2), "avg availablity: ",round(dat.availability_365_avg,2), "# entire apt: ",dat.bor_entire_count, "# private room: ",dat.bor_privateRoom_count, "# shared room: ",dat.bor_sharedRoom_count, "# all properties: ", dat.bor_properties_count,"\n")

header3 = ["Borough" , "Minimum Price", "Average Price","Maximum Price", "Avg Min Nights", "Avg Reviews Count", "Monthly Avg Reviews", "Avg Listings Count", "Avg Availability Days", "Entire Apt/House", "Private Rooms", "Shared Rooms", "All Properties Count"]

#from tabulate import tabul

#for dat in data3:
#    print(tabul(dat.borough_name ,round(dat.bor_price_min, 2),
#          round(dat.price_avg, 2), round(dat.bor_price_max, 2),
#          round(dat.minimum_nights_avg,2), round(dat.number_of_reviews_avg,2), round(dat.reviews_per_month_avg,2),round(dat.calculated_host_listings_avg,2),round(dat.availability_365_avg,2),dat.bor_entire_count,dat.bor_privateRoom_count,dat.bor_sharedRoom_count, dat.bor_properties_count, headers = header3, tablefmt = "grid" ))
print("==========================================")

#print(tabulate([['Alice', 24], ['Bob', 19]], headers=['Name', 'Age']))
print("==========================================")



# Linh Nguyen
# 11/18/2020
import pandas as pd
import seaborn as sns

df = pd.read_csv('./AB_NYC_2019.csv')
# Condition added for data cleansing Behzad 11.20.2020
#print("df" , len(df))
df = df[df["price"] > 0 ]
df.loc[df['number_of_reviews'] == "", 'number_of_reviews'] = 0
df.loc[df['reviews_per_month'] == "", 'reviews_per_month'] = 0

#print("df second time" ,len(df))

# we can pnly correlate numerical variables
correlatable =  df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                 'calculated_host_listings_count', 'availability_365']]  
corr = correlatable.corr()
sns.heatmap(corr)
print("==========================================")

import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression

#####################################################################

# Linear Regression Behzad 11.24.2020
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


col_list = [ "neighbourhood_group","room_type", "price" , "minimum_nights", "reviews_per_month" , "calculated_host_listings_count" ,"availability_365"  ]
df_reg = pd.read_csv('./AB_NYC_2019.csv', usecols = col_list)

df_reg = df_reg[(df_reg["price"] > 0) & (df_reg["room_type"] == "Entire home/apt") ]  # only select entire houses/ apartments
df_reg["reviews_per_month"].fillna(0, inplace = True)
df_reg.loc[df_reg['availability_365'] > 365, 'availability_365'] = 365
dataset = df_reg[[ "neighbourhood_group" ,"minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365" , "price"] ]




X = dataset.iloc[:, :-1]
y =dataset.iloc[:, 5]

boroughs_reg = pd.get_dummies(X["neighbourhood_group"], drop_first= True)   # create numbers for categorical variable
X = X.drop ("neighbourhood_group", axis = 1)   #drop the borough column
X = pd.concat([X,boroughs_reg ], axis = 1)    # replace the dropped column with the created dummy varibale


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



lm = LinearRegression()
lm.fit(X_train, y_train)



params = np.append(lm.intercept_,lm.coef_)
y_pred = lm.predict(X_test)

r2_test=r2_score(y_test,y_pred)                    # r-squared =  1 - RSS/ TSS ( closer to 1 the better) # r=squared is exteremely low
print("LINEAR REGRESSION: ")
#print("TEST  R-Squared: ", r2_test)

X_s = X_train
Y_s = y_train
X_s = sm.add_constant(X_s) # adding a constant
model = sm.OLS(Y_s, X_s).fit()
predictions = model.predict(X_s) 
print_model = model.summary()
print(print_model)

print('Linear Regression Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# RIDGE Regression
print("====================================")
print("RIDGE REGRESSION: ")
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

mse=cross_val_score(lm,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print("mean_mse: " ,mean_mse)


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)

print("Ridge best Lambda: ", ridge_regressor.best_params_)
print("Ridge best mean_mse: ", ridge_regressor.best_score_)   # no difference
print("\n")
prediction_ridge=ridge_regressor.predict(X_test)
rr = Ridge(alpha = 1)
rr.fit(X_train, y_train)
pred_train_rr= rr.predict(X_train)
print("Ridge: Train RMSE" , np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print("Ridge: Train R-squared" , r2_score(y_train, pred_train_rr))
pred_test_rr= rr.predict(X_test)
print("Ridge: Test RMSE" , np.sqrt(mean_squared_error(y_test,pred_test_rr))) 
print("Ridge: Test R-squared", r2_score(y_test, pred_test_rr))



import seaborn as sns
sns.distplot(y_test-prediction_ridge)

print("====================================")

# LASSO Regression
print("LASSO REGRESSION: ")
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
print("LASSO best Lambda(Tuning Parameter): ",lasso_regressor.best_params_)   # It did not help much
print("LASSO best mean_mse: ",lasso_regressor.best_score_)    
prediction_lasso=lasso_regressor.predict(X_test)

import seaborn as sns
sns.distplot(y_test-prediction_lasso)

model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train, y_train)
pred_train_lasso= model_lasso.predict(X_train)
print("\n")
print("LASSO REGRESSION: ")
print("LASSO: Train RMSE" ,np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print("LASSO: Train R-squared" ,r2_score(y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(X_test)
print("LASSO: Test RMSE" ,np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print("LASSO: Test R-squared" , r2_score(y_test, pred_test_lasso))


#########################################################################################
#Binary Classification Logistic Regression
# We want to see if by latitude longitude room_type minimum_nights reviews_per_month calculated_host_listings_count availability_365,
# can we predict if the property is the entire house or only a room/ shared room. In this setting, we use logistic regression to predict room type.
# 1 is the entire house and 0 is room or shared room
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.model_selection import train_test_split

col_list_LogReg = ["latitude","longitude" ,"room_type" , "minimum_nights", "reviews_per_month" , "calculated_host_listings_count" ,"availability_365"  ]
df_LogReg = pd.read_csv('./AB_NYC_2019.csv', usecols = col_list_LogReg)

df_LogReg["reviews_per_month"].fillna(0, inplace = True)

df_LogReg['Is house/apt'] = np.where(df_LogReg['room_type']== 'Entire home/apt', 1, 0)

dataset_LogReg = df_LogReg[[ "latitude","longitude" ,"minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365" , "Is house/apt"]]


#sns.countplot( x= "Is house/apt" , data = dataset_LogReg)

plt.clf()
dataset_LogReg["minimum_nights"].plot.hist(bins = 5)
plt.clf()
sns.countplot( x= "Is house/apt" , data = dataset_LogReg)
sns.boxplot(x= "Is house/apt" , y = "minimum_nights" , data = dataset_LogReg )
sns.boxplot(x= "Is house/apt" , y = "reviews_per_month" , data = dataset_LogReg )


X_LogReg = dataset_LogReg.drop("Is house/apt" , axis =1)
y_LogReg = dataset_LogReg["Is house/apt"]

#from sklearn.cross_validation import train_test_split

X_train_LR , X_test_LR, y_train_LR, y_test_LR = train_test_split(X_LogReg, y_LogReg, test_size = 0.50, random_state = 1)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train_LR, y_train_LR)
predictions_LR =  logmodel.predict(X_test_LR)

from sklearn.metrics import classification_report

classification_report(y_test_LR, predictions_LR)

print("\n Logistic Regression classification Report: ")
print(classification_report(y_test_LR, predictions_LR))

from sklearn.metrics import confusion_matrix                                    # Import confusion matrix
confusion_matrix(y_test_LR, predictions_LR)                                     # [0,0]True negative  [0,1]False Positive
print("\n Logistic Regression Confusion Matrix:")                                                        # [1,0]False Negative [1,1]True Positive
print(confusion_matrix(y_test_LR, predictions_LR))

from sklearn.metrics import accuracy_score
accuracy_score(y_test_LR, predictions_LR)
print("\n Logistic Regression Accuracy Score: ")
print(accuracy_score(y_test_LR, predictions_LR))
print("\n")

##### RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
np.random.seed(0)
clf = RandomForestClassifier(n_jobs = 2, random_state=0)
clf.fit(X_train_LR, y_train_LR)  
clf.predict(X_test_LR)
preds_ishouse = clf.predict(X_test_LR)
print("\n")
print("Is it a House/Apt? Confusion Matrix Based on Random Forest Classifier")
print(pd.crosstab(y_test_LR, preds_ishouse, rownames = ["Actual"], colnames = ["Predicted"]))   #Create Confusion Matrix 
print("\n")
print("True Negative[1,1]   False Positive[1,2] ")
print("False Negative[2,1]  True Positive[2,2] ")
print("Is it a House/Apt? Confusion Matrix (Standardized) Based on Random Forest Classifier")
print(pd.crosstab(y_test_LR, preds_ishouse, rownames = ["Actual"], colnames = ["Predicted"] , normalize=True))


###########################################################
## Bar charts by Ei  11.29.2020
# the chart for average price by neighborhood in Manhattan doesn't seem to make sense. I added distribution of room type instead.

boroughs_list = list(set(map(lambda x: x.neighbourhood_group, data)))
neighbourhoods = list(set(map(lambda x: x.neighbourhood, data)))
room_t = list(set(map(lambda x: x.room_type, data)))

bor = []
y_avg = []
y_listing = []
y_ava = []
for b in boroughs_list:
    b_filtered = list(filter(lambda x:x.neighbourhood_group == b,data))
    bor.append(b)
    price_lst = list(map(lambda x: x.price, b_filtered))
    listing = list(map(lambda x: x.property_id, b_filtered))
    ava = list(map(lambda x: x.availability_365, b_filtered))
    avg_b = sum(price_lst)/len(price_lst)
    num_listing = len(listing)
    avg_ava = sum(ava)/len(ava)
    y_avg.append(avg_b)
    y_listing.append(num_listing)
    y_ava.append(avg_ava)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_title("Average price by borough")
ax.bar(bor,y_avg)

fig2 = plt.figure()
ax = fig2.add_axes([0,0,1,1])
ax.set_title("Number of listings by borough")
ax.bar(bor,y_listing)

fig3 = plt.figure()
ax = fig3.add_axes([0,0,1,1])
ax.set_title("Average availability by borough")
ax.bar(bor,y_ava)
plt.show()

x_room = []
y_room = []
for r in room_t:
    r_filtered = list(filter(lambda x:x.room_type == r,data))
    x_room.append(r)
    listing = list(map(lambda x: x.property_id, r_filtered))
    rlst = len(listing)
    y_room.append(rlst)

fig4 = plt.figure()
ax = fig4.add_axes([0,0,1,1])
ax.set_title("Distribution by room type")
ax.bar(x_room,y_room)
plt.show()

###########################################################
#Behzad 11.29.2020
#Random Forest for Only Manhattan Only Private room
## if the room price is more expensive than the mean, the value is 1 otherwise it is 0
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
col_list_rf = [ "neighbourhood_group","room_type", "price" , "minimum_nights", "reviews_per_month" , "calculated_host_listings_count" ,"availability_365"  ]
df_rf = pd.read_csv('./AB_NYC_2019.csv', usecols = col_list_rf)

df_rf = df_rf[(df_rf["price"] > 0) & (df_rf["room_type"] == "Private room") & (df_rf["neighbourhood_group"] == "Manhattan") ]  # only select entire houses/ apartments
df_rf["reviews_per_month"].fillna(0, inplace = True)
df_rf.loc[df_rf['availability_365'] > 365, 'availability_365'] = 365

df_rf_p_mean = df_rf["price"].mean()
print("Price mean private room in Manhattan", df_rf_p_mean)

df_rf['Is expensive'] = np.where(df_rf['price'] > df_rf_p_mean, 1, 0)
dataset_rf = df_rf[[ "neighbourhood_group" ,"minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365" , "price", "room_type", "Is expensive"] ]

from sklearn.ensemble import RandomForestClassifier

sns.countplot( x= "Is expensive" , data = dataset_rf)

np.random.seed(0)

dataset_rf["is_train"] = np.random.uniform(0,1, len(dataset_rf)) <= 0.70

#print(dataset_rf["is_train"])

# create test row and train row
train_rf, test_rf = dataset_rf[dataset_rf["is_train"] == True], dataset_rf[dataset_rf["is_train"] == False]

print("# observation in train: ", len(train_rf))
print("# observation in test: " ,len(test_rf))
#features_rf = df_rf[[ "minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365" ] ]
#print("features_rf", features_rf)

train_rf["Is expensive"] = train_rf["Is expensive"].astype("bool")
test_rf["Is expensive"] = test_rf["Is expensive"].astype("bool")
y_rf= train_rf["Is expensive"] 

#print("\n",type(y_rf) , "\n")
clf = RandomForestClassifier(n_jobs = 2, random_state=0)  # Creating a random forest Classifier                      
clf.fit(train_rf[["minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365"]], y_rf)                         # training the classifier                
clf.predict(test_rf[["minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365"]])
#print(clf.predict_proba(test_rf[["minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365"]])[0:10])

preds_rf = clf.predict(test_rf[["minimum_nights", "reviews_per_month", "calculated_host_listings_count", "availability_365"]])
print("\n")
print("Confusion Matrix Based on Random Forest Classifier")
print(pd.crosstab(test_rf["Is expensive"], preds_rf, rownames = ["Actual"], colnames = ["Predicted"]))   #Create Confusion Matrix 
print("\n")
print("True Negative[1,1]   False Positive[1,2] ")
print("False Negative[2,1]  True Positive[2,2] ")
print(pd.crosstab(test_rf["Is expensive"], preds_rf, rownames = ["Actual"], colnames = ["Predicted"] , normalize=True))
##################################################

##Mostafa Salem 11/30/2020
##Bar Charts on averages of price, minimum nights, availability, and reviews by room type
room_types = []
p_avg = []
n_avg = []
a_avg = []
r_avg = []
room_t = list(set(map(lambda x: x.room_type, data)))
for r in room_t:
    room_filt = list(filter(lambda x:x.room_type == r, data))
    room_types.append(r)
    minnights_filt = list(map(lambda x:x.minimum_nights, room_filt))
    price_filt = list(map(lambda x: x.price, room_filt))
    reviews_filt = list(map(lambda x:x.reviews_per_month, room_filt))
    available_filt = list(map(lambda x: x.availability_365, room_filt))
    price_avg = sum(price_filt)/len(price_filt)
    mn_avg = sum(minnights_filt)/len(minnights_filt)
    avail_avg = sum(available_filt)/len(available_filt)
    review_avg = sum(reviews_filt)/len(reviews_filt)
    p_avg.append(price_avg)
    n_avg.append(mn_avg)
    a_avg.append(avail_avg)
    r_avg.append(review_avg)
    
    
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_title("Average Price by Room Type")
ax.bar(room_types,p_avg)

fig2 = plt.figure()
ax = fig2.add_axes([0,0,1,1])
ax.set_title("Average Minimum Nights by Room Type")
ax.bar(room_types,n_avg)
plt.show()

fig3 = plt.figure()
ax = fig3.add_axes([0,0,1,1])
ax.set_title("Average Availability by Room Type")
ax.bar(room_types, a_avg)

fig4 = plt.figure()
ax = fig4.add_axes([0,0,1,1])
ax.set_title("Average Reviews by Room Type")
ax.bar(room_types, r_avg)

#############################################################################################
##Jesina Dangol 11/30/2020
##Display Top 10 records for highest and lowest price from user desired input neighborhoods
##Display chart for these min & max prices correlation with minimum nights
"""
import pandas as pd

col_list = [ "neighbourhood_group","neighbourhood", "price","minimum_nights" ]
df = pd.read_csv('./AB_NYC_2019.csv', usecols = col_list)

def Display(lst):

    if len(ff) == 0:
        print("hmmm...cannot find that neighbourhood.")
    else:
        print("\n")
        boroughs = list(set((ff["neighbourhood_group"])))
        price_set = list(set(ff["price"]))
        minval = price_set
        price_set.sort()
        print("Highest Prices from Neighbourhood", check,',',boroughs[0])
        for i in range (10):
            print(i+1,'\t','$',price_set[len(price_set)-(1 + i)] )
        print("\n")
    
        print("Lowest Prices from Neighbourhood", check,',',boroughs[0])
        for i in range (10):
            print(i+1,'\t','$',minval[i] )

            
        import matplotlib.pyplot as plt   
        ff.groupby("minimum_nights").min().plot.line()
        plt.legend(['Minimum prices'])
        plt.show()
        ff.groupby("minimum_nights").max().plot.line()
        plt.legend(['Maximum prices'])
        plt.show()
        
        
#Getting neighbourhood input from user

check = input("Neighbourhood: ")
while check != 'Exit':
    ff = df[(df.neighbourhood  == check) & (df["price"] > 0)]
    Display(ff)
    check = input("Neighbourhood: ") 
    
print("\n")
print("Thank you for choosing us!")
"""
##################################################################################
