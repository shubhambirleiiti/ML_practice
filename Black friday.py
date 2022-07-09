import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_train=pd.read_csv (r"C:\Users\Asus\OneDrive - IIT Kanpur\Desktop\train.csv")
print(df_train .head().to_string())
print(df_train .columns )
print(df_train .info())
print(df_train .describe().to_string() )
print(df_train .corr().to_string() )
#sns.heatmap(df.corr(),annot=True)
#plt.show()
#plt.bar(df["Occupation"],df["Marital_Status"])
#plt.show()
df_test=pd.read_csv (r"C:\Users\Asus\OneDrive - IIT Kanpur\Desktop\test.csv")
print(df_test )
final_data=df_train.append(df_test )
print(final_data.shape )
####data preprossing starts
"""del final_data["User_ID"]
print(final_data .shape)"""
final_data.drop("User_ID",axis=1,inplace=True)
print(final_data.shape)
print(final_data["Gender"].value_counts() )
final_data["Gender"]=final_data["Gender"].map({"M":1,"F":0})
print(final_data ["Gender"].head())
print(final_data .info())
print(final_data["Age"].value_counts() )
print(final_data['Age'].unique())
final_data["Age"]=final_data["Age"].map({"0-17":1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})
print(final_data["Age"])
print(final_data['City_Category'].unique())
df_city=pd.get_dummies(final_data ["City_Category"])
print(df_city)
final_data=pd.concat([final_data,df_city],axis=1)
print(final_data.shape)
del final_data["City_Category"]
print(final_data.shape)
print(final_data.info())
print(final_data['Stay_In_Current_City_Years'].unique())
final_data["Stay_In_Current_City_Years"]=final_data["Stay_In_Current_City_Years"].str.replace("+","")
print(final_data.head().to_string() )
final_data["Stay_In_Current_City_Years"]=final_data["Stay_In_Current_City_Years"].astype(int)
print(final_data.info())
print(final_data.isnull().sum())
print(final_data["Product_Category_2"].value_counts() )
x=final_data["Product_Category_2"].mode()[0]
print(x)
final_data["Product_Category_2"].fillna(x,inplace=True)
print(final_data.isnull().sum())
y=final_data["Product_Category_3"].mode()[0]
print(y)
final_data["Product_Category_3"].fillna(y,inplace=True)
print(final_data.isnull().sum())
final_data["A"]=final_data["A"].astype(int)
final_data["B"]=final_data["B"].astype(int)
final_data["C"]=final_data["C"].astype(int)
del final_data["Product_ID"]
print(final_data.info())
#sns.barplot("Age","Purchase",hue="Gender",data=final_data)
#print(plt.show())
final_test=final_data[final_data["Purchase"].isnull()]
final_train=final_data[~final_data["Purchase"].isnull()]
print(final_test.head())
print(final_train.head())
X=final_train.drop("Purchase",axis=1)
y=final_train["Purchase"]
print(X)
print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size= .8,random_state= 40)
"""from sklearn.linear_model import LinearRegression
lin_regr=LinearRegression ()
lin_regr.fit(X_train,y_train)
y_predicted=lin_regr.predict(X_test)
y_pred=pd.DataFrame (y_predicted )
print(y_test.head())
print(y_pred.head())"""
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor ()
tree.fit(X_train ,y_train )
y_predicted=tree.predict(X_test)
y_pred=pd.DataFrame (y_predicted )
#tree.fit(X_train ,y_train )
print(y_test.head())
print(y_pred.head())
"""from sklearn .model_selection import cross_val_score
mse=cross_val_score(lin_regr,y_test,y_pred,scoring= "neg_mean_squared_error",cv=5)
print(mse)"""
"""final_data["City_Category"]=final_data["City_Category"].map({"A":1,"B":2,"C":3})
print(final_data["City_Category"])"""