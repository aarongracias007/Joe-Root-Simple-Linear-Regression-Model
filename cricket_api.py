import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

#Joe Root innings by innings file path
file_path=r"C:\Users\A_Gracias\OneDrive - Iskraemeco, d.d\Documents\Personal\ML Simple Linear Regression Project\Joe Root Test Runs innings by innings.xlsx"
#jrd is Joe Root Data
jrd=pd.read_excel(file_path)
jrd["Runs"]=jrd["Runs"].astype(str)
jrd["Runs"]=jrd["Runs"].str.replace("*","",regex=False)
jrd=jrd[jrd["Runs"]!="DNB"]
jrd["Runs"]=jrd["Runs"].astype(int)
jrd["Cumulative_Runs"]=jrd["Runs"].cumsum()
jrd["Innings"]=range(1,len(jrd)+1)
x=jrd[["Innings"]]
y=jrd["Cumulative_Runs"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
innings_range = pd.DataFrame(range(1, 401), columns=["Innings"])
predicted_runs = model.predict(innings_range)
plt.figure(figsize=(10,6))
plt.scatter(jrd['Innings'], jrd['Cumulative_Runs'], color='blue', label='Actual Data')
plt.plot(innings_range, predicted_runs, color='red', label='Linear Regression Line')
plt.axhline(y=15921, color='green', linestyle='--', label="Sachin's Total Runs (15,921)")
plt.legend()
plt.xlabel('Innings')
plt.ylabel('Cumulative Runs')
plt.title('Joe Root Cumulative Runs Prediction')
plt.show()