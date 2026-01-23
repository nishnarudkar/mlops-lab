import pandas as pd
from sklearn.linear_model import LinearRegression    
X = [[1],[2],[3],[4]]
y = [2,4,6,8] 
model = LinearRegression().fit(X,y)
print("Prediction for 5:", model.predict([[5]])[0])
