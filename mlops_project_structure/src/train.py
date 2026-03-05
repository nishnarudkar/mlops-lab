from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

data = load_iris()

X = data.data
y = data.target

model = RandomForestClassifier()
model.fit(X, y)

print("Model trained successfully")
