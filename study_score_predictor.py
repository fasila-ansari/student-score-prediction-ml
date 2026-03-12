import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Scores": [35,40,50,55,65,70,80,90]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Scores"]

model = LinearRegression()
model.fit(X,y)

hours = float(input("Enter number of study hours: "))

prediction = model.predict([[hours]])

print("Predicted score:", prediction[0])
plt.scatter(X,y)
plt.plot(X,model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score Prediction")
plt.show()