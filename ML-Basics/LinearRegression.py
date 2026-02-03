from sklearn.linear_model import LinearRegression
import numpy
X=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y=[30,50,70,90,110,130,150,170,190,210]
model=LinearRegression()
model.fit(X,y)
print(model.predict([[11]]))
print(type(X))
print(type(y))

