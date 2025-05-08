# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Start

2.Import numpy, pandas, StandardScaler

3.Add bias, initialize theta, update iteratively

4.Load 50_Startups.csv

5.Extract and convert features (X) and target (y)

6.Normalize X and y

7.Train model using gradient descent → get theta

8.Normalize new input, predict, inverse scale result

9.Print prediction

10.End


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Deepak.R
RegisterNumber:  212223040031
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())

# Assuming the last column is your target variable 'y' and the preceding columns a
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

#Learn model parameters
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![Screenshot 2025-03-06 212207](https://github.com/user-attachments/assets/11621de8-fd06-425a-a223-66ada2035fcb)

![Screenshot 2025-03-06 212428](https://github.com/user-attachments/assets/eddaa17c-2924-49af-a9ca-265bb4dd7296)

![Screenshot 2025-03-06 212439](https://github.com/user-attachments/assets/4fb96338-d31c-4930-9d61-6b06848da02e)

![Screenshot 2025-03-06 212451](https://github.com/user-attachments/assets/003b76a4-7d91-4e3f-91f4-4a858c1ea2cc)

![Screenshot 2025-03-06 212458](https://github.com/user-attachments/assets/4c532290-1fc3-40bf-994b-6b8c98089e84)

![Screenshot 2025-03-06 212509](https://github.com/user-attachments/assets/f392904f-4a76-485f-b795-5f06b0df5ff1)





![Screenshot 2025-03-06 212516](https://github.com/user-attachments/assets/3f919944-57f5-41d5-a6d9-c5c1f67662ae)

![Screenshot 2025-03-06 212524](https://github.com/user-attachments/assets/fafbfea8-7cb5-40e4-8f7f-27beb759d555)

![Screenshot 2025-03-06 212531](https://github.com/user-attachments/assets/76136852-912c-4e45-8134-dc706f019e1a)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
