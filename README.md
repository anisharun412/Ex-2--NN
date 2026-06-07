<H3>Name : Arunsamy D</H3>
<H3>Register no : 212224240016</H3>
<H3>Experiment No. 2 </H3>

# Implementation of Perceptron for Binary Classification
## AIM:
To implement a perceptron for classification using Python<BR>

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


## ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
## PROGRAM:
### Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


### Load Dataset

```python
df = pd.read_excel("Iris.xlsx")

df.head()
```

### Removing Null Values

```python
print("Before : \n")
print(df.isnull().sum())

for i in df:
    if df[i].dtype in ['float64', 'int64']:
        df[i] = df[i].fillna(df[i].median())

print("\nAfter : \n")
print(df.isnull().sum())
```

### Select Two Classes

```python
df = df[df['species'] != 'Iris-virginica']

print(df['species'].value_counts())
```


### Plot Dataset

```python
X = df.drop('species', axis=1).values
y = df['species'].values

y_binary = np.where(y == 'Iris-setosa', 1, -1)

plt.scatter(
    X[y_binary == 1, 0],
    X[y_binary == 1, 1],
    label='Setosa'
)

plt.scatter(
    X[y_binary == -1, 0],
    X[y_binary == -1, 1],
    label='Versicolor'
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()
```

### 3D Plot of Dataset

```python
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(8,6))

ax = plt.axes(projection='3d')

ax.scatter3D(
    X[y_binary == 1, 0],
    X[y_binary == 1, 1],
    X[y_binary == 1, 2],
    label='Setosa'
)

ax.scatter3D(
    X[y_binary == -1, 0],
    X[y_binary == -1, 1],
    X[y_binary == -1, 2],
    label='Versicolor'
)

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')

ax.set_title('3D Iris Dataset Visualization')

plt.legend()
plt.show()
```


### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

print(X[:5])
```


### Split the Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_binary,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
```


### Perceptron Class

```python
class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.errors = []

    def activation(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):

            error_count = 0

            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias

                y_pred = self.activation(linear_output)

                update = self.lr * (y[idx] - y_pred)

                self.weights += update * x_i
                self.bias += update

                if update != 0:
                    error_count += 1

            self.errors.append(error_count)

    def predict(self, X):

        linear_output = np.dot(X, self.weights) + self.bias

        return self.activation(linear_output)
```


### Train Model

```python
model = Perceptron(
    learning_rate=0.01,
    n_iters=100
)

model.fit(X_train, y_train)
```


### Plot Errors

```python
plt.plot(
    range(1, len(model.errors)+1),
    model.errors,
    marker='o'
)

plt.xlabel("Iterations")
plt.ylabel("Errors")
plt.title("Error vs Iteration")
plt.grid(True)

plt.show()
```


### Predict Test Data

```python
y_pred = model.predict(X_test)

print(y_pred)
```

### Calculate Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy =", accuracy * 100, "%")
```

## OUTPUT:
### Load Dataset

<img width="1750" height="232" alt="image" src="https://github.com/user-attachments/assets/3a663198-7d92-41e7-934a-cdba40e9e31f" />




### Removing Null Values

<img width="1748" height="344" alt="image" src="https://github.com/user-attachments/assets/ba882459-8418-48a4-ab1c-db43e94fa05e" />




### Select Two Classes

<img width="1747" height="103" alt="image" src="https://github.com/user-attachments/assets/affa14a0-9790-433b-9a93-be6359a99653" />




### 2D Scatter Plot of Dataset

<img width="567" height="435" alt="image" src="https://github.com/user-attachments/assets/dad02223-7471-4fc9-a9a6-71f83ba17d5d" />




### 3D Scatter Plot of Dataset

<img width="490" height="506" alt="image" src="https://github.com/user-attachments/assets/21f991ad-c7eb-4b9c-8d8b-c240e08e1001" />




### Feature Scaling

<img width="1745" height="114" alt="image" src="https://github.com/user-attachments/assets/d9e9c20c-7d98-4be6-902c-2e1d81583c21" />




### Train-Test Split


<img width="1750" height="49" alt="image" src="https://github.com/user-attachments/assets/c4cf4277-cb7a-47f2-999f-b69df874258e" />


### Error vs Iteration Plot

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/e3524745-6204-4715-a9fa-9202cb8e1568" />



### Predicted Values
<img width="1747" height="34" alt="image" src="https://github.com/user-attachments/assets/f4ba2c88-56ab-42b8-9ff0-84ebbcb50809" />



### Accuracy

<img width="1752" height="34" alt="image" src="https://github.com/user-attachments/assets/fda6db7f-5486-4881-ad98-ac6ed944d412" />


 

## RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
