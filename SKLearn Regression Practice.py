import sklearn
import pandas as pd 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# Import boston housing prices toy data set 
from sklearn.datasets import load_boston
boston = load_boston()

# See the description of the data set
# print(boston.DESCR)

# Convert it into pandas data frame
bos = pd.DataFrame(boston.data,columns=boston.feature_names)
print(bos.head())

# Add the target column to the data frame
bos['PRICE'] = boston.target

# See the summary statistics of the dataset 
print(bos.describe())

# Split train-test dataset
X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)

# Run linear regression
lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

# Mean Squared Error
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)

# Use correlation matrix to measure the linear relationships 
correlation_matrix = bos.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

# We can see that RM has a strong positive correlation with MEDV (0.7) where as LSTAT has a high negative correlation with MEDV(-0.74)

