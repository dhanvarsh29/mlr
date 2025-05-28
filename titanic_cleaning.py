import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\AIML\titanic.csv.csv")


print(df.head())
print(df.info())        
print(df.describe())    

# Fill missing numerical values with mean or median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing categorical values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop columns with too many missing values (optional)
df.drop(columns=['Cabin'], inplace=True)
# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
from sklearn.preprocessing import StandardScaler


num_features = ['Age', 'Fare']

scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])



sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()


Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Age'] >= Q1 - 1.5 * IQR) & (df['Age'] <= Q3 + 1.5 * IQR)]
