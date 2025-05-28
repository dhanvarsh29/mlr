import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset 
df = pd.read_csv(r"C:\AIML\titanic.csv.csv")
 # Corrected filename


# Basic info and data types
print(df.info())

# Statistical summary for numeric features
print(df.describe())

# Summary for categorical columns
print(df.describe(include=['object', 'category']))
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')

plt.subplot(1,2,2)
sns.boxplot(x=df['Age'])
plt.title('Age Boxplot')

plt.show()
# Countplot of 'Survived'
sns.countplot(data=df, x='Survived')
plt.title('Survival Counts')
plt.show()

# Countplot of 'Pclass' (Passenger Class)
sns.countplot(data=df, x='Pclass')
plt.title('Passenger Class Distribution')
plt.show()
# Correlation heatmap (only numeric features)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot (for numeric features and 'Survived' as hue)
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']].dropna(), hue='Survived')
plt.show()
fig = px.scatter(df, x='Age', y='Fare', color='Survived',
                 title='Age vs Fare colored by Survival',
                 hover_data=['Name', 'Pclass'])
fig.show()
