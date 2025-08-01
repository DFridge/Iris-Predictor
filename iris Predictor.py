import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

#Loading the dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(r'C:\Pyt Files\Projects\iris Predictor\Data\iris.data', header=None, names=column_names)

# Taking input from the user
print("Welcome to the Iris Predictor!")

print("Please Enter the following details:")
sepal_length = float(input("Sepal Length (cm): "))
sepal_width = float(input("Sepal Width (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width = float(input("Petal Width (cm): "))

user_input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Preparing the data labels
encoder = LabelEncoder()
df['class_encoded'] = encoder.fit_transform(df['class'])

# Splitting the dataset into training and testing sets
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
Y = df['class_encoded']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Training the KNN model
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, Y_train)

# Predict and evaluating the model
user_prediction = KNN.predict(user_input_df)
predicted_class = encoder.inverse_transform(user_prediction)[0]
print(f"\nPredicted Class for User Input: {predicted_class}")

# Showing accuracy of the model
Y_pred = KNN.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Visualizing the data
df['class_name'] = encoder.inverse_transform(df['class_encoded'])
plt.figure(figsize=(10, 6))
sns.scatterplot(data = df, x = 'petal_length', y = 'petal_width', hue = 'class_name', palette = 'Set2' , s = 70)
plt.scatter(petal_length, petal_width, color='black', marker = 'X', s = 200, label = 'Your Flower')

plt.title('Iris Flower Classification')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.tight_layout()
plt.show()