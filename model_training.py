# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
df = pd.read_csv('diabetes.csv')

# Prepare features and target
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)        # Transform testing data

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)  # Train the model

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)*100:3f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')