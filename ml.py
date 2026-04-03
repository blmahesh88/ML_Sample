import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (assuming a CSV file like 'loan_data.csv' with columns: age, income, credit_score, loan_amount, approved)
# Replace with actual dataset path
df = pd.read_csv('loan_data.csv')

# Preprocessing
le = LabelEncoder()
df['approved'] = le.fit_transform(df['approved'])  # Assuming 'approved' is categorical

X = df.drop('approved', axis=1)
y = df['approved']

# Handle categorical features if any (e.g., gender)
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))