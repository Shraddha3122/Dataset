

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
head = 'C:/Users/iTA/Downloads/dataset.csv'
data = pd.read_csv(head)

# Encode the target variable
label_encoder = LabelEncoder()
data['Depression'] = label_encoder.fit_transform(data['Depression'])

# Separate features and target variable
X = data.drop(columns=['Depression', 'SEQN'])  # Dropping 'SEQN' as it seems to be an identifier
y = data['Depression']

# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Handle missing values
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

if not numerical_columns.empty:
    X[numerical_columns] = numerical_imputer.fit_transform(X[numerical_columns])
if not categorical_columns.empty:
    X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])

# Encode categorical features if still present
if not categorical_columns.empty:
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
