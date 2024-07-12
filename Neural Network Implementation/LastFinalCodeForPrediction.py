import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from keras.callbacks import History 

# Load the data into a pandas DataFrame
file_path="C:\\Users\\tahak\\Desktop\\TWEEDE JAAR\\tweede semester\\practise enterprise 2\\third spring\\codes and dataset\\dataset(csv files)\\DatasetWithOutput.csv"
df = pd.read_csv(file_path)

#I have worked on dataset and since they have no impact on output, I dropped these features.
df = df.drop(['Day of the week','Number of layovers'], axis=1)

# Check if any values in the row are NaN or empty
for index, row in df.iterrows():
    if row.isna().any() or row.empty:
        print(f"Row {index} contains at least one NaN or empty value")

# identify columns with constant values
constant_cols = [col for col in df.columns if df[col].nunique() == 1]

# print the constant columns
print("Columns with constant values:", constant_cols)

#to see redundant columns
redundant_cols = []
for i in range(len(df.columns)):
    col1 = df.iloc[:, i]
    for j in range(i+1, len(df.columns)):
        col2 = df.iloc[:, j]
        if col1.equals(col2):
            redundant_cols.append(df.columns[j])
# Drop the redundant columns
df = df.drop(redundant_cols, axis=1)
print("Dropped redundant columns:", redundant_cols)

corr_matrix = df.corr(numeric_only=True)
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_ratio = corr_matrix.iloc[i, j]
        #print(f"The correlation ratio between '{corr_matrix.columns[i]}' and '{corr_matrix.columns[j]}' is {corr_ratio:.2f}")
#According to correlation matrix, there is almost no correlation between package value and output, That's why I also dropped package value.
df = df.drop(['Package value'], axis=1)

#to avoid multicollinearity 
df = df.drop(['Discounts or promotions','Storage cost'], axis=1)

""" for not to cut python code,  take this picture in comment
#to show the correlation picture
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
"""

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(exclude=['object']).columns

# Encode categorical columns for model-based feature importance
le = LabelEncoder()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Split data into features (X) and target (y)
X = df_encoded.drop('Output', axis=1)
y = df_encoded['Output']

# Fit a RandomForestRegressor to get feature importances
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)

# Get feature importances
importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
importances = importances.sort_values('importance', ascending=False)

# Get least important categorical and numerical features
least_important_categorical = importances[importances['feature'].isin(categorical_cols)].tail(4)['feature'].tolist()
least_important_numerical = importances[importances['feature'].isin(numerical_cols)].tail(3)['feature'].tolist()

#Drop least important features(features which are dropped:
df = df.drop(least_important_categorical + least_important_numerical, axis=1)


categorical_cols = [col for col in categorical_cols if col not in least_important_categorical]

# Print dropped features
print("Dropped least important categorical features:", least_important_categorical)
print("Dropped least important numerical features:", least_important_numerical)

#I also dropped these features because available capacity and flight capacity are almost same features. 
#These both features can be combined into one feature.
# But, according to circumstances and our domain knowledge, this is not possible right now
df = df.drop(['Available capacity','Economic conditions','Route popularity','Market competition'], axis=1)
numerical_cols = [col for col in numerical_cols if col not in ['Available capacity', 'Economic conditions', 'Route popularity', 'Market competition']]
categorical_cols = [col for col in categorical_cols if col not in ['Route popularity', 'Market competition','Economic conditions']]
print('------column names after feature engineering-----')
column_names = df.columns.tolist()
print(column_names)

# Update numerical_cols to remove the dropped columns
numerical_cols = [col for col in numerical_cols if col not in least_important_numerical]

# Apply MinMaxScaler to the numerical columns
scaler = MinMaxScaler()
numerical_cols = [col for col in numerical_cols if col != 'Output']  # Add this line
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Output', axis=1), df['Output'], test_size=0.3, random_state=42)

# Encode categorical features using the LabelEncoder
le = LabelEncoder()
for col in categorical_cols:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Create a neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Convert your input data to NumPy arrays
X_train_np = X_train.to_numpy().astype(np.float32)
y_train_np = y_train.to_numpy().astype(np.float32)

# Train the model
history=model.fit(X_train_np, y_train_np, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

print(history.history.keys())

"""
# And then plot the learning curve after the training:
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
"""
# Convert the test data to NumPy arrays
X_test_np = X_test.to_numpy().astype(np.float32)
y_test_np = y_test.to_numpy().astype(np.float32)

# Evaluate the model on the test data
test_loss = model.evaluate(X_test_np, y_test_np, verbose=1)
test_predictions = model.predict(X_test_np)

# Convert the predicted values back to the original scale
y_pred = test_predictions.flatten()

# Calculate the Mean Absolute Error
mae = mean_absolute_error(y_test_np, y_pred)
print(f"Mean Absolute Error (Test Set): {mae:.2f}")

# Calculate the R2 Score
r2 = r2_score(y_test_np, y_pred)
print(f"R-squared (Test Set): {r2:.2f}")
"""
# Visualize the errors
plt.figure(figsize=(14, 8))
error = y_test_np - y_pred
sns.histplot(error, bins=25, kde=True)
plt.xlabel('Prediction Error')
_ = plt.ylabel('Count')
plt.show()
"""
# Function to preprocess user input
def preprocess_user_input(user_input, le_dict, numerical_cols, scaler):
    user_input_df = pd.DataFrame(user_input, index=[0])
    
    # Remove 'Output' from numerical_cols
    numerical_cols = [col for col in numerical_cols if col != 'Output']
    
    # Apply MinMaxScaler to the numerical columns
    user_input_df[numerical_cols] = scaler.transform(user_input_df[numerical_cols])
    
    # Encode categorical features using the LabelEncoder dictionary
    for col, le in le_dict.items():
        user_input_df[col] = le.transform(user_input_df[col])

    return user_input_df
go=0
while go<1 : 
# Get user input and preprocess
    distance = None
    while distance is None or not (100 <= distance <= 10000):
        try:
            distance = float(input("Enter Distance (between 100 and 10000): "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    cargo_weight = None
    while cargo_weight is None or not (100 <= cargo_weight <= 100000):
        try:
            cargo_weight = float(input("Enter Cargo weight (between 100 and 100000): "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    cargo_volume = None
    while cargo_volume is None or not (100 <= cargo_volume <= 100000):
        try:
            cargo_volume = float(input("Enter Cargo volume (between 100 and 100000): "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")     
    valid_cargo_types = ['perishable goods', 'general cargo', 'hazardous materials']
    cargo_type = None
    while cargo_type is None or cargo_type not in valid_cargo_types:
        cargo_type = input("Enter Cargo type (perishable goods, general cargo, hazardous materials): ")
        if cargo_type not in valid_cargo_types:
            print("Invalid input. Please enter one of the valid cargo types.")
    flight_capacity = None
    while flight_capacity is None or not (100 <= flight_capacity <= 1000):
        try:
            flight_capacity = float(input("Enter Flight capacity (between 100 and 1000): "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    fuel_price = None
    while fuel_price is None or not (20 <= fuel_price <= 200):
        try:
            fuel_price = float(input("Enter Fuel price (between 20 and 200): "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    insurance_cost = None
    while insurance_cost is None or not (1 <= insurance_cost <= 5):
        try:
            insurance_cost = float(input("Enter Insurance cost (between 1 and 5): "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    valid_cargo_fragility = ['low', 'medium', 'high']   
    cargo_fragility = None
    while cargo_fragility is None or cargo_fragility not in valid_cargo_fragility:
        cargo_fragility = input("Enter Cargo fragility (low, medium, high): ")
        if cargo_fragility not in valid_cargo_fragility:
            print("Invalid input. Please enter one of the valid cargo fragility levels.")   
    valid_urgency = ['low', 'medium', 'high']
    urgency = None
    while urgency is None or urgency not in valid_urgency:
        urgency = input("Enter Urgency (low, medium, high): ")
        if urgency not in valid_urgency:
            print("Invalid input. Please enter one of the valid urgency levels.")                                       
    user_input = {
        'Distance': distance,
        'Cargo weight': cargo_weight,
        'Cargo volume': cargo_volume,
        'Cargo type': cargo_type,
        'Flight capacity': flight_capacity,
        'Fuel price': fuel_price,
        'Insurance cost': insurance_cost,
        'Cargo fragility': cargo_fragility,
        'Urgency': urgency
    }
    # Create a LabelEncoder dictionary to store the trained LabelEncoders
    le_dict = {}
    for col in categorical_cols:
      le_dict[col] = LabelEncoder()
      le_dict[col].fit(df[col])



    preprocessed_user_input = preprocess_user_input(user_input, le_dict, numerical_cols, scaler)

    # Predict price for the user input
    predicted_price = model.predict(preprocessed_user_input)

    # Display the predicted price
    print("Predicted Price:", predicted_price[0][0])
    go=int(input("If you want to finish press 1 otherwise 0: "))

