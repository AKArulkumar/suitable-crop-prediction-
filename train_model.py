import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('dataset/tamilnadu_crop_data.csv')

# Function to determine suitability
def determine_suitability(row):
    suitable_conditions = {
        'Alluvial': {'Rice': ['Winter', 'Summer'], 'Groundnut': ['Summer']},
        'Red Soil': {'Sugarcane': ['Summer'], 'Rice': ['Winter'], 'Groundnut': ['Monsoon']},
        'Black Soil': {'Cotton': ['Monsoon'], 'Paddy': ['Monsoon']},
        'Laterite': {'Rubber': ['Winter']},
        'Sandy': {'Cotton': ['Summer']}
    }
    soil_type = row['Soil Type']
    crop = row['Crop']
    season = row['Ideal Season']
    if soil_type in suitable_conditions and crop in suitable_conditions[soil_type]:
        if season in suitable_conditions[soil_type][crop]:
            return 1  # Suitable
    return 0  # Not suitable

# Apply suitability function
data['Suitability'] = data.apply(determine_suitability, axis=1)

# Save the updated dataset
data.to_csv('dataset/tamilnadu_crop_data_with_suitability.csv', index=False)

# Features and target
X = data[['District', 'Soil Type', 'Crop', 'Ideal Season', 'Average Temperature (°C)', 'Average Rainfall (mm)']]
y = data['Suitability']

# Label encode categorical columns
label_encoder = LabelEncoder()

# Use .loc to avoid SettingWithCopyWarning
X.loc[:, 'District'] = label_encoder.fit_transform(X['District'])
X.loc[:, 'Soil Type'] = label_encoder.fit_transform(X['Soil Type'])
X.loc[:, 'Crop'] = label_encoder.fit_transform(X['Crop'])
X.loc[:, 'Ideal Season'] = label_encoder.fit_transform(X['Ideal Season'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the trained model
with open('model/trained_crop_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

print("Model trained and saved as 'trained_crop_model.pkl'.")
