import csv
import random
import pandas as pd
import numpy as np

# Function to generate a value with the desired correlation to the output
def generate_value_with_correlation(corr, output, min_val, max_val):
    random_value = random.uniform(min_val, max_val)
    target_value = output * corr + random_value * (1 - corr)
    value = np.clip(target_value, min_val, max_val)
    return value if random.random() > 0.1 else random_value

# Set the number of rows
num_rows = 30000

# Define the list of features and their data types
features = [
    ("Distance", "numerical", (100, 10000)),
    ("Cargo weight", "numerical", (100, 100000)),
    ("Cargo volume", "numerical", (100, 100000)),
    ("Cargo type", "categorical", ["perishable goods", "hazardous materials", "general cargo"]),
    ("Flight capacity", "numerical", (100, 1000)),
    ("Available capacity", "numerical", (100, 1000)),
    ("Season", "categorical", ["high season", "low season"]),
    ("Day of the week", "categorical", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
    ("Number of layovers", "numerical", (0, 7)),
    ("Fuel price", "numerical", (20, 200)),
    ("Market competition", "ordinal", ["low", "medium", "high"]),
    ("Weather conditions", "ordinal", ["good", "moderate", "bad"]),
    ("Economic conditions", "ordinal", ["weak", "moderate", "strong"]),
    ("Package value", "numerical", (20, 500)),
    ("Insurance cost", "numerical", (1, 5)),
    ("Taxes and fees", "numerical", (1, 10)),
    ("Loading/unloading time", "numerical", (1, 15)),
    ("Handling cost", "numerical", (3, 25)),
    ("Storage cost", "numerical", (3, 25)),
    ("Route popularity", "ordinal", ["low", "medium", "high"]),
    ("Cargo fragility", "ordinal", ["low", "medium", "high"]),
    ("Urgency", "ordinal", ["low", "medium", "high"]),
    ("Special handling requirements", "ordinal", ["none", "minor", "major"]),
    ("Discounts or promotions", "numerical", (0, 100)),
    ("Contract type", "categorical", ["individual", "courier service", "corporate"])
]

# Open a CSV file for writing
with open("last.csv", "w", newline="") as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Write the header row
    header_row = [f[0] for f in features]
    writer.writerow(header_row)

    # Set the desired correlations
    desired_corr = {
        'Distance': 0.78,
        'Cargo weight': 0.52,
        'Cargo volume': 0.45,
        'Flight capacity': 0.1,
        'Available capacity': 0.1,
        'Number of layovers': 0.1,
        'Fuel price': 0.27,
        'Package value': 0.01,
        'Insurance cost': 0.04,
        'Taxes and fees': 0.1,
        'Loading/unloading time': 0.1,
        'Handling cost': 0.12,
        'Discounts or promotions': 0.23,
        'Storage cost':0.1
        # Add any other desired correlations here
    }

    # Initialize output_values list
    output_values = []

    # Write the data rows
    for i in range(num_rows):
        # First, generate the output value
        output = random.uniform(-1000, 1000)

        row = []
        for f in features:
            if f[0] in desired_corr:
                # Generate a value with the desired correlation to the output
                value = generate_value_with_correlation(desired_corr[f[0]], output, f[2][0], f[2][1])
            else:
                if f[1] == "numerical":
                    # Generate a random numerical value within the specified range
                    value = random.uniform(f[2][0], f[2][1])
                elif f[1] == "categorical":
                    # Pick a random categorical value
                    value = random.choice(f[2])
                elif f[1] == "ordinal":
                    # Pick a random ordinal value
                    value = random.choice(f[2])

            if f[1] == "numerical" and f[0] != "Number of layovers":
                row.append(value)
            elif f[1] == "numerical" and f[0] == "Number of layovers":
                row.append(int(value))
            else:
                row.append(value)

        output_values.append(output)
        writer.writerow(row)

print("Dataset created successfully.")

def calculate_output(row):
    # Convert ordinal features to numerical values
    cargo_fragility_map = {'low': 0, 'medium': 1, 'high': 2}
    urgency_map = {'low': 0, 'medium': 1, 'high': 2}
    special_handling_map = {'none': 0, 'minor': 1, 'major': 2}
    row['Cargo fragility'] = cargo_fragility_map[row['Cargo fragility']]
    row['Urgency'] = urgency_map[row['Urgency']]
    row['Special handling requirements'] = special_handling_map[row['Special handling requirements']]

    # Most important features
    output = row['Distance'] * 0.15 + row['Cargo weight'] * 0.002 + row['Cargo volume'] * 0.001 \
             - row['Fuel price'] * 0.01 - row['Cargo fragility'] * 0.1 - row['Urgency'] * 0.1 \
             - row['Special handling requirements'] * 0.1

    # Second most important features
    output += row['Handling cost'] * 0.05 + row['Flight capacity'] * 0.001 + row['Loading/unloading time'] * 0.05 \
              - (row['Contract type'] == "courier service") * 0.01 + (row['Market competition'] == "medium") * 0.5 \
              - (row['Weather conditions'] == "moderate") * 0.5 - (row['Economic conditions'] == "moderate") * 0.5

    # Third important features
    output += row['Available capacity'] * 0.001 + row['Package value'] * 0.0001 - row['Insurance cost'] * 0.05 \
              - row['Taxes and fees'] * 0.05 - row['Storage cost'] * 0.05 \
              + (row['Season'] == "low season") * 0.01 + (row['Route popularity'] == "medium") * 0.1

    # Least important features
    output += row['Discounts or promotions'] * 0.001 + row['Number of layovers'] * 0.001 \
              + (row['Day of the week'] == "Tuesday") * 0.001
    
    output = max(output, 0)

    return output



# Read the generated dataset with input features
dataset = pd.read_csv("last.csv")

# Calculate output values for each row
output_values = []
for index, row in dataset.iterrows():
    output = calculate_output(row)

    # Introduce a 2% probability of an outlier
    if random.random() < 0.02:
        outlier_multiplier = random.uniform(1.5, 2.5)
        output *= outlier_multiplier

    output_values.append(output)

# Add the output values as a new column to the dataset
dataset['Output'] = output_values

# Save the updated dataset with output values to a new CSV file
dataset.to_csv("LASTDATA.csv", index=False)

print("Output column added successfully.")
