import pandas as pd

# Read the CSV file into a DataFrame
file_path = 'train_vec.csv'
df = pd.read_csv(file_path)

# Save the header separately
header = df.iloc[0]

# Shuffle the rows (excluding the header)
df_shuffled = df.iloc[1:].sample(frac=1).reset_index(drop=True)

# Concatenate the shuffled DataFrame with the header
df_final = pd.concat([header.to_frame().T, df_shuffled], ignore_index=True)

# Save the shuffled DataFrame to a new CSV file
df_final.to_csv('train_vec.csv', index=False)