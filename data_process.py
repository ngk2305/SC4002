import pandas as pd
df = pd.read_csv('text_emotion.csv')
unique_labels_count = df['sentiment'].nunique()
print(unique_labels_count)