import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (adjust the file path if necessary)
df = pd.read_csv('/Users/ivovanmiert/Documents/SCRIPTIE/GITHUB/Data_CEP/Collection/Data/Wyscout/full_dataframe_concatenated.csv')

# Count occurrences of each 'primary_type'
primary_type_counts = df['primary_type'].value_counts()

# Display the counts
# print(primary_type_counts)

# Plot the counts as a bar chart
plt.figure(figsize=(10, 6))
primary_type_counts.plot(kind='bar', color='skyblue')

# Add labels and title
plt.xlabel('Primary Type')
plt.ylabel('Number of Occurrences')
plt.title('Number of Occurrences for Each Primary Type')

# Display the plot
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Adjust the layout to avoid label cutoff
plt.show()