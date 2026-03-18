# import pandas as pd

# mendeley = pd.read_csv("../data/Mendeley_dataset.csv")
# student = pd.read_csv("../data/Student_dataset.csv")

# cols = ["artist_name","track_name","release_date","genre","lyrics"]

# merged = pd.concat([mendeley[cols], student[cols]])

# merged.to_csv("../data/Merged_dataset.csv", index=False)

# print("Merged dataset created")
import pandas as pd

# Load only the Mendeley dataset
mendeley = pd.read_csv("data/Mendeley_dataset.csv")

# Define the columns you want to keep
cols = ["artist_name", "track_name", "release_date", "genre", "lyrics"]

# Create a new DataFrame with just those columns from Mendeley
mendeley_subset = mendeley[cols]

# Save to CSV
mendeley_subset.to_csv("data/Merged_dataset.csv", index=False)

print("Dataset with selected columns created")