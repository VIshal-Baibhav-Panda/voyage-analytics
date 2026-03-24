import pandas as pd

# Load flights dataset
df = pd.read_csv("data/raw/flights.csv")

def recommend_destination(source):
    # Filter based on source
    filtered = df[df["from"] == source]

    # Get top destinations
    destinations = filtered["to"].value_counts().head(3).index.tolist()

    if len(destinations) > 0:
        return destinations

    return ["Goa", "Delhi", "Mumbai"]