import pandas as pd

# Load users dataset once
df = pd.read_csv("data/raw/users.csv")

def predict_gender(user_code):
    user = df[df["userCode"] == int(user_code)]

    if not user.empty:
        return user.iloc[0]["gender"]
    
    return "Unknown"