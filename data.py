import json
import pandas as pd

def get_data(path="Sarcasm_Headlines_Dataset_v2.json"):
    records = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(records)[["headline", "is_sarcastic"]]
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    return train_df, test_df

# print(df.head())
# print(df.shape)