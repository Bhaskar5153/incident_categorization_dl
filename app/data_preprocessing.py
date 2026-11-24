import pandas as pd

from .config import Settings
from .data_ingestion import ingest

# print("Yes, its working")

def prepare_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["subject"] = df["subject"].fillna("")
    df["description"] = df["description"].fillna("")

    df["text"] = (df["subject"] + " " + df["description"]).str.strip()

    df = df[(df["text"] != "") & (df["category"] != "")].copy()
    return df

if __name__ == "__main__":
    raw_df = ingest(Settings())

    prepared_df = prepare_text(raw_df)

    print(prepared_df["text"][0])

