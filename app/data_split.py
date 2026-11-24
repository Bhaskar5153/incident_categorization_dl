import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Settings
from .data_ingestion import ingest
from .data_preprocessing import prepare_text


def split_data(df: pd.DataFrame):

    groups = df[["scenario_group", "category"]].drop_duplicates()

    remaining_groups, test_groups = train_test_split(groups, test_size=0.20, random_state=42, stratify=groups["category"])

    train_groups, val_groups = train_test_split(remaining_groups, test_size=0.25, random_state=42, stratify=remaining_groups["category"])

    train_df = df[df["scenario_group"].isin(train_groups["scenario_group"])].copy()

    val_df = df[df["scenario_group"].isin(val_groups["scenario_group"])].copy()

    test_df = df[df["scenario_group"].isin(test_groups["scenario_group"])].copy()

    return train_df, val_df, test_df


if __name__ == "__main__":
    raw_df = ingest(Settings())

    prepared_df = prepare_text(raw_df)

    train_df, val_df, test_df = split_data(prepared_df)
    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)