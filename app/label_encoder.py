from sklearn.preprocessing import LabelEncoder

from .config import Settings
from .data_ingestion import ingest
from .data_preprocessing import prepare_text
from .data_split import split_data
from .text_tokenization import build_tokenizer


def encode_labels(train_categories, val_categories, test_categories):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_categories)
    y_val = encoder.transform(val_categories)
    y_test = encoder.transform(test_categories)

    return y_train, y_val, y_test, encoder

if __name__ == "__main__":
    raw_df = ingest(Settings())
        
    prepared_df = prepare_text(raw_df)
    train_df, val_df, test_df = split_data(prepared_df)

    y_train, y_val, y_test, encoder = encode_labels(train_df["category"], val_df["category"], test_df["category"])

    print("Category mapping...")
    for class_id, category in enumerate(encoder.classes_):
        print(f"{category} -> {class_id}")