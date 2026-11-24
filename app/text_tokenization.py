import tensorflow as tf

from .config import Settings
from .data_ingestion import ingest
from .data_preprocessing import prepare_text
from .data_split import split_data


def build_tokenizer(training_text):
    tokenizer = tf.keras.layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=100,
        split="whitespace"
    )

    tokenizer.adapt(tf.constant(training_text.tolist()))

    return tokenizer



if __name__ == "__main__":
    raw_df = ingest(Settings())
    
    prepared_df = prepare_text(raw_df)
    train_df, val_df, test_df = split_data(prepared_df)

    tokenizer = build_tokenizer(train_df["text"])

    X_train = tokenizer(tf.constant(train_df["text"].tolist()))
    X_test = tokenizer(tf.constant(test_df["text"].tolist()))
    X_val = tokenizer(tf.constant(val_df["text"].tolist()))

    vocalubary = tokenizer.get_vocabulary()


    # print(vocalubary)
    print(X_train)


