import json
import hashlib

import tensorflow as tf 

from .config import Settings
from .data_ingestion import ingest
from .data_preprocessing import prepare_text
from .data_split import split_data
from .text_tokenization import build_tokenizer
from .label_encoder import encode_labels
from pathlib import Path

# from tf.keras.models import Sequencial 
import ipdb


def build_model(vocabulary_size, number_of_classes, sequence_length):
    # ipdb.set_trace()
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(sequence_length,), dtype="int64"),
        tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=32, mask_zero=True, name="word_embeddings"),
        tf.keras.layers.GlobalAveragePooling1D(name="ticket_representation"),
        tf.keras.layers.Dense(units=64, activation="relu", name="hidden"),
        tf.keras.layers.Dense(units=number_of_classes, activation="softmax", name="category"),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # ipdb.set_trace()
    return model



def save_artifacts(model,model_path, tokenizer, encoder, test_df, history, ):
    # ipdb.set_trace()
    model_path = Path(model_path).with_suffix(".keras")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)

    tokenizer_data = {
        "config": tokenizer.get_config(),
        "vocabulary": tokenizer.get_vocabulary()
    }

    tokenizer_path = model_path.parent / "tokenizer.json"

    with tokenizer_path.open(mode="w", encoding="utf-8") as file:
        json.dump(tokenizer_data, file, indent=2, ensure_ascii=False)

    test_snapshot = (
        test_df[["ticket_id", "text", "category"]].sort_values("ticket_id").to_json(orient="records")
    )

    metadata = {
        "format": "token_ticket_category_v1",
        "model_file": model_path.name,
        "tokenizer_file": tokenizer_path.name,
        "classes": encoder.classes_.tolist(),
        "sequence_length": int(model.input_shape[1]),
        "test_ticket_ids": test_df["ticket_id"].tolist(),
        "test_sha256": hashlib.sha256(
            test_snapshot.encode()
        ).hexdigest(),
        "history": {
            metric:[float(value) for value in values] for metric, values in history.history.items()
        }
    }

    metadata_path = model_path.parent / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)




def main():
    settings = Settings()

    raw_df = ingest(Settings())
        
    prepared_df = prepare_text(raw_df)
    train_df, val_df, test_df = split_data(prepared_df)

    token_counts = train_df["text"].str.split().str.len()

    print(f"\n Training ticket length: {token_counts}")

    tokenizer = build_tokenizer(train_df["text"].fillna("").astype(str))

    X_train = tokenizer(tf.constant(train_df["text"].tolist()))
    X_val = tokenizer(tf.constant(val_df["text"].tolist()))

    y_train, y_val, y_test, encoder = encode_labels(train_df["category"], val_df["category"], test_df["category"])

    model = build_model(
        vocabulary_size=len(tokenizer.get_vocabulary()),
        number_of_classes=len(encoder.classes_),
        sequence_length=X_train.shape[1],
    )
    # ipdb.set_trace()

    model.summary()

    history = model.fit(
        X_train, y_train, validation_data=[X_val, y_val], epochs=50, batch_size=32
    )
    print(history)

    save_artifacts(model=model, model_path=settings.model_path, tokenizer=tokenizer, encoder=encoder, test_df=test_df, history=history)



if __name__ == "__main__":
    main()
    # ipdb.set_trace()

