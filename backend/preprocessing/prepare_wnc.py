import os
import pandas as pd
import re


BASE_DIR = r"C:\Users\Rasika\BiasLess"

RAW_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "wnc",
    "bias_data",
    "WNC"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "processed"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "wnc_bias_clean.csv"
)


COLUMNS = [
    "id",
    "src_tok",
    "tgt_tok",
    "src_raw",
    "tgt_raw",
    "src_POS_tags",
    "tgt_parse_tags"
]


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_split(filename, split):
    path = os.path.join(RAW_DIR, filename)

    print(f"Reading: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        keep_default_na=False
    )

    print(f"  Rows loaded: {len(df):,}")

    df["src_raw"] = df["src_raw"].apply(clean_text)
    df["tgt_raw"] = df["tgt_raw"].apply(clean_text)

    df = df[
        (df["src_raw"].str.len() > 0) &
        (df["tgt_raw"].str.len() > 0)
    ].copy()

    df["split"] = split

    return df


def main():

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train = load_split(
        "biased.word.train",
        "train"
    )

    dev = load_split(
        "biased.word.dev",
        "dev"
    )

    test = load_split(
        "biased.word.test",
        "test"
    )

    df = pd.concat(
        [train, dev, test],
        ignore_index=True
    )

    print("\nCombined dataset")
    print("----------------")
    print(f"Total rows: {len(df):,}")

    print("\nRows by split:")
    print(df["split"].value_counts())

    # Remove exact duplicate biased sentences.
    # Keep the first occurrence according to train -> dev -> test.
    before = len(df)

    df = df.drop_duplicates(
        subset=["src_raw"],
        keep="first"
    ).reset_index(drop=True)

    removed = before - len(df)

    print(f"\nDuplicate source sentences removed: {removed:,}")

    print("\nRows after deduplication:")
    print(df["split"].value_counts())

    # Check leakage across splits
    train_texts = set(
        df.loc[df["split"] == "train", "src_raw"]
    )

    dev_texts = set(
        df.loc[df["split"] == "dev", "src_raw"]
    )

    test_texts = set(
        df.loc[df["split"] == "test", "src_raw"]
    )

    print("\nCross-split leakage check")
    print("-------------------------")
    print(
        "TRAIN ∩ DEV:",
        len(train_texts & dev_texts)
    )
    print(
        "TRAIN ∩ TEST:",
        len(train_texts & test_texts)
    )
    print(
        "DEV ∩ TEST:",
        len(dev_texts & test_texts)
    )

    # Save only the information needed for BiasLess.
    output = df[
        [
            "id",
            "src_raw",
            "tgt_raw",
            "split"
        ]
    ].copy()

    output["label"] = 1

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\nFinal columns:")
    print(output.columns.tolist())

    print("\nFirst 5 rows:")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()