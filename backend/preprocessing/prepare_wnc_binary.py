import os
import pandas as pd
import re

BASE_DIR = r"C:\Users\Rasika\BiasLess"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "wnc_bias_clean.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "wnc_binary_clean.csv"
)


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():

    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "id": str,
            "src_raw": str,
            "tgt_raw": str,
            "split": str
        }
    )

    print(f"Original WNC pairs: {len(df):,}")

    # Clean source and target text
    df["src_raw"] = df["src_raw"].apply(clean_text)
    df["tgt_raw"] = df["tgt_raw"].apply(clean_text)

    # Remove empty pairs
    df = df[
        (df["src_raw"].str.len() > 0) &
        (df["tgt_raw"].str.len() > 0)
    ].copy()

    # --------------------------------------------------
    # Create binary examples
    # --------------------------------------------------

    biased = pd.DataFrame({
        "id": df["id"] + "_src",
        "source_id": df["id"],
        "text": df["src_raw"],
        "label": 1,
        "split": df["split"]
    })

    neutral = pd.DataFrame({
        "id": df["id"] + "_tgt",
        "source_id": df["id"],
        "text": df["tgt_raw"],
        "label": 0,
        "split": df["split"]
    })

    binary_df = pd.concat(
        [biased, neutral],
        ignore_index=True
    )

    # --------------------------------------------------
    # Clean text duplicates
    # --------------------------------------------------

    print("\nBefore duplicate removal:")
    print(f"Total examples: {len(binary_df):,}")

    before = len(binary_df)

    binary_df = binary_df.drop_duplicates(
        subset=["text"],
        keep="first"
    ).reset_index(drop=True)

    removed = before - len(binary_df)

    print(f"Duplicate texts removed: {removed:,}")

    # --------------------------------------------------
    # Check class balance
    # --------------------------------------------------

    print("\nClass distribution:")
    print(binary_df["label"].value_counts().sort_index())

    print("\nClass distribution by split:")
    print(
        pd.crosstab(
            binary_df["split"],
            binary_df["label"]
        )
    )

    # --------------------------------------------------
    # Cross-split leakage check
    # --------------------------------------------------

    train_texts = set(
        binary_df.loc[
            binary_df["split"] == "train",
            "text"
        ]
    )

    dev_texts = set(
        binary_df.loc[
            binary_df["split"] == "dev",
            "text"
        ]
    )

    test_texts = set(
        binary_df.loc[
            binary_df["split"] == "test",
            "text"
        ]
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

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    binary_df = binary_df[
        [
            "id",
            "source_id",
            "text",
            "label",
            "split"
        ]
    ]

    binary_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\nSaved:")
    print(OUTPUT_FILE)

    # --------------------------------------------------
    # Show examples
    # --------------------------------------------------

    print("\nExample biased samples:")
    print(
        binary_df[
            binary_df["label"] == 1
        ][["text", "label"]]
        .head(5)
        .to_string(index=False)
    )

    print("\nExample neutralized samples:")
    print(
        binary_df[
            binary_df["label"] == 0
        ][["text", "label"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()