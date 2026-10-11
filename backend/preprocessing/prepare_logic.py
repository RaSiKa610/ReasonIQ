import os
import pandas as pd


BASE_DIR = r"C:\Users\Rasika\BiasLess"

LOGIC_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "logic",
    "data"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "logic_edu_clean.csv"
)


def load_split(filename, split_name):

    path = os.path.join(
        LOGIC_DIR,
        filename
    )

    df = pd.read_csv(path)

    df = df[
        [
            "source_article",
            "updated_label"
        ]
    ].copy()

    df.columns = [
        "text",
        "label"
    ]

    df["split"] = split_name

    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
    )

    return df


def main():

    print("Loading LOGIC education dataset...")

    train = load_split(
        "edu_train.csv",
        "TRAIN"
    )

    dev = load_split(
        "edu_dev.csv",
        "DEV"
    )

    test = load_split(
        "edu_test.csv",
        "TEST"
    )

    print()
    print("=" * 60)
    print("ORIGINAL DATA")
    print("=" * 60)

    print(f"TRAIN: {len(train)}")
    print(f"DEV:   {len(dev)}")
    print(f"TEST:  {len(test)}")

    train_texts = set(train["text"])

    dev_before = len(dev)

    dev = dev[
        ~dev["text"].isin(train_texts)
    ].copy()

    dev_texts = set(dev["text"])

    test_before = len(test)

    test = test[
        ~test["text"].isin(train_texts)
        & ~test["text"].isin(dev_texts)
    ].copy()

    print()
    print("=" * 60)
    print("LEAKAGE REMOVAL")
    print("=" * 60)

    print(
        f"DEV removed:   {dev_before - len(dev)}"
    )

    print(
        f"TEST removed:  {test_before - len(test)}"
    )

    print(
        f"TRAIN kept:    {len(train)}"
    )

    print(
        f"DEV kept:      {len(dev)}"
    )

    print(
        f"TEST kept:     {len(test)}"
    )

    df = pd.concat(
        [
            train,
            dev,
            test
        ],
        ignore_index=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print(f"Total samples: {len(df)}")

    print("\nSplit counts:")
    print(
        df["split"].value_counts()
    )

    print("\nLabel counts:")
    print(
        df["label"].value_counts()
    )

    print("\nLabel × Split:")
    print(
        pd.crosstab(
            df["label"],
            df["split"]
        )
    )

    print()
    print("=" * 60)
    print("LEAKAGE CHECK")
    print("=" * 60)

    train_texts = set(
        df[df["split"] == "TRAIN"]["text"]
    )

    dev_texts = set(
        df[df["split"] == "DEV"]["text"]
    )

    test_texts = set(
        df[df["split"] == "TEST"]["text"]
    )

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

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()