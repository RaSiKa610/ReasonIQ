import pandas as pd
import os

RAW_PATH = "../datasets/raw/logic/data"
PROCESSED_PATH = "../datasets/processed/logic"

os.makedirs(PROCESSED_PATH, exist_ok=True)


def clean_dataset(filename, output_filename):
    input_path = os.path.join(RAW_PATH, filename)
    output_path = os.path.join(PROCESSED_PATH, output_filename)

    df = pd.read_csv(input_path)

    df = df[["source_article", "updated_label"]].copy()

    df["source_article"] = df["source_article"].astype(str).str.strip()
    df["updated_label"] = df["updated_label"].astype(str).str.strip()

    before = len(df)

    df = df.drop_duplicates(
        subset=["source_article"],
        keep="first"
    )

    removed = before - len(df)

    df.to_csv(output_path, index=False)

    print(f"\n{filename}")
    print(f"Original rows : {before}")
    print(f"Duplicates removed : {removed}")
    print(f"Final rows : {len(df)}")
    print(f"Saved to : {output_path}")

    print("\nClass distribution:")
    print(df["updated_label"].value_counts())


clean_dataset(
    "edu_train.csv",
    "edu_train_clean.csv"
)

clean_dataset(
    "edu_dev.csv",
    "edu_dev_clean.csv"
)

clean_dataset(
    "edu_test.csv",
    "edu_test_clean.csv"
)

print("\nCleaning completed successfully.")