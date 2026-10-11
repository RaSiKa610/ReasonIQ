import os
import re
import pandas as pd


BASE_DIR = r"C:\Users\Rasika\BiasLess"

AAEC_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "aaec",
    "ArgumentAnnotatedEssays-2.0"
)

BRAT_DIR = os.path.join(
    AAEC_DIR,
    "brat-project-final"
)

SPLIT_FILE = os.path.join(
    AAEC_DIR,
    "train-test-split.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "processed"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "aaec_components.csv"
)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_ann_file(ann_file, txt_file):
    annotations = []

    with open(txt_file, "r", encoding="utf-8") as f:
        essay_text = f.read()

    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.startswith("T"):
                continue

            parts = line.split("\t")

            if len(parts) < 3:
                continue

            annotation_id = parts[0]
            annotation_info = parts[1]
            annotated_text = "\t".join(parts[2:])

            info_parts = annotation_info.split()

            if len(info_parts) < 3:
                continue

            label = info_parts[0]

            if label not in ["MajorClaim", "Claim", "Premise"]:
                continue

            try:
                start = int(info_parts[1])
                end = int(info_parts[2])
            except ValueError:
                continue

            extracted_text = essay_text[start:end]

            annotations.append({
                "annotation_id": annotation_id,
                "label": label,
                "start": start,
                "end": end,
                "text": clean_text(extracted_text),
                "ann_text": clean_text(annotated_text)
            })

    return annotations


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading official train/test split...")

    split_df = pd.read_csv(
        SPLIT_FILE,
        sep=";"
    )

    split_map = dict(
        zip(
            split_df["ID"],
            split_df["SET"]
        )
    )

    print(f"Total essays in split file: {len(split_map)}")

    all_rows = []

    total_essays = 0
    successful_essays = 0

    for essay_id, split in split_map.items():

        txt_file = os.path.join(
            BRAT_DIR,
            essay_id + ".txt"
        )

        ann_file = os.path.join(
            BRAT_DIR,
            essay_id + ".ann"
        )

        total_essays += 1

        if not os.path.exists(txt_file):
            print(f"WARNING: Missing TXT file: {essay_id}")
            continue

        if not os.path.exists(ann_file):
            print(f"WARNING: Missing ANN file: {essay_id}")
            continue

        annotations = parse_ann_file(
            ann_file,
            txt_file
        )

        if annotations:
            successful_essays += 1

        for annotation in annotations:

            all_rows.append({
                "essay_id": essay_id,
                "text": annotation["text"],
                "label": annotation["label"],
                "split": split,
                "start": annotation["start"],
                "end": annotation["end"]
            })

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("\nERROR: No annotations were extracted.")
        return

    df = df[
        [
            "essay_id",
            "text",
            "label",
            "split",
            "start",
            "end"
        ]
    ]

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("AAEC PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Essays found:       {total_essays}")
    print(f"Essays with data:   {successful_essays}")
    print(f"Extracted samples:  {len(df)}")

    print("\nLABEL COUNTS:")
    print(df["label"].value_counts())

    print("\nSPLIT COUNTS:")
    print(df["split"].value_counts())

    print("\nLABEL × SPLIT:")
    print(
        pd.crosstab(
            df["label"],
            df["split"]
        )
    )

    print("\nMISSING VALUES:")
    print(df.isnull().sum())

    print("\nDUPLICATES:")
    print(df["text"].duplicated().sum())

    print("\nSAMPLE:")
    print(df.head(10).to_string(index=False))

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()