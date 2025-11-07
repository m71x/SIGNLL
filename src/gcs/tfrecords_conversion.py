import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# === 1. Define paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "training_data", "twitter_sentiment")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")

# === 2. Download dataset from Kaggle ===
def download_kaggle_dataset():
    print("Downloading Sentiment140 dataset from Kaggle...")
    os.system(f'kaggle datasets download -d kazanova/sentiment140 -p "{DATA_DIR}"')
    os.system(f'unzip -o "{os.path.join(DATA_DIR, "sentiment140.zip")}" -d "{DATA_DIR}"')

# === 3. Convert to DataFrame ===
def load_dataset():
    print("Loading CSV...")
    df = pd.read_csv(
        CSV_PATH,
        encoding="latin-1",
        header=None,
        names=["target", "ids", "date", "flag", "user", "text"]
    )
    df = df[["text", "target"]]
    df["target"] = df["target"].apply(lambda x: 1 if x == 4 else 0)  # 0=negative, 1=positive
    print(f"Loaded {len(df)} rows.")
    return df

# === 4. Serialize TFRecord Example ===
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(text, label):
    feature = {
        "text": _bytes_feature(text),
        "label": _int64_feature(label)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

# === 5. Write TFRecord ===
def write_tfrecord(df, path):
    with tf.io.TFRecordWriter(path) as writer:
        for _, row in df.iterrows():
            example = serialize_example(row["text"], int(row["target"]))
            writer.write(example)
    print(f"✅ Wrote {len(df)} examples to {path}")

# === 6. Main routine ===
if __name__ == "__main__":
    # Step 1: Download if not already present
    if not os.path.exists(CSV_PATH):
        download_kaggle_dataset()
    else:
        print("Dataset already exists. Skipping download.")
    
    # Step 2: Load and split
    df = load_dataset()
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Step 3: Write TFRecords
    train_path = os.path.join(DATA_DIR, "train.tfrecord")
    test_path = os.path.join(DATA_DIR, "test.tfrecord")

    write_tfrecord(train_df, train_path)
    write_tfrecord(test_df, test_path)

    print("\n🎉 Conversion complete!")
    print(f"TFRecords saved in: {DATA_DIR}")
