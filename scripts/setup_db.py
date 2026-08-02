import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "destinations.csv"
DB_PATH = PROJECT_ROOT / "db" / "tourism.db"

CREATE_TABLE_SQL = """
CREATE TABLE destinations (
    id                      INTEGER PRIMARY KEY,
    name                    TEXT NOT NULL,
    category                TEXT NOT NULL,
    subcategory             TEXT,
    location                TEXT,
    district                TEXT,
    description             TEXT,
    entrance_fee_lkr        INTEGER,
    accessibility           TEXT,
    best_time_to_visit      TEXT,
    opening_hours           TEXT,
    significance_or_activities TEXT,
    image_filenames         TEXT,   -- semicolon-separated list, e.g. "a.jpg;b.jpg"
    source_urls             TEXT
);
"""


def build_database():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CSV_PATH}. Make sure destinations.csv is in data/."
        )

    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    required_cols = {
        "id", "name", "category", "location", "district", "description",
        "entrance_fee_lkr", "accessibility", "best_time_to_visit",
        "opening_hours", "significance_or_activities", "image_filenames",
        "source_urls",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in destinations.csv — each row needs a unique id.")

    empty_desc = df[df["description"].isna() | (df["description"].str.strip() == "")]
    if not empty_desc.empty:
        print(f"WARNING: {len(empty_desc)} row(s) have an empty description: "
              f"{empty_desc['name'].tolist()}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS destinations;")
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    df.to_sql("destinations", conn, if_exists="append", index=False)
    conn.commit()

    print(f"Loaded {len(df)} rows into {DB_PATH}")

    print("\n--- Sample: 3 rows ---")
    for row in cur.execute("SELECT id, name, category, entrance_fee_lkr FROM destinations LIMIT 3;"):
        print(row)

    print("\n--- Count by category ---")
    for row in cur.execute("SELECT category, COUNT(*) FROM destinations GROUP BY category;"):
        print(row)

    print("\n--- Example structured query: beaches with entrance_fee_lkr = 0 ---")
    for row in cur.execute(
        "SELECT name, location FROM destinations WHERE category='beach' AND entrance_fee_lkr=0;"
    ):
        print(row)

    conn.close()
    print("\nDone. Database saved at:", DB_PATH)


if __name__ == "__main__":
    build_database()
