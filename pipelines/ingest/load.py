import os

import pandas as pd
from sqlalchemy import create_engine


def load_data():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/autopricer")
    engine = create_engine(db_url)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

    csv_files = {
        "regions": "regions.csv",
        "vehicles": "vehicles.csv",
        "enquiries": "enquiries.csv",
        "sales": "sales.csv",
    }

    with engine.connect() as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS raw;")

    for table_name, file_name in csv_files.items():
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Skipping {file_name}, does not exist.")
            continue

        print(f"Loading {file_name} into raw.{table_name}...")
        df = pd.read_csv(file_path)

        # Load in chunks to avoid memory explosion with 50k+ rows
        df.to_sql(
            table_name, engine, schema="raw", if_exists="replace", index=False, chunksize=10000
        )
        print(f"Loaded {len(df)} rows into raw.{table_name}")


if __name__ == "__main__":
    load_data()
