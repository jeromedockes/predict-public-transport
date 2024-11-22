SCHOOL_URL = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/exports/parquet?lang=fr&timezone=Europe%2FParis"
HOLIDAYS_URL = (
    "https://etalab.github.io/jours-feries-france-data/csv/jours_feries_metropole.csv"
)

import pathlib
import requests
import duckdb

OUT_DIR = pathlib.Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def _download_school():
    response = requests.get(SCHOOL_URL)
    response.raise_for_status()
    out_file = OUT_DIR / "school_holidays.parquet"
    out_file.write_bytes(response.content)
    con = duckdb.connect(OUT_DIR / "ratp.duckdb")
    con.sql(
        f"create table if not exists school_holidays as select * from read_parquet('{out_file}')"
    )


def _download_holidays():
    response = requests.get(HOLIDAYS_URL)
    response.raise_for_status()
    out_file = OUT_DIR / "holidays.csv"
    out_file.write_bytes(response.content)
    con = duckdb.connect(OUT_DIR / "ratp.duckdb")
    con.sql(
        f"create table if not exists holidays as select * from read_csv('{out_file}')"
    )


if __name__ == "__main__":
    _download_school()
    _download_holidays()
