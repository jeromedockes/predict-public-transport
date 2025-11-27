"""
https://eu.ftp.opendatasoft.com/stif/Validations/Donnees_de_validation.pdf
https://prim.iledefrance-mobilites.fr/en/jeux-de-donnees/histo-validations-reseau-ferre

"""
import itertools
import pathlib
import shutil

import requests
import duckdb

OUT_DIR = pathlib.Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

API_ROOT = "https://data.iledefrance-mobilites.fr/api/explore/v2.1"
RAIL_METADATA_URL = "catalog/datasets/histo-validations-reseau-ferre/exports/json"
SURFACE_METADATA_URL = "catalog/datasets/histo-validations-reseau-surface/exports/json"


def _download(url, key):
    print(f"Download {key!r}")
    validation_metadata = requests.get(f"{API_ROOT}/{url}").json()

    for year_metadata in validation_metadata:
        print(year_metadata["annee"])
        year_dir = OUT_DIR / year_metadata["annee"]
        year_dir.mkdir(exist_ok=True)
        year_data = requests.get(year_metadata[key]["url"])
        (year_dir / f"{key}.zip").write_bytes(year_data.content)


def download_data():
    _download(RAIL_METADATA_URL, "reseau_ferre")
    _download(SURFACE_METADATA_URL, "reseau_de_surface")


def extract():
    print("Extract")
    for year_data in OUT_DIR.glob("*/*.zip"):
        print(year_data)
        shutil.unpack_archive(year_data, year_data.parent)


def get_connection():
    return duckdb.connect(OUT_DIR / "ratp.duckdb")


def _to_utf8(path):
    out = path.with_name(f"{path.name}.utf8")
    with open(path, "rb") as in_stream:
        with open(out, "wb") as out_stream:
            while chunk := in_stream.read(1024):
                out_stream.write(chunk.decode("latin-1").encode("utf-8"))
    return out


def _load(table_name, key):
    print("Create db")
    long_key = {"rs": "SURFACE", "rf": "FER"}
    con = get_connection()
    for i, year_csv in enumerate(
        filter(
            lambda p: p.suffix != ".utf8",
            itertools.chain(
                OUT_DIR.glob(f"*/data-{key}*/*NB*"),
                OUT_DIR.glob(f"*/*NB_{long_key}.txt"),
            ),
        )
    ):
        print(i, year_csv)
        utf8 = _to_utf8(year_csv)
        with open(utf8, "r") as f:
            line = f.readline()
            delim = ";" if ";" in line else "\t"
        types = {
            "NB_VALD": "VARCHAR",
            "CODE_STIF_RES": "VARCHAR",
            "CODE_STIF_LIGNE": "VARCHAR",
            "CODE_STIF_ARRET": "VARCHAR",
        }
        if key == "rf":
            types.pop("CODE_STIF_LIGNE")
        else:
            types.pop("CODE_STIF_ARRET")
        try:
            try:
                con.sql(
                    f"insert into {table_name} SELECT * from "
                    f"read_csv('{utf8}', delim='{delim}', types={types}, nullstr=['?', '']);"
                )
            except duckdb.CatalogException:
                con.sql(
                    f"CREATE TABLE {table_name} as SELECT * from "
                    f"read_csv('{utf8}', delim='{delim}', types={types}, nullstr=['?', '']);"
                )
        except duckdb.InvalidInputException:
            print(f"FAILED: {year_csv}")
            continue


def load():
    _load("rail", "rf")
    _load("surface", "rs")


if __name__ == "__main__":
    download_data()
    extract()
    load()
