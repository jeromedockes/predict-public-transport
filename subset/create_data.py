from pathlib import Path

import duckdb

data_dir = Path(__file__).parents[1] / "data"
subset_dir = Path(__file__).parent

con = duckdb.connect(data_dir / "ratp.duckdb", read_only=True)

holidays_pq = subset_dir / "holidays.parquet"
con.sql(f"COPY (SELECT * FROM holidays) TO '{holidays_pq}' (FORMAT 'parquet');")

school_holidays_pq = subset_dir / "school_holidays.parquet"
con.sql(
    f"COPY (SELECT * FROM school_holidays) TO '{school_holidays_pq}' "
    "(FORMAT 'parquet');"
)

for name, code in [("T3a", 13), ("T2", 12)]:
    usage_pq = subset_dir / f"{name}.parquet"
    con.sql(
        f"""
    COPY
    (SELECT * FROM surface WHERE
        CODE_STIF_TRNS = 100
        AND trim(CODE_STIF_RES) = '112'
        AND trim(CODE_STIF_LIGNE) = '{code}')
    TO '{usage_pq}'
    (FORMAT 'parquet');
        """
    )
