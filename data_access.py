import ibis
from ibis import _
import polars as pl


def add_school_holidays(surface, con):
    holidays = con.table("school_holidays")
    paris = holidays.filter(_.location == "Paris", _.population.isin(["-", "Élèves"]))
    start = paris.select(
        date=_.start_date.cast("date"),
        is_school_holiday=1,
    )
    end = paris.select(
        date=_.end_date.cast("date"),
        is_school_holiday=0,
    )
    holiday_events = start.union(end).order_by(_.date.asc())
    return surface.asof_join(holiday_events, holiday_events.date <= surface.DAY).drop(
        "date"
    )


def add_holidays(surface, con):
    holidays = con.table("holidays")
    return surface.mutate(is_holiday=surface.DAY.isin(holidays.date))


def load_surface(con):
    surface = con.table("surface")
    surface = surface.mutate(
        N=_.NB_VALD.strip().re_replace("Moins de 5", "4").cast(int)
    )
    surface = surface.mutate(
        LIBELLE_LIGNE=(
            _.LIBELLE_LIGNE.isin(["?", "NON DEFINI", "LIGNE NON DEFINIE", ""]).ifelse(
                ibis.null(), _.LIBELLE_LIGNE
            )
        )
    )
    code_cols = ["CODE_STIF_TRNS", "CODE_STIF_RES", "CODE_STIF_LIGNE"]
    surface = surface.mutate(
        **{c: _[c].cast(str).strip().try_cast(int) for c in code_cols}
    )
    surface = surface.filter(*[_[c].notnull() for c in code_cols])
    surface = surface.mutate(
        LINE=ibis.array([_[c].cast(str).strip() for c in code_cols]).join("__")
    )
    surface = surface.mutate(DAY=_.JOUR)
    surface = surface.group_by("DAY", "LINE").agg(
        N=_.N.sum(), LINE_NAME=_.LIBELLE_LIGNE.first()
    ).order_by(["DAY", "LINE"])
    return surface


def ibis_regular_time_grid(surface):
    all_days = surface.alias("s").sql(
        """
    SELECT CAST(d AS DATE) AS DAY FROM
    generate_series(
        (SELECT min(DAY) FROM s), (SELECT max(DAY) FROM s) + INTERVAL '10 days', INTERVAL '1 day') AS t(d)
    """
    )
    all_points = all_days.cross_join(surface.select("LINE").distinct())
    surface = (
        all_points.left_join(
            surface, (surface.LINE == all_points.LINE, surface.DAY == all_points.DAY)
        )
        .drop(["DAY_right", "LINE_right"])
        .order_by("DAY", "LINE")
    )
    return surface


def polars_regular_time_grid(df):
    dates = pl.DataFrame(
        {"DAY": pl.date_range(df["DAY"].min(), df["DAY"].max(), "1d", eager=True)}
    )
    points = dates.join(df.select("LINE").unique(), how="cross")
    df = points.join(df, on=["DAY", "LINE"], how="left")
    return df


def add_lagged_features(surface):
    lags = {f"N_lag_{lag}": _.N.lag(lag) for lag in [3, 4, 5, 6, 7, 14, 21, 28, 35]}
    surface = surface.group_by("LINE").order_by("DAY").mutate(**lags)
    averages = {}
    avg_lag = 3
    for width in [3, 7, 30, 90]:
        w = ibis.window(
            group_by="LINE",
            order_by="DAY",
            preceding=width + avg_lag,
            following=-avg_lag,
        )
        averages[f"N_lag_{avg_lag}_avg_{width}"] = _.N.mean().over(w)
    surface = surface.mutate(**averages)
    return surface.order_by("DAY", "LINE")


def add_datetime_features(surface):
    return surface.mutate(
        day_of_month=_.DAY.day(),
        day_of_year=_.DAY.day_of_year(),
        month=_.DAY.month(),
        year=_.DAY.year(),
        week=_.DAY.week_of_year(),
        weekday=_.DAY.day_of_week.index(),
    )


def load_surface_features(con):
    surface = load_surface(con)
    surface = ibis_regular_time_grid(surface)
    surface = add_lagged_features(surface)
    surface = add_datetime_features(surface)
    surface = add_school_holidays(surface, con)
    surface = add_holidays(surface, con)
    return surface.drop("N")


def load_data_points(con):
    return (
        load_surface(con)
        .select(["DAY", "LINE", "N"])
        .order_by(["DAY", "LINE"])
        .to_polars()
    )
