import json
import os
from functools import cache
import pandas as pd


_TESTS_TO_COMP_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "csvs", "tests.json")

@cache
def cached_reader(csv_path: str, filter_query: str = None, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(csv_path, **kwargs)
    if filter_query is not None:
        return df.query(filter_query)
    else:
        return df

@cache
def get_tests(remove_urine = False, remove_percent=False, remove_pct=False, tests_json_path: str =_TESTS_TO_COMP_PATH) -> list[str]:
    with open(tests_json_path) as f:
        tsts = json.load(f)
    if remove_urine:
        tsts = list(filter(lambda x: not(x == "SPECIFIC_GRAVITY" or x == "PH_u") , tsts))
    if remove_percent:
        tsts = list(filter(lambda x: "perc" not in x, tsts))
    if remove_pct:
        tsts = list(filter(lambda x: x not in ["PCT"], tsts))
    return tsts
