import pandas as pd
from . import utils
import os
import click
import joblib


def read_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name="2024")
    df.rename(columns={"What country do you work in?": "country",
                    "Where is the closest major city or hub?": "location",
                    "Where are you located?": "location_2",
                    "Biotech sub industry?": "industry",
                    "Company Details - public/private/start-up/ subsidiary of ": "company_type",
                    "Company Detail - Approximate Company Size": "company_size",
                    "What degrees do you have? ": "degree",
                    "Years of Experience": "yoe",
                    "Compensation - Annual Base Salary/Pay": "base_salary",
                    "Compensation - Annual Target Bonus ($)": "bonus",
                    "Compensation - Annual Equity/Stock Option": "stock",
                    "Compensation - Sign on Bonus Value": "sign_on",
                    "[Optional] Work Life Balance - On average, how many hours do you work per week": "wlb"}, inplace=True)
    df.loc[df["location"].isna() & df["location_2"].notna(), "location"] = df.loc[df["location"].isna() & df["location_2"].notna(), "location_2"]
    df = df[["country", "location", "industry", "company_type", "company_size", "degree", "yoe", "base_salary", "bonus", "stock", "sign_on", "wlb"]]

    return df


def filter_country(df: pd.DataFrame) -> pd.DataFrame:
    substrings = ["Boston", "Southeast", "VA", "Virginia", "New England", "Midwest", "US", "NY", "NC", "CO", "San Diego", "west", "Tennessee"]

    mask = df["country"].isna() & df["location"].str.contains('|'.join(substrings), case=False, na=False)
    df.loc[mask, "country"] = "USA"
    df = df[df["country"].str.contains('|'.join(["USA", "US", "United States", "America"]), case=False, na=False)]
    df = df.dropna()
    df = df.drop(columns=["country"])

    return df


def process_industry(df: pd.DataFrame) -> pd.DataFrame:
    return utils.drop_counts(df, "industry")


def process_location(df):
    df["location"] = df["location"].str.lower()
    df["location"] = df["location"].replace({"boston": "new england (ma, ct, ri, nh, vt, me)",
                                             "boston ": "new england (ma, ct, ri, nh, vt, me)",
                                             "cambridge ": "new england (ma, ct, ri, nh, vt, me)",
                                             "philadelphia": "pharma central (ny, nj, pa)",
                                             "philadelphia ": "pharma central (ny, nj, pa)",
                                             "philly": "pharma central (ny, nj, pa)",
                                             "new jersey": "pharma central (ny, nj, pa)",
                                             "nyc": "pharma central (ny, nj, pa)",
                                             " nyc": "pharma central (ny, nj, pa)",
                                             "san diego": "west coast (california & pacific northwest)",
                                             "bay area": "west coast (california & pacific northwest)",
                                             "bay area ": "west coast (california & pacific northwest)",
                                             "bay are": "west coast (california & pacific northwest)",
                                             "seattle": "west coast (california & pacific northwest)",
                                             "los angeles": "west coast (california & pacific northwest)",
                                             "los angeles ": "west coast (california & pacific northwest)",
                                             "san diego, ca": "west coast (california & pacific northwest)",
                                             "san francisco": "west coast (california & pacific northwest)",
                                             "kansas city": "midwest (from oh to ks, north to nd)",
                                             "indianapolis": "midwest (from oh to ks, north to nd)",
                                             "chicago": "midwest (from oh to ks, north to nd)",
                                             "washington dc": "dc metro area (dc, va, md, de)",
                                             "tampa": "carolinas & southeast (from nc to ar, south fl and la)",
                                             "denver": "south & mountain west (tx to az, north to mt)",
                                             "tennessee ": "south & mountain west (tx to az, north to mt)"})
    df = utils.drop_counts(df, "location")

    return df


def process_degree(df):
    df["degree"] = df["degree"].apply(lambda x: "PhD" if ("PhD" in x or "PharmD" in x or "MD" in x) else x)
    df["degree"] = df["degree"].apply(lambda x: "MS" if "Masters" in x else x)
    df["degree"] = df["degree"].apply(lambda x: "BS" if "Bachelors" in x else x)
    df["degree"] = df["degree"].apply(lambda x: "Other" if x not in ["PhD", "MS", "BS"] else x)

    return df


def process_wlb(df):
    df["wlb"] = df["wlb"].replace({"60+": ">60",
                                   "51 - 60": "51-60",
                                   "50-60": "51-60",
                                   "46-50": "41-50",
                                   "41 - 45": "41-50",
                                   "46 - 50": "41-50",
                                   "40-50": "41-50",
                                   "36 - 40": "31-40",
                                   "30-40": "31-40",
                                   "30 - 35": "31-40",
                                   "Less than 30": "<30",
                                   "<20": "<30"})
    return df


def process_bonus(df, column_name):
    df.loc[df[column_name].astype(str).str.contains("%"), column_name] = (
        df.loc[df[column_name].astype(str).str.contains("%"), column_name]
        .astype(str)
        .str.extract(r'(\d{1,2})%')[0]
        .astype(float) / 100
    )
    df.loc[df[column_name].astype(str).str.contains("No", case=False, na=False), column_name] = 0.0
    df = df[~df[column_name].apply(lambda x: isinstance(x, str))]
    df.loc[df[column_name].astype(float) <= 10, column_name] = df.loc[df[column_name].astype(float) <= 10, "base_salary"] * df.loc[df[column_name].astype(float) <= 1, column_name].astype(float)
    df = df.fillna(0)
    df[column_name] = df[column_name].astype(float)

    return df


def process_salary(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["base_salary"] < 300, "base_salary"] *= 2080
    df["sign_on"] = df["sign_on"].fillna(0)
    df = process_bonus(df, "bonus")
    df = process_bonus(df, "stock")
    df["total_compensation"] = df["base_salary"] + df["bonus"].astype(float) + df["stock"].astype(float) + df["sign_on"]
    df = df.drop(columns=["base_salary", "bonus", "stock", "sign_on"])

    return df


@click.command()
@click.option(
    "--raw_data_path",
    default="data/raw_data",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    default="data/processed_data",
    help="Location where the resulting files will be saved"
)
def main(raw_data_path: str, dest_path: str):
    df = read_data(os.path.join(raw_data_path, "r_biotech salary and company survey.xlsx"))
    df = filter_country(df)
    df = process_industry(df)
    df = process_location(df)
    df = process_degree(df)
    df = process_wlb(df)
    df = process_salary(df)

    X_train, y_train, X_val, y_val, test_df, dv, scaler = utils.vectorize_data(df)

    os.makedirs(dest_path, exist_ok=True)
    utils.dump_pkl(dv, os.path.join(dest_path, "dv.pkl"))
    utils.dump_pkl((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    utils.dump_pkl((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    test_df.to_csv(os.path.join(dest_path, "test.csv"), index=False)
    joblib.dump(scaler, os.path.join(dest_path, "scaler.save"))


if __name__ == "__main__":
    main()