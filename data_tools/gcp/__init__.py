from pandas import DataFrame
import google


def bq_to_df(
    query: str, client: google.cloud.bigquery.Client, location: str = None
) -> DataFrame:
    return client.query(query, location=location).to_dataframe()
