import logging

import pandas as pd
import os
from sqlalchemy import text

from .config import Settings
from .database import build_engine


logger = logging.getLogger(__name__)


def ingest(settings: Settings) -> pd.DataFrame:
    engine = build_engine(settings=settings)

    try:
        quote = engine.dialect.identifier_preparer.quote_identifier
        schema = quote(settings.pg_schema)
        table = quote(settings.pg_table)

        query = text(
            f"""
            SELECT
                ticket_id,
                subject,
                description,
                category,
                scenario_group
            FROM {schema}.{table}
            ORDER by ticket_id
        """
        )

        with engine.connect() as connection:
            with connection.begin():
                connection.execute(
                    text("SET TRANSACTION READ ONLY")
                )

                data_frame = pd.read_sql_query(sql=query, con=connection)

    # except Exception as e:
    #     logger.info(f"Error occured while reading the data from database: {e}")

    finally:
        engine.dispose()


    logger.info("Loaded %s ticekts from the database")

    return data_frame

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_frame = ingest(settings=Settings())

    print(f"The data frame shape: {data_frame.shape}")
    print(f"Columns: {data_frame.columns}")
    print(data_frame.head(10))




    

