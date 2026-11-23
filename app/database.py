from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

from .config import Settings

def build_engine(settings: Settings) -> Engine:
    url = URL.create(
        drivername="postgresql+psycopg",
        username=settings.pg_user,
        password=settings.pg_password.get_secret_value(),
        host=settings.pg_host,
        port=settings.pg_port,
        database=settings.pg_database
    )

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        hide_parameters=True,
        # connect_args={
        #     "connection_timeout": 10,
            
        # }
    )


# if __name__ == "__main__":
#     settings = Settings()
#     engine = build_engine(settings=settings)
#     print(engine)



