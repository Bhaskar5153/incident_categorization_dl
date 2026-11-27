import os
from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    pg_host:str = "localhost"
    pg_port:int = 5432
    pg_database: str = "it_incidents"
    pg_user:str = "postgres"
    pg_password: SecretStr = SecretStr("")
    pg_schema: str = "public"
    pg_table: str = "it_support_tickets"

    model_path: Path = Path("artifacts/ticket_category_model.pkl")

    
    

