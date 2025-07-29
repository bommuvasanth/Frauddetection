from dotenv import load_dotenv
load_dotenv() # Load environment variables at the very beginning

from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017/"
    MONGODB_DB_NAME: str = "fraud_detection_db"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings() 