# db.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# ✅ Force load the .env file from the same directory as this db.py
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

# ✅ Get environment variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DBNAME")

if not MONGO_URI or not DB_NAME:
    raise ValueError("❌ Missing MONGO_URI or MONGO_DBNAME in .env")

# ✅ Create client and select database
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ✅ Define collections
users_collection = db["users"]
uploads_collection = db["uploads"]
sessions_collection = db["sessions"]

print("✅ Connected to MongoDB Atlas:", DB_NAME)
