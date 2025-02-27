from airtable import Airtable
import os
from dotenv import load_dotenv

load_dotenv()

class AirtableClient:
    def __init__(self):
        self.api_key = os.getenv('AIRTABLE_API_KEY')
        self.base_id = os.getenv('AIRTABLE_BASE_ID')
        self.table_name = os.getenv('AIRTABLE_TABLE_NAME')
        self.airtable = Airtable(self.base_id, self.table_name, self.api_key)
    
    def fetch_all_records(self):
        records = self.airtable.get_all()
        return [record['fields'] for record in records] 