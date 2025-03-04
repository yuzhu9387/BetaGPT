import os
import json
import tempfile
from airtable import Airtable
from dotenv import load_dotenv

load_dotenv()


class AirtableClient:
    def __init__(self):
        self.api_key = os.getenv('AIRTABLE_API_KEY')
        self.base_id = os.getenv('AIRTABLE_BASE_ID')
        self.table_name = os.getenv('AIRTABLE_TABLE_NAME')
        self.airtable = Airtable(self.base_id, self.table_name, self.api_key)
        self.airtable_records = 'airtable_records.txt'
        
        # check and fetch records
        self.check_and_fetch_records()

    def check_and_fetch_records(self):
        """
        check if the records file exists, if not, fetch all records
        """
        if not os.path.exists(self.airtable_records):
            print(f"Records file not found at {self.airtable_records}, fetching from Airtable...")
            self.fetch_all_records()
        else:
            print(f"Records file already exists at {self.airtable_records}, skipping fetch")

    def fetch_all_records(self, batch_size=100):
        """
        fetch all Airtable records and save to a default file
        """
        save_path = self.airtable_records
        self.delete_records_file()
        try:
            with open(save_path, 'w', encoding='utf-8') as permanent_file:
                offset = None
                # fetch all records in pages
                while True:
                    if offset:
                        records = self.airtable.get_all(offset=offset, page_size=batch_size)
                    else:
                        records = self.airtable.get_all(page_size=batch_size)
                    
                    if not records:
                        break
                    
                    if len(records) == batch_size:
                        offset = records[-1]['id']
                    else:
                        offset = None
                    
                    for record in records:
                        fields = record.get('fields', {})
                        permanent_file.write(json.dumps(fields, ensure_ascii=False) + '\n')
                    
                    if not offset:
                        break
                    
        except Exception as e:
            print(f"Error fetching Airtable records: {str(e)}")
            raise

    def delete_records_file(self):
        """
        delete airtable_records.txt file
        """
        try:
            if os.path.exists(self.airtable_records):
                print(f"Deleting existing records file: {self.airtable_records}")
                os.remove(self.airtable_records)
                print("Records file deleted successfully")
            else:
                print(f"No records file found at {self.airtable_records}")
        except Exception as e:
            print(f"Error deleting records file: {str(e)}")
            raise
