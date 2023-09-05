import json
from pathlib import Path
from pymongo import MongoClient

class MongoDBManager:
    ROOT_DIR = Path(__file__).parent.parent / 'data' / 'simulated2'

    def __init__(
        self,
        host='0.0.0.0',
        port=27017,
        database_name='agreement-tracking',
        collection_name='example_collection'
    ):
        self.client = MongoClient(host, port)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

    def write_data(self, data):
        self.collection.insert_one(data)

    def read_data(self, query):
        return self.collection.find_one(query)

    def query_data(self, query):
        return self.collection.find(query)

    def list_data(self):
        return list(self.collection.find())

    def close_connection(self):
        self.client.close()


class DatasetManager(MongoDBManager):
    def __init__(self):
        super().__init__(
            database_name='agreement-tracking',
            collection_name='datasets'
        )


# Usage
if __name__ == "__main__":
    db_manager = MongoDBManager()

    # Dummy write
    db_manager.write_data({'name': 'Alice', 'age': 25})

    # Dummy read
    print(db_manager.read_data({'name': 'Alice'}))

    # Dummy query
    for item in db_manager.query_data({'age': {'$gte': 20}}):
        print(item)

    # Dummy list
    for item in db_manager.list_data():
        print(item)

    db_manager.close_connection()
