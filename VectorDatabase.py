from pymilvus import MilvusClient


class VectorDatabase:
    def __init__(self, db_path: str = "./milvus_demo.db"):
        """Initialize the Milvus client."""
        self.client = MilvusClient(db_path)

    def create_collection(self, name: str, dim: int):
        """Create a collection with the specified name and dimension."""
        self.client.create_collection(collection_name=name, dimension=dim)

    def insert_data(self, collection_name: str, docs: list, vectors: list):
        """Insert documents with vectors into the collection."""
        data = [
            {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
            for i in range(len(vectors))
        ]
        self.client.insert(collection_name=collection_name, data=data)

    def search_data(self, collection_name: str, vector: list, limit: int = 2):
        """
        Search for similar vectors in the collection.
        """
        return self.client.search(
            collection_name=collection_name,
            data=[vector],
            filter="subject == 'history'",
            limit=limit,
            output_fields=["text", "subject"],
        )

    def query_data(self, collection_name: str):
        """Query the collection based on a filter."""
        return self.client.query(
            collection_name=collection_name,
            filter="subject == 'history'",
            output_fields=["text", "subject"],
        )

    def delete_data(self, collection_name: str):
        """Delete documents from the collection based on a filter."""
        return self.client.delete(
            collection_name=collection_name, filter="subject == 'history'"
        )
