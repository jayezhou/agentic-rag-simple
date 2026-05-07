import os
import time
import requests
import config
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

class DashScopeEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        self.batch_size = 10

    def _embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch = [t if t.strip() else " " for t in batch]

            payload = {"model": self.model, "input": batch}

            for attempt in range(3):
                try:
                    response = requests.post(self.base_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        all_embeddings.extend([item["embedding"] for item in response.json()["data"]])
                        break
                    elif response.status_code == 429:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        print(f"API Error: {response.text}")
                        response.raise_for_status()
                except Exception as e:
                    if attempt == 2: raise e
                    time.sleep(2 ** attempt)

        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

class VectorDbManager:
    __client: QdrantClient
    __dense_embeddings: DashScopeEmbeddings
    def __init__(self):
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        self.__dense_embeddings = DashScopeEmbeddings(model=config.DENSE_MODEL)

    def create_collection(self, collection_name):
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            sample_embedding = self.__dense_embeddings.embed_query("test")
            embedding_size = len(sample_embedding)

            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=embedding_size,
                    distance=qmodels.Distance.COSINE
                ),
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            print(f"✓ Collection already exists: {collection_name}")

    def clear_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Clearing all points from collection: {collection_name}")
                self.__client.delete(
                    collection_name=collection_name,
                    points_selector=qmodels.FilterSelector(filter=qmodels.Filter()),
                )
                print(f"✓ Collection cleared: {collection_name}")
        except Exception as e:
            print(f"Warning: could not clear collection {collection_name}: {e}")

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                    client=self.__client,
                    collection_name=collection_name,
                    embedding=self.__dense_embeddings,
                )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")
