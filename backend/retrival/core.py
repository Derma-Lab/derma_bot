import logging

import chromadb
from chromadb.errors import InvalidCollectionException
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_chroma():
    logger.info("Initializing ChromaDB client")
    client = chromadb.PersistentClient(path="./data")
    collection_name = "synthetic_documents"
    try:
        collection = client.get_collection(name=collection_name)
    except InvalidCollectionException:
        collection = client.create_collection(name=collection_name)

    logger.info("Loading sentence transformer model")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    return client, collection, model