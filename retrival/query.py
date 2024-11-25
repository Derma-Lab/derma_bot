import logging

from derma_bot.retrival.core import init_chroma

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def query(query_text: str):
    client, collection, model = init_chroma()
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    logger.info("Query results:")
    for result in results['documents']:
        print(result)
        logger.debug(f"Result document: {result}")



if __name__ == "__main__":
    query("What is acne?")