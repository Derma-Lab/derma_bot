import asyncio
import logging
import random

import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from derma_bot.retrival.core import logger, init_chroma

N = 10

class SyntheticDocument(BaseModel):
        symptoms: str
        diagnosis: str

async def generate_document() -> str:
    symptoms_list = [
        "Red, inflamed bumps on face and body",
        "Whiteheads and blackheads appearing frequently",
        "Painful, deep cysts under the skin",
        "Oily skin with frequent breakouts",
        "Small red bumps mainly on forehead and cheeks",
        "Pus-filled pimples that are tender to touch",
        "Recurring breakouts during menstrual cycle",
        "Bumps that leave dark spots after healing",
        "Skin feels rough with many small bumps",
        "Large painful nodules under the skin"
    ]

    diagnoses_list = [
        "Moderate inflammatory acne requiring topical treatment",
        "Mild comedonal acne - basic skincare needed",
        "Severe nodular acne requiring oral medication",
        "Hormonal acne related to sebum production",
        "Mild papular acne responding to OTC treatment",
        "Moderate pustular acne needing antibiotics",
        "Hormonal acne linked to menstrual changes",
        "Post-inflammatory hyperpigmentation with active acne",
        "Comedonal acne with inflammatory components",
        "Severe cystic acne requiring specialist care"
    ]

    idx = random.randint(0, len(symptoms_list) - 1)
    document = SyntheticDocument(
        symptoms=symptoms_list[idx],
        diagnosis=diagnoses_list[idx]
    )

    return f"""
        symptoms
        {document.symptoms}

        diagnoses
        {document.diagnosis}
    """


async def main():
    client, collection, model = init_chroma()
    
    for i in range(N):
        document = await generate_document()

        logger.info(f"Encoding document: {document}")
        embedding = model.encode([document])
        
        logger.info(f"Adding document {i+1} to collection")
        collection.add(
            documents=[document],
            embeddings=embedding.tolist(),
            ids=[f"doc_{i}"],
            metadatas=[{"source": "example"}]
        )


if __name__ == "__main__":
    asyncio.run(main())