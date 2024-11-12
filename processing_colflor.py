from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import BatchFeature

from processing_florence2 import Florence2Processor

from processing_utils import BaseVisualRetrieverProcessor


class ColFlorProcessor(BaseVisualRetrieverProcessor, Florence2Processor):
    """
    Processor for ColPali.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_image = Image.new("RGB", (16, 16), color="black")

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColFlor2.
        """
        texts_doc = ["<OCR>"] * len(images)
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
        )

        new_part = torch.ones((batch_doc['attention_mask'].size()[0], 577)).to(batch_doc['attention_mask'].device)
        batch_doc['full_attention_mask'] = torch.cat([new_part, batch_doc['attention_mask']], dim=1)

        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColFlor2.
        """
        if suffix is None:
            suffix = "<pad>" * 10
        texts_query: List[str] = []

        for query in queries:
            query = f"Question: {query}"
            query += suffix  # add suffix (pad tokens)
            texts_query.append(query)

        batch_query = self.tokenizer(
            #images=[self.mock_image] * len(texts_query),
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length= max_length + self.image_seq_length,
        )

        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)