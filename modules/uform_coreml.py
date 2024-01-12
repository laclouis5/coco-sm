import coremltools as ct
import torch
import uform
from PIL import Image
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = uform.get_model("unum-cloud/uform-vl-multilingual-v2").eval()

        self.image_encoder = ct.models.MLModel(
            "../build/uform-vl-multilingual-v2_image-encoder.mlpackage",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        self.text_encoder = ct.models.MLModel(
            "../build/uform-vl-multilingual-v2_text-encoder.mlpackage",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        self.max_length = self.model.text_encoder.max_position_embeddings

    def preprocess_image(self, image: Image.Image | list[Image.Image]) -> torch.Tensor:
        return self.model.preprocess_image(image)

    def preprocess_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        return self.model.preprocess_text(text)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        predictions = self.image_encoder.predict(
            [{"image": img.unsqueeze(0)} for img in image]
        )

        return torch.cat(
            [torch.from_numpy(pred["embeddings"]) for pred in predictions],
            dim=0,
        )

    def encode_text(self, text: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = text["input_ids"][:, : self.max_length].to(dtype=torch.int32)
        attention_masks = text["attention_mask"][:, : self.max_length]

        predictions = self.text_encoder.predict(
            [
                {"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0)}
                for ids, mask in zip(input_ids, attention_masks)
            ]
        )

        return torch.cat(
            [torch.from_numpy(pred["embeddings"]) for pred in predictions],
            dim=0,
        )


def image_forward_fn(
    model: Model, images: Image.Image | list[Image.Image], device, transform
):
    images = model.preprocess_image(images)
    return model.encode_image(images)


def text_forward_fn(model: Model, texts: str | list[str], device, transform):
    texts = model.preprocess_text(texts)
    return model.encode_text(texts)


embedding_dim = 256
image_preprocess = None
text_preprocess = None

model = Model()
