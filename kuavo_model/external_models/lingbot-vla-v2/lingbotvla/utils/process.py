from typing import TYPE_CHECKING, Any, Callable, Dict
from PIL import Image
import os
import torch
from lingbotvla.data.multimodal.preprocess import conv_preprocess
from lingbotvla.data.constants import IMAGE_INPUT_INDEX
if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from lingbotvla.data.chat_template import ChatTemplate

MAX_PIXELS = 768 * 28 * 28
ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
}

def process_sample(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    data_root: str,
    **kwargs,
):
    """
    Processes multimodal example with qwen2vl's pre-processor.
    """
    source = (
        kwargs["source_name"] if "source_name" in kwargs else sample["source"]
    )  # source_name if use multisource_dataset
    conversations = sample["conversations"] if "conversations" in sample else sample["text"]  # text-only data
    conversations = conv_preprocess(source, conversations, **kwargs)

    token_num_inputs, image_inputs = {}, {}
    image_grid_thw = None
    if "image" in sample:
        images = []
        with open(os.path.join(data_root,sample["image"]), 'rb') as f:
            images.append(Image.open(f).convert("RGB"))

        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num
    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs)
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    input_ids = tokenized_example["input_ids"]

    position_ids = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
    )["position_ids"]
    tokenized_example["position_ids"] = position_ids.squeeze()  # (dim, l)

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example.update(image_inputs)
    return [tokenized_example]
