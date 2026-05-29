import dataclasses
import random
from collections.abc import Sequence

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_kuavo_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "state": np.random.rand(16),
        "cam_h": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        # "cam_r": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "cam_l": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class KuavoLeftInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

        # Enable prompt augmentation for training (randomize prompt phrasing)
    # use_prompt_augmentation: bool = False
    # List of prompt templates for augmentation (only used when use_prompt_augmentation=True)
    # prompt_templates: Sequence[str] | None = None

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["cam_h"])
        # wrist_r_image = _parse_image(data["cam_r"])
        wrist_l_image = _parse_image(data["cam_l"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_l_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        # if self.use_prompt_augmentation:
        #     inputs["prompt"] = self._get_random_conditional_prompt()
        # elif "prompt" in data:
        inputs["prompt"] = data["prompt"]

        return inputs

    # def _get_random_conditional_prompt(self) -> str:
    #     """
    #     Generate randomized conditional prompts from the provided templates.
        
    #     Returns:
    #         A randomly selected prompt from the prompt_templates list.
            
    #     Raises:
    #         ValueError: If prompt_templates is None or empty when augmentation is enabled.
    #     """
    #     if not self.prompt_templates:
    #         raise ValueError(
    #             "prompt_templates must be provided when use_prompt_augmentation=True. "
    #             "Please provide a list of prompt templates in the config."
    #         )
    #     return random.choice(self.prompt_templates)


@dataclasses.dataclass(frozen=True)
class KuavoLeftOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :8])}
