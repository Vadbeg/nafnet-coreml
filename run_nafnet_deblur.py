"""Script for running CoreML NAFNet debur"""


import math
from pathlib import Path
from typing import Tuple

import coremltools as ct
import numpy as np
import typer
from cv2 import cv2
from PIL import Image
from tqdm import tqdm

INPUT_SHAPES = [(256 * i, 256 * i) for i in range(1, 20)]


def _add_margin(
    pil_img: Image.Image,
    top: int,
    right: int,
    bottom: int,
    left: int,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom

    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))

    return result


def _load_and_prepare_image(
    image_path: Path,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    image = Image.open(str(image_path)).convert("RGB")

    max_size_along_axis = max(image.size)
    new_size = (
        math.ceil(max_size_along_axis / 256) * 256,
        math.ceil(max_size_along_axis / 256) * 256,
    )

    if new_size not in INPUT_SHAPES:
        raise ValueError(f"Such size is not supported: {new_size}")

    add_width = new_size[0] - image.size[0]
    add_height = new_size[1] - image.size[1]

    image_region = (
        math.ceil(add_height / 2),
        math.ceil(add_width / 2),
        math.ceil(add_height / 2) + image.size[1],
        math.ceil(add_width / 2) + image.size[0],
    )
    image = _add_margin(
        pil_img=image,
        top=math.ceil(add_height / 2),
        right=math.ceil(add_width / 2),
        bottom=math.floor(add_height / 2),
        left=math.floor(add_width / 2),
    )

    return image, image_region


def _run_nafnet_deblur(
    data_root: Path = typer.Option(
        default=..., help="Path to folder with *.jpg images"
    ),
    model_path: Path = typer.Option(
        default=Path("weights/nafnet_reds_64_fp8.mlmodel"),
        help="Path to CoreML model for enlightment",
    ),
    save_root: Path = typer.Option(
        default=Path("results"), help="Path to folder for saving results"
    ),
    show_results: bool = typer.Option(
        default=False, help="Whether to show results or not"
    ),
) -> None:
    """
    Model for image debluring
    """
    save_root.mkdir(parents=True, exist_ok=True)

    model_coreml = ct.models.MLModel(str(model_path))
    image_paths = list(data_root.glob(pattern="**/*.jpg"))
    image_paths += list(data_root.glob(pattern="**/*.png"))

    for curr_image_path in tqdm(image_paths, desc="Processing images..."):
        image, image_region = _load_and_prepare_image(image_path=curr_image_path)

        result = model_coreml.predict(data={"image": image})["result"]
        result = np.uint8(result)

        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        open_cv_image = open_cv_image[
            image_region[0] : image_region[2],
            image_region[1] : image_region[3],
        ]
        result = result[
            image_region[0] : image_region[2],
            image_region[1] : image_region[3],
        ]

        combined = cv2.hconcat([open_cv_image, result])
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        save_path = save_root / curr_image_path.name
        cv2.imwrite(str(save_path), combined)

        if show_results:
            cv2.imshow("Combined", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    typer.run(_run_nafnet_deblur)
