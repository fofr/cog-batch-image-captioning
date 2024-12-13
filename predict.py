import os
import shutil
import zipfile
import base64
import csv
import time
from cog import BasePredictor, Input, Path, Secret
from PIL import Image
import tempfile
import requests
import ssl
import certifi
import gc
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import asyncio
import json
from captioner import Captioner
from typing import List

SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
            self,
            image_zip_archive: Path = Input(
                description="ZIP archive containing images to process"
            ),
            caption_prefix: str = Input(
                description="Optional prefix for image captions", default=""
            ),
            caption_suffix: str = Input(
                description="Optional suffix for image captions", default=""
            ),
            resize_images_for_captioning: bool = Input(
                description="Whether to resize images for captioning. This makes captioning cheaper",
                default=True,
            ),
            include_images: bool = Input(
                description="Whether to include the original images in the response zip",
                default=False
            ),
            max_dimension: int = Input(
                description="Maximum dimension (width or height) for resized images",
                default=1024,
            ),
            model: str = Input(
                description="AI model to use for captioning. Your OpenAI or Anthropic account will be charged for usage, see their pricing pages for details.",
                choices=[
                    "gpt-4o-2024-08-06",
                    "gpt-4o-mini",
                    "gpt-4o",
                    "gpt-4-turbo",
                    "claude-3-5-sonnet-20240620",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                ],
                default="gpt-4o-2024-08-06",
            ),
            openai_api_key: str = Input(
                description="API key for OpenAI",
                default=None,
            ),
            anthropic_api_key: str = Input(
                description="API key for Anthropic",
                default=None,
            ),
            google_generativeai_api_key: str = Input(
                description="API key for Google Generative AI",
                default=None,
            ),
            system_prompt: str = Input(
                description="System prompt for image analysis",
                default="""
Write a four sentence caption for this image. In the first sentence describe the style and type (painting, photo, etc) of the image. Describe in the remaining sentences the contents and composition of the image. Only use language that would be used to prompt a text to image model. Do not include usage. Comma separate keywords rather than using "or". Precise composition is important. Avoid phrases like "conveys a sense of" and "capturing the", just use the terms themselves.

Good examples are:

"Photo of an alien woman with a glowing halo standing on top of a mountain, wearing a white robe and silver mask in the futuristic style with futuristic design, sky background, soft lighting, dynamic pose, a sense of future technology, a science fiction movie scene rendered in the Unreal Engine."

"A scene from the cartoon series Masters of the Universe depicts Man-At-Arms wearing a gray helmet and gray armor with red gloves. He is holding an iron bar above his head while looking down on Orko, a pink blob character. Orko is sitting behind Man-At-Arms facing left on a chair. Both characters are standing near each other, with Orko inside a yellow chestplate over a blue shirt and black pants. The scene is drawn in the style of the Masters of the Universe cartoon series."

"An emoji, digital illustration, playful, whimsical. A cartoon zombie character with green skin and tattered clothes reaches forward with two hands, they have green skin, messy hair, an open mouth and gaping teeth, one eye is half closed."
""",
            ),
            message_prompt: str = Input(
                description="Message prompt for image captioning",
                default="Caption this image please",
            ),
    ) -> List[Path]:

        operator = Captioner(
                image_zip_archive = image_zip_archive,
                caption_prefix = caption_prefix,
                caption_suffix = caption_suffix,
                resize_images_for_captioning = resize_images_for_captioning,
                include_images = include_images,
                max_dimension = max_dimension,
                model = model,
                openai_api_key = openai_api_key,
                anthropic_api_key = anthropic_api_key,
                google_generativeai_api_key = google_generativeai_api_key,
                system_prompt = system_prompt,
                message_prompt = message_prompt

            )

        result = operator.compute()
        del operator

        return result
        