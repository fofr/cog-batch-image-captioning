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

SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")


class Predictor(BasePredictor):
    def setup(self):
        self.temp_folder = tempfile.TemporaryDirectory(delete=False)

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
    ) -> Path:
        return asyncio.run(
            self.predict_async(
                image_zip_archive,
                caption_prefix,
                caption_suffix,
                resize_images_for_captioning,
                include_images,
                max_dimension,
                model,
                openai_api_key,
                anthropic_api_key,
                google_generativeai_api_key,
                system_prompt,
                message_prompt
            )
        )

    async def predict_async(
            self,
            image_zip_archive,
            caption_prefix,
            caption_suffix,
            resize_images_for_captioning,
            include_images,
            max_dimension,
            model,
            openai_api_key,
            anthropic_api_key,
            google_generativeai_api_key,
            system_prompt,
            message_prompt,
    ) -> Path:

        if model.startswith("gpt"):
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for GPT models")
            api_key = openai_api_key
        elif model.startswith("claude"):
            if not anthropic_api_key:
                raise ValueError("Anthropic API key is required for Claude models")
            api_key = anthropic_api_key
        elif model.startswith("gemini"):
            if not google_generativeai_api_key:
                raise ValueError(
                    "Google Generative AI API key is required for Gemini models"
                )
            api_key = google_generativeai_api_key

        if not (model.startswith("gpt") or model.startswith("claude") or model.startswith("gemini")):
            raise ValueError("Model type is not supported")

        await self.extract_images_from_zip(image_zip_archive, SUPPORTED_IMAGE_TYPES)

        original_images = []
        if include_images:
            supported_images = [filename for filename in os.listdir(self.temp_folder.name)
                                if filename.lower().endswith(SUPPORTED_IMAGE_TYPES)]
            for filename in supported_images:
                image_path = os.path.join(self.temp_folder.name, filename)
                new_path = os.path.join(self.temp_folder.name, f"original_{filename}")
                shutil.copy(image_path, new_path)
                original_images.append(f"original_{filename}")
            del supported_images

        captioning_requests = []
        results = []
        errors = []
        csv_path = os.path.join(self.temp_folder.name, "captions.csv")
        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["caption", "image_file"])

            images = self.process_images(
                original_images, resize_images_for_captioning, max_dimension
            )
            for image_path in images:
                captioning_requests.append(
                    self.generate_caption(
                        image_path,
                        model,
                        api_key,
                        system_prompt,
                        message_prompt,
                        caption_prefix,
                        caption_suffix,
                    )
                )
            start_time = time.time()
            responses = await asyncio.gather(*captioning_requests)
            del captioning_requests
            for image in images:
                os.unlink(image)

            gc.collect()

            end_time = time.time()
            print(f"Caption completed in {end_time - start_time:.2f} seconds")

            images = [filename for filename in os.listdir(self.temp_folder.name)
                      if filename.lower().endswith(SUPPORTED_IMAGE_TYPES) and
                      filename not in original_images
                      ]

            for filename, caption in zip(images, responses):
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(self.temp_folder.name, txt_filename)
                with open(txt_path, "w") as txt_file:
                    txt_file.write(caption)

                csvwriter.writerow([caption, filename])

                results.append({"filename": filename, "caption": caption})

            del images
            gc.collect()

        output_zip_path = "/tmp/captions_and_csv.zip"
        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.temp_folder.name):
                root_path = os.path.abspath(root)  # Cache root path
                for file in files:
                    file_path = os.path.join(root_path, file)
                    if file.endswith((".txt", ".csv")):
                        zipf.write(file_path, file)
                    elif file in original_images:
                        clean_filename = file[9:]
                        zipf.write(file_path, clean_filename)

                gc.collect()

        if errors:
            print("\nError Summary:")
            for error in errors:
                print(f"File: {error['filename']}, Error: {error['error']}")

        del original_images
        del errors

        gc.collect()
        self.temp_folder.cleanup()
        return Path(output_zip_path)

    async def download_zip(self, url):
        """Download the zip file asynchronously using aiohttp."""
        temp_zip_path = os.path.join(self.temp_folder.name, "downloaded_zipfile.zip")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        conn = aiohttp.TCPConnector(ssl=ssl_context)
        start_time = time.time()
        async with aiohttp.ClientSession(connector=conn) as session:
            async with session.get(url) as response:
                with open(temp_zip_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        gc.collect()
                        await asyncio.sleep(0.01)

        end_time = time.time()
        print(f"Download completed in {end_time - start_time:.2f} seconds")
        gc.collect()
        return temp_zip_path

    async def extract_images_from_zip(
            self, image_zip_archive: str, supported_image_types: tuple
    ):

        with zipfile.ZipFile(image_zip_archive, "r") as zip_ref:
            for file in zip_ref.namelist():
                if (file.lower().endswith(supported_image_types) and
                        not file.startswith("__MACOSX/") and
                        not os.path.basename(file).startswith("._")):
                    filename = os.path.basename(file)
                    source = zip_ref.open(file)
                    target_path = os.path.join(self.temp_folder.name, filename)
                    with open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 256)
                    del source, filename, target_path
                    gc.collect()

    def process_images(self, original_images, resize_images_for_captioning: bool, max_dimension: int):
        """Process images concurrently, resizing if necessary."""
        supported_images = [
            filename for filename in os.listdir(self.temp_folder.name)
            if filename.lower().endswith(SUPPORTED_IMAGE_TYPES) and filename not in original_images
        ]

        if resize_images_for_captioning:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for filename in supported_images:
                    image_path = os.path.join(self.temp_folder.name, filename)
                    futures.append(executor.submit(self.resize_image, image_path, max_dimension))

                resized_images = [future.result() for future in futures]
                del futures
                gc.collect()
        else:
            resized_images = original_images

        del supported_images
        return resized_images

    def resize_image(self, image_path: str, max_dimension: int) -> str:
        with Image.open(image_path) as img:
            width, height = img.size

            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int((height / width) * max_dimension)
                else:
                    new_height = max_dimension
                    new_width = int((width / height) * max_dimension)

                img = img.resize((new_width, new_height), Image.LANCZOS)

            jpeg_image_path = image_path.rsplit('.', 1)[0] + ".jpeg"
            img = img.convert('RGB')
            img.save(jpeg_image_path, "JPEG", quality=90)

        img = None
        gc.collect()
        return jpeg_image_path

    async def generate_caption(
            self,
            image_path: str,
            model: str,
            api_key,
            system_prompt: str,
            message_prompt: str,
            caption_prefix: str,
            caption_suffix: str,
    ):

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        image_type = os.path.splitext(image_path)[1][1:].lower()
        if image_type == "jpg":
            image_type = "jpeg"

        message_content = self.prepare_message_content(
            message_prompt, caption_prefix, caption_suffix
        )

        max_retries = 3
        retry_delay = 5

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Offloading to a thread to make the sync function "feel" async
            for attempt in range(max_retries):
                try:
                    if model.startswith("gpt"):
                        result = await loop.run_in_executor(
                            pool,
                            self.generate_openai_caption,
                            model,
                            api_key,
                            system_prompt,
                            message_content,
                            image_type,
                            base64_image
                        )
                        break
                    elif model.startswith("claude"):
                        result = await loop.run_in_executor(
                            pool,
                            self.generate_claude_caption,
                            model,
                            api_key,
                            system_prompt,
                            message_content,
                            image_type,
                            base64_image
                        )
                        break
                    elif model.startswith("gemini"):
                        result = await loop.run_in_executor(
                            pool,
                            self.generate_gemini_caption,
                            api_key,
                            system_prompt,
                            message_content,
                            image_path
                        )
                        break
                except (Exception) as e:
                    if attempt < max_retries - 1:
                        print(f"API error: {str(e)}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise e
            return result

    def prepare_message_content(
            self, message_prompt: str, caption_prefix: str, caption_suffix: str
    ) -> str:
        message_content = message_prompt
        if caption_prefix and caption_suffix:
            message_content += f"\n\nPlease prefix the caption with '{caption_prefix}' and suffix it with '{caption_suffix}', ensuring correct grammar and flow. Do not change the prefix or suffix."
        elif caption_prefix:
            message_content += f"\n\nPlease prefix the caption with '{caption_prefix}', ensuring correct grammar and flow. Do not change the prefix."
        elif caption_suffix:
            message_content += f"\n\nPlease suffix the caption with '{caption_suffix}', ensuring correct grammar and flow. Do not change the suffix."
        return str(message_content)

    def generate_openai_caption(
            self,
            model: str,
            api_key: str,
            system_prompt: str,
            message_content: str,
            image_type: str,
            base64_image: str,
    ) -> str:
        url = "https://api.openai.com/v1/chat/completions"  # Example endpoint, adjust accordingly
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message_content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_type};base64,{base64_image}",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 300
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Failed to get caption: {response.text}")

    def generate_claude_caption(
            self,
            model: str,
            api_key: str,
            system_prompt: str,
            message_content: str,
            image_type: str,
            base64_image: str,
    ) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": f"{api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "max_tokens": 300,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_type}",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": message_content
                        }
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            raise Exception(f"Failed to get caption: {response.text}")

    def generate_gemini_caption(
            self,
            api_key: str,
            system_prompt: str,
            message_content: str,
            image_path: str,
    ) -> str:
        raise NotImplemented("Gemini captioning not implemented")