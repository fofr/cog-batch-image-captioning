from operation import ImageOperation
from PIL import Image
from cog import Path
import json
import tempfile
import requests
import gc
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import asyncio
import certifi
import os
import shutil
import base64
import csv
import time
import ssl
import zipfile


SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")


class Captioner(ImageOperation):

    def validateParameters(self):
        if self.parameters.get("image_zip_archive") is None:
            raise ValueError("Image dataset was not provided for captioning")
        
        model = self.parameters.get("model")
        
        if model.startswith("gpt"):
            if not self.parameters.get("openai_api_key"):
                raise ValueError("OpenAI API key is required for GPT models")
            self.parameters["api_key"] = self.parameters.get("openai_api_key")
        elif model.startswith("claude"):
            if not self.parameters.get("anthropic_api_key"):
                raise ValueError("Anthropic API key is required for Claude models")
            self.parameters["api_key"] = self.parameters.get("anthropic_api_key")
        elif model.startswith("gemini"):
            if not self.parameters.get("google_generativeai_api_key"):
                raise ValueError(
                    "Google Generative AI API key is required for Gemini models"
                )
            self.parameters["api_key"] = self.parameters.get("google_generativeai_api_key")
        else:
            raise ValueError("Model type is not supported")
        

    def _compute(self):
        
        return asyncio.run(
            self._predict_async()
        )

    async def _predict_async(self) -> Path:
        
        temp_folder = tempfile.TemporaryDirectory(delete=False)
        
        await self._extract_images_from_zip(self.parameters.get("image_zip_archive"), SUPPORTED_IMAGE_TYPES, temp_folder)

        original_images = []
        if self.parameters.get("include_images"):
            supported_images = [filename for filename in os.listdir(temp_folder.name)
                                if filename.lower().endswith(SUPPORTED_IMAGE_TYPES)]

            for filename in supported_images:
                image_path = os.path.join(temp_folder.name, filename)
                new_path = os.path.join(temp_folder.name, f"original_{filename}")
                shutil.copy(image_path, new_path)
                original_images.append(f"original_{filename}")
            del supported_images

        captioning_requests = []
        results = []
        errors = []
        csv_path = os.path.join(temp_folder.name, "captions.csv")
        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["caption", "image_file"])

            processed_images = self._process_images(
                original_images, 
                self.parameters.get("resize_images_for_captioning"), 
                self.parameters.get("max_dimension"), 
                temp_folder
            )

            processed_images = sorted(processed_images)

            for image_path in processed_images:
                print(f" - Captioning {image_path}")
                captioning_requests.append(
                    self._generate_caption(
                        image_path,
                        self.parameters.get("model"),
                        self.parameters.get("api_key"),
                        self.parameters.get("system_prompt"),
                        self.parameters.get("message_prompt"),
                        self.parameters.get("caption_prefix"),
                        self.parameters.get("caption_suffix")
                    )
                )
            start_time = time.time()
            responses = await asyncio.gather(*captioning_requests)

            del captioning_requests

            gc.collect()

            end_time = time.time()
            print(f"Caption completed in {end_time - start_time:.2f} seconds")

            images = [filename for filename in os.listdir(temp_folder.name)
                      if filename.lower().endswith(SUPPORTED_IMAGE_TYPES) and
                      filename not in original_images
                      ]
            images = sorted(images)
            for filename, caption in zip(images, responses):
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(temp_folder.name, txt_filename)
                with open(txt_path, "w") as txt_file:
                    txt_file.write(caption)

                csvwriter.writerow([caption, filename])

                results.append({"filename": filename, "caption": caption})

            del images

            for image in processed_images:
                os.unlink(image)
            del processed_images
            gc.collect()

        output_zip_path = "/tmp/captions_and_csv.zip"
        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_folder.name):
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
        temp_folder.cleanup()
        return [Path(output_zip_path)]

    
    async def _extract_images_from_zip(
            self, image_zip_archive: str, supported_image_types: tuple, temp_folder
    ):

        with zipfile.ZipFile(image_zip_archive, "r") as zip_ref:
            idx = 0
            for file in zip_ref.namelist():
                if (file.lower().endswith(supported_image_types) and
                        not file.startswith("__MACOSX/") and
                        not os.path.basename(file).startswith("._")):
                    idx += 1
                    filename = os.path.basename(file)
                    source = zip_ref.open(file)
                    target_path = os.path.join(temp_folder.name, f"Image_{idx}.{filename.split(".")[-1]}")
                    with open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 256)
                    del source, filename, target_path
                    gc.collect()

    def _process_images(self, original_images, resize_images_for_captioning: bool, max_dimension: int, temp_folder):
        """Process images concurrently, resizing if necessary."""
        supported_images = [
            filename for filename in os.listdir(temp_folder.name)
            if filename.lower().endswith(SUPPORTED_IMAGE_TYPES) and filename not in original_images
        ]

        if resize_images_for_captioning:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for filename in supported_images:
                    image_path = os.path.join(temp_folder.name, filename)
                    futures.append(executor.submit(self._resize_image, image_path, max_dimension))

                resized_images = [future.result() for future in futures]
                del futures
                gc.collect()
        else:
            resized_images = original_images

        del supported_images
        return resized_images

    def _resize_image(self, image_path: str, max_dimension: int) -> str:
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

            os.unlink(image_path)
            jpeg_image_path = image_path.rsplit('.', 1)[0] + ".jpeg"
            img = img.convert('RGB')
            img.save(jpeg_image_path, "JPEG", quality=90)
            

        img = None
        gc.collect()
        return jpeg_image_path

    async def _generate_caption(
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

        message_content = self._prepare_message_content(
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
                            self._generate_openai_caption,
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
                            self._generate_claude_caption,
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
                            self._generate_gemini_caption,
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

    def _prepare_message_content(
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

    def _generate_openai_caption(
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

    def _generate_claude_caption(
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

    def _generate_gemini_caption(
            self,
            api_key: str,
            system_prompt: str,
            message_content: str,
            image_path: str,
    ) -> str:
        raise NotImplemented("Gemini captioning not implemented")

    