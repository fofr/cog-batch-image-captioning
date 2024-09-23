import os
import shutil
import zipfile
import base64
import csv
import time
import google.generativeai as genai
from cog import BasePredictor, Input, Path, Secret
from openai import OpenAI, OpenAIError
from anthropic import Anthropic
from PIL import Image


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
        openai_api_key: Secret = Input(
            description="API key for OpenAI",
            default=None,
        ),
        anthropic_api_key: Secret = Input(
            description="API key for Anthropic",
            default=None,
        ),
        google_generativeai_api_key: Secret = Input(
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
        if os.path.exists("/tmp/outputs"):
            shutil.rmtree("/tmp/outputs")
        os.makedirs("/tmp/outputs")

        if model.startswith("gpt"):
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for GPT models")
            client = OpenAI(api_key=openai_api_key.get_secret_value())
        elif model.startswith("claude"):
            if not anthropic_api_key:
                raise ValueError("Anthropic API key is required for Claude models")
            client = Anthropic(api_key=anthropic_api_key.get_secret_value())
        elif model.startswith("gemini"):
            if not google_generativeai_api_key:
                raise ValueError(
                    "Google Generative AI API key is required for Gemini models"
                )
            genai.configure(api_key=google_generativeai_api_key.get_secret_value())
            client = genai.GenerativeModel(model_name=model)

        self.extract_images_from_zip(image_zip_archive, SUPPORTED_IMAGE_TYPES)

        image_count = sum(
            1
            for filename in os.listdir("/tmp/outputs")
            if filename.lower().endswith(SUPPORTED_IMAGE_TYPES)
        )
        print(f"Number of images to be captioned: {image_count}")
        print("===================================================")

        original_images = []
        for filename in os.listdir("/tmp/outputs"):
            if filename.lower().endswith(SUPPORTED_IMAGE_TYPES):
                image_path = os.path.join("/tmp/outputs", filename)
                cpy = Image.open(image_path)
                new_path = "/tmp/outputs/original_" + filename
                cpy.save(new_path)
                original_images.append("original_" + filename)

        results = []
        errors = []
        csv_path = os.path.join("/tmp/outputs", "captions.csv")
        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["caption", "image_file"])

            for filename in os.listdir("/tmp/outputs"):
                if filename.lower().endswith(SUPPORTED_IMAGE_TYPES) and filename not in original_images:
                    print(f"Processing {filename}")

                    image_path = os.path.join("/tmp/outputs", filename)
                    if resize_images_for_captioning:
                        image_path = self.resize_image_if_needed(
                            image_path, max_dimension
                        )
                    try:
                        caption = self.generate_caption(
                            image_path,
                            model,
                            client,
                            system_prompt,
                            message_prompt,
                            caption_prefix,
                            caption_suffix,
                        )
                        print(f"Caption: {caption}")

                        txt_filename = os.path.splitext(filename)[0] + ".txt"
                        txt_path = os.path.join("/tmp/outputs", txt_filename)

                        with open(txt_path, "w") as txt_file:
                            txt_file.write(caption)

                        csvwriter.writerow([caption, filename])

                        results.append({"filename": filename, "caption": caption})
                    except (OpenAIError, Exception) as e:
                        print(f"Error processing {filename}: {str(e)}")
                        errors.append({"filename": filename, "error": str(e)})
                    print("===================================================")

        output_zip_path = "/tmp/captions_and_csv.zip"
        with zipfile.ZipFile(output_zip_path, "w") as zipf:
            for root, dirs, files in os.walk("/tmp/outputs"):
                for file in files:
                    if file.endswith(".txt") or file.endswith(".csv"):
                        zipf.write(os.path.join(root, file), file)
                    elif file in original_images:
                        clean_filename = file[9:]
                        zipf.write(os.path.join(root, file), clean_filename)

        if errors:
            print("\nError Summary:")
            for error in errors:
                print(f"File: {error['filename']}, Error: {error['error']}")

        return Path(output_zip_path)

    def extract_images_from_zip(
        self, image_zip_archive: Path, supported_image_types: tuple
    ):
        with zipfile.ZipFile(image_zip_archive, "r") as zip_ref:
            for file in zip_ref.namelist():
                if (
                    file.lower().endswith(supported_image_types)
                    and not file.startswith("__MACOSX/")
                    and not os.path.basename(file).startswith("._")
                ):
                    filename = os.path.basename(file)
                    source = zip_ref.open(file)
                    target = open(os.path.join("/tmp/outputs", filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)

        print("Files extracted:")
        for root, dirs, files in os.walk("/tmp/outputs"):
            for f in files:
                print(f"{os.path.join(root, f)}")

    def resize_image_if_needed(self, image_path: str, max_dimension: int) -> str:
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
                img.save(image_path)
                print(f"Resized from {width}x{height} to {new_width}x{new_height}")
            else:
                print(
                    f"Not resizing. {width}x{height} within max dimension of {max_dimension}"
                )

        return image_path

    def generate_caption(
        self,
        image_path: str,
        model: str,
        client,
        system_prompt: str,
        message_prompt: str,
        caption_prefix: str,
        caption_suffix: str,
    ) -> str:
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

        for attempt in range(max_retries):
            try:
                if model.startswith("gpt"):
                    return self.generate_openai_caption(
                        model,
                        client,
                        system_prompt,
                        message_content,
                        image_type,
                        base64_image,
                    )
                elif model.startswith("claude"):
                    return self.generate_claude_caption(
                        model,
                        client,
                        system_prompt,
                        message_content,
                        image_type,
                        base64_image,
                    )
                elif model.startswith("gemini"):
                    return self.generate_gemini_caption(
                        client,
                        system_prompt,
                        message_content,
                        image_path,
                    )
            except (OpenAIError, Exception) as e:
                if attempt < max_retries - 1:
                    print(f"API error: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e

        raise Exception("Max retries reached. Unable to generate caption.")

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
        return message_content

    def generate_openai_caption(
        self,
        model: str,
        client,
        system_prompt: str,
        message_content: str,
        image_type: str,
        base64_image: str,
    ) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message_content,
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
            max_tokens=300,
        )
        return response.choices[0].message.content

    def generate_claude_caption(
        self,
        model: str,
        client,
        system_prompt: str,
        message_content: str,
        image_type: str,
        base64_image: str,
    ) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_type}",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": message_content,
                        },
                    ],
                },
            ],
        )
        return response.content[0].text

    def generate_gemini_caption(
        self,
        client,
        system_prompt: str,
        message_content: str,
        image_path: str,
    ) -> str:
        image = Image.open(image_path)
        prompt = f"{system_prompt}\n\n{message_content}"
        response = client.generate_content([prompt, image])
        return response.text
