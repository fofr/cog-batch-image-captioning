import os
import shutil
import zipfile
import base64
import csv
import time
from cog import BasePredictor, Input, Path, Secret
from openai import OpenAI, OpenAIError
from PIL import Image


SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


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
            description="OpenAI model to use. Your OpenAI account will be charged for usage, see pricing: https://openai.com/api/pricing/",
            choices=["gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            default="gpt-4o-2024-08-06",
        ),
        openai_api_key: Secret = Input(
            description="OpenAI API key",
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

        key = (
            openai_api_key.get_secret_value()
            if openai_api_key
            else os.environ.get("OPENAI_API_KEY")
        )

        if not key:
            raise ValueError("OpenAI API key is required")

        client = OpenAI(api_key=key)

        self.extract_images_from_zip(image_zip_archive, SUPPORTED_IMAGE_TYPES)

        image_count = sum(
            1
            for filename in os.listdir("/tmp/outputs")
            if filename.lower().endswith(SUPPORTED_IMAGE_TYPES)
        )
        print(f"Number of images to be captioned: {image_count}")
        print("===================================================")

        results = []
        errors = []
        csv_path = os.path.join("/tmp/outputs", "captions.csv")
        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["caption", "image_file"])

            for filename in os.listdir("/tmp/outputs"):
                if filename.lower().endswith(SUPPORTED_IMAGE_TYPES):
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
                    except OpenAIError as e:
                        print(f"Error processing {filename}: {str(e)}")
                        errors.append({"filename": filename, "error": str(e)})
                    except Exception as e:
                        print(f"Unexpected error processing {filename}: {str(e)}")
                        errors.append({"filename": filename, "error": str(e)})
                    print("===================================================")

        output_zip_path = "/tmp/captions_and_csv.zip"
        with zipfile.ZipFile(output_zip_path, "w") as zipf:
            for root, dirs, files in os.walk("/tmp/outputs"):
                for file in files:
                    if file.endswith(".txt") or file.endswith(".csv"):
                        zipf.write(os.path.join(root, file), file)

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
        client: OpenAI,
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

        # Prepare the message content based on prefix and suffix
        message_content = message_prompt
        if caption_prefix and caption_suffix:
            message_content += f"\n\nPlease prefix the caption with '{caption_prefix}' and suffix it with '{caption_suffix}', ensuring correct grammar and flow. Do not change the prefix or suffix."
        elif caption_prefix:
            message_content += f"\n\nPlease prefix the caption with '{caption_prefix}', ensuring correct grammar and flow. Do not change the prefix."
        elif caption_suffix:
            message_content += f"\n\nPlease suffix the caption with '{caption_suffix}', ensuring correct grammar and flow. Do not change the suffix."

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
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
            except OpenAIError as e:
                if attempt < max_retries - 1:
                    print(
                        f"OpenAI API error: {str(e)}. Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    raise e

        raise Exception("Max retries reached. Unable to generate caption.")
