from cog import BasePredictor, Input, Path, Secret
import os
import shutil
import zipfile
import base64
import csv
from openai import OpenAI
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
Write a two sentence caption for this image. Describe in the first sentence the contents and composition of the image. In the second sentence describe the style and type (painting, photo, etc) of the image. Only use language that would be used to prompt a text to image model. Do not include usage.
""",
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

        results = []
        csv_path = os.path.join("/tmp/outputs", "captions.csv")
        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["caption", "image_file"])

            for filename in os.listdir("/tmp/outputs"):
                if filename.lower().endswith(SUPPORTED_IMAGE_TYPES):
                    image_path = os.path.join("/tmp/outputs", filename)
                    if resize_images_for_captioning:
                        image_path = self.resize_image_if_needed(
                            image_path, max_dimension
                        )
                    caption = self.generate_caption(
                        image_path, model, client, system_prompt
                    )
                    full_caption = f"{caption_prefix}{caption}{caption_suffix}"

                    txt_filename = os.path.splitext(filename)[0] + ".txt"
                    txt_path = os.path.join("/tmp/outputs", txt_filename)

                    with open(txt_path, "w") as txt_file:
                        txt_file.write(full_caption)

                    csvwriter.writerow([full_caption, filename])

                    results.append({"filename": filename, "caption": full_caption})

        output_zip_path = "/tmp/captions_and_csv.zip"
        with zipfile.ZipFile(output_zip_path, "w") as zipf:
            for root, dirs, files in os.walk("/tmp/outputs"):
                for file in files:
                    if file.endswith(".txt") or file.endswith(".csv"):
                        zipf.write(os.path.join(root, file), file)

        return Path(output_zip_path)

    def extract_images_from_zip(
        self, image_zip_archive: Path, supported_image_types: tuple
    ):
        # Unzip the archive and flatten the directory structure
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

        # Print directory structure
        print("Directory structure after extraction:")
        for root, dirs, files in os.walk("/tmp/outputs"):
            level = root.replace("/tmp/outputs", "").count(os.sep)
            indent = " " * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = " " * 4 * (level + 1)
            for f in files:
                print(f"{sub_indent}{f}")

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
                    f"{image_path} size {width}x{height} within max dimension of {max_dimension}, not resizing"
                )

        return image_path

    def generate_caption(
        self, image_path: str, model: str, client: OpenAI, system_prompt: str
    ) -> str:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        image_type = os.path.splitext(image_path)[1][1:].lower()
        if image_type == "jpg":
            image_type = "jpeg"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Caption this image please",
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

        caption = response.choices[0].message.content
        print(f"{image_path}: {caption}")
        return caption
