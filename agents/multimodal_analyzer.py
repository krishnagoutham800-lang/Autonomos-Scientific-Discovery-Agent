import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY missing from .env")

client = genai.Client(api_key=api_key)

IMAGE_DIR = BASE_DIR / "storage" / "extracted_images"

OUTPUT_DIR = BASE_DIR / "storage" / "multimodal_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_image(image_info):

    image_path = Path(image_info["image_path"])

    prompt = """
    You are a scientific multimodal research assistant.

    Analyze this scientific research paper page.

    Identify any visible:

    1. Scientific figures and diagrams
    2. Charts and graphs
    3. Tables and experimental results
    4. Medical or scientific images
    5. Important visual relationships

    Explain what the visual information represents.

    For charts, identify visible axes, labels and trends.
    For tables, summarize important values when readable.

    Do not invent missing values or labels.
    If something is unclear, explicitly mention it.

    Return a structured analysis with:
    - Visual elements identified
    - Description
    - Important observations
    - Scientific relevance
    - Uncertainties
    """

    image_bytes = image_path.read_bytes()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            )
        ]
    )

    return response.text or ""


def analyze_all_images():

    image_files = list(IMAGE_DIR.glob("*.png"))

    if not image_files:
        print("No extracted images found.")
        return

    for image_path in image_files:

        print(f"\nAnalyzing: {image_path.name}")

        try:

            image_info = {
                "image_path": str(image_path),
                "paper_name": image_path.name
            }

            analysis = analyze_image(image_info)

            result = {
                "image": image_path.name,
                "analysis": analysis
            }

            output_path = (
                OUTPUT_DIR / f"{image_path.stem}.json"
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(f"Analysis saved: {output_path.name}")

        except Exception as error:
            print(f"Analysis error: {error}")


if __name__ == "__main__":
    analyze_all_images()