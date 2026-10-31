from ingestion.multimodal_extractor import (
    extract_all_paper_images
)

from agents.multimodal_analyzer import (
    analyze_all_images
)


print("\nSTEP 1: Extracting PDF pages...")

images = extract_all_paper_images()

print(f"\nTotal page images extracted: {len(images)}")

if images:

    print("\nSTEP 2: Starting Gemini Vision analysis...")

    analyze_all_images()

    print("\nPhase 8 completed successfully!")

else:
    print("\nNo page images were extracted.")