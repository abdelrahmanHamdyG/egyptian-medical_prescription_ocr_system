from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# 1. Load the processor and model
# 'microsoft/trocr-base-handwritten' is best for handwriting
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# 2. Load and preprocess the image (must be a cropped line of text)
image_path = "output_eval/0_100_crop_1.jpg"  # Replace with your actual file path
image = Image.open(image_path).convert("RGB")

# 3. Predict
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Recognized text: {generated_text}")

