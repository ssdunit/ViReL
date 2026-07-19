import json
import os
import re
from io import BytesIO
import openpyxl
from openpyxl.drawing.image import Image as OpenPyxlImage
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from peft import PeftModel
from data_format import dataset_format

base_model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
adapter_path = "finalmodel/"
excel_output_path = "flower_evaluation_results_new.xlsx"
thumb_dir = "temp_thumbnails"
os.makedirs(thumb_dir, exist_ok=True)

# Load label map mapping string numbers to flower names
print("Loading label mapping...")
with open(f"{adapter_path}/label_map.json", "r") as f:
  label_map = json.load(f)

print("Loading model and adapter...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=(
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ),
)
base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
processor = AutoProcessor.from_pretrained(adapter_path)
model.eval()

# Load the test split
print("Loading evaluation dataset...")
_,_,test_subset = dataset_format()

results_data = []

print("Running inference on test dataset...")
for idx, example in enumerate(tqdm(test_subset)):
  # Load and convert image
  test_image = example["image"].convert("RGB")

  # Generate 120x120 Thumbnail and save it temporarily
  thumb = test_image.copy()
  thumb.thumbnail((120, 120))
  thumb_path = os.path.join(thumb_dir, f"thumb_{idx}.png")
  thumb.save(thumb_path)

  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image"},
              {
                  "type": "text",
                  "text": (
                      "You are an expert botanist, Identify the flower species in this image and output its label ID number in valid JSON."
                  ),
              },
          ],
      }
  ]

  prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
  inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to(
      "cuda"
  )

  # Generate
  with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        eos_token_id=processor.tokenizer.eos_token_id,
        repetition_penalty=1.1,
        do_sample=False,
    )

  generated_text = processor.batch_decode(
      generated_ids, skip_special_tokens=True
  )[0]

  # Extract actual label from dataset
  actual_label_id = str(example["label"])
  actual_species = label_map.get(actual_label_id, "Unknown")

  # Initialize prediction values
  predicted_json_str = "{}"
  match_status = "INCORRECT"

  # Find the {...} string within the prompt block
  match = re.search(r"\{.*?\}", generated_text, re.DOTALL)
  if match:
    predicted_json_str = match.group(0).replace("\n", "") # Clean up string for Excel cell
    try:
      output_json = json.loads(predicted_json_str)
      if "label_id" in output_json and str(output_json["label_id"]) == actual_label_id:
        match_status = "CORRECT"
      # Fallback just in case it outputs text like your sample file:
      elif "flower_type" in output_json and str(output_json["flower_type"]).lower() == actual_species.lower():
        match_status = "CORRECT"
    except Exception:
      pass

  # Append matching row
  results_data.append({
      "Thumbnail_Path": thumb_path,
      "Actual Label": actual_species,
      "Predicted Label": predicted_json_str,
      "Match Status": match_status,
  })

print("Writing records out to formatted Excel file...")

# Create structural workbook layout
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Evaluation Report"

# Write exact headers from your reference file
headers = ["Visual Image", "Actual Label", "Predicted Label", "Match Status"]
ws.append(headers)

# Add records and embed images dynamically
for row_idx, row_data in enumerate(results_data, start=2):
  # Write text values
  ws.cell(row=row_idx, column=2, value=row_data["Actual Label"])
  ws.cell(row=row_idx, column=3, value=row_data["Predicted Label"])
  ws.cell(row=row_idx, column=4, value=row_data["Match Status"])

  # Inject image into column A
  if os.path.exists(row_data["Thumbnail_Path"]):
    img = OpenPyxlImage(row_data["Thumbnail_Path"])
    ws.add_image(img, f"A{row_idx}")

  ws.row_dimensions[row_idx].height = 95

# Set custom column widths matching the reference layout
ws.column_dimensions["A"].width = 18
ws.column_dimensions["B"].width = 25
ws.column_dimensions["C"].width = 45
ws.column_dimensions["D"].width = 15

# Save final artifact
wb.save(excel_output_path)
print(f"Sheet generated and saved to: {excel_output_path}")

# Calculate metrics summary
correct_count = sum(1 for r in results_data if r["Match Status"] == "CORRECT")
print(
    f"Accuracy: {correct_count}/{len(results_data)} ("
    f"{correct_count/len(results_data)*100:.2f}%)"
)

# Clean up temporary thumbnail folder
for r in results_data:
  if os.path.exists(r["Thumbnail_Path"]):
    os.remove(r["Thumbnail_Path"])
os.rmdir(thumb_dir)