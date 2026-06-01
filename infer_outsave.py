import os
import json
from datetime import datetime
from PIL import Image
import torch
import time

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

# ---------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

overall_start_time = time.time()

print("Loading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Model loaded!")

# ---------------------------------------------------
# TASK LIST
# ---------------------------------------------------

ALLOWED_TASKS = [
    "clean",
    "repair",
    "replace",
    "inspect",
    "organize",
    "remove",
    "install",
    "arrange",
    "check",
    "monitor"
]

# ---------------------------------------------------
# PROMPT
# ---------------------------------------------------

SYSTEM_PROMPT = f"""
You are an expert visual reasoning AI.

Analyze the image carefully.

Your job:
1. Describe the scene
2. Detect important objects
3. Detect possible issues/problems
4. Infer actionable tasks

Allowed tasks:
{ALLOWED_TASKS}

Return STRICT JSON ONLY.

Schema:
{{
    "scene_description": "",
    "detected_objects": [],
    "possible_issues": [],
    "suggested_tasks": [],
    "confidence": "low/medium/high"
}}

Rules:
- Only output valid JSON
- Avoid hallucinations
- Be concise
- Tasks must be visually supported
"""

# ---------------------------------------------------
# IMAGE PATH
# ---------------------------------------------------

IMAGE_PATH = "images/test2.jpg"

image = Image.open(IMAGE_PATH).convert("RGB")

# ---------------------------------------------------
# BUILD MESSAGE
# ---------------------------------------------------

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": SYSTEM_PROMPT
            }
        ]
    }
]

# ---------------------------------------------------
# PREPARE INPUTS
# ---------------------------------------------------

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
)

inputs = inputs.to(model.device)

# ---------------------------------------------------
# GENERATE
# ---------------------------------------------------

inference_start_time = time.time()

print("Running inference...")

generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False
)

inference_end_time = time.time()

inference_time = inference_end_time - inference_start_time

# ---------------------------------------------------
# DECODE OUTPUT
# ---------------------------------------------------

output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

# ---------------------------------------------------
# PRINT RAW OUTPUT
# ---------------------------------------------------

print("\nRAW OUTPUT:\n")
print(output_text)

# ---------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ---------------------------------------------------

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------
# GENERATE UNIQUE FILE NAME
# ---------------------------------------------------

image_name = os.path.splitext(
    os.path.basename(IMAGE_PATH)
)[0]

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

base_filename = f"{image_name}_{timestamp}"

# ---------------------------------------------------
# SAVE RAW OUTPUT
# ---------------------------------------------------

raw_output_path = (
    f"outputs/{base_filename}_raw.txt"
)

with open(raw_output_path, "w") as f:
    f.write(output_text)

print(f"\nRaw output saved to: {raw_output_path}")

# # ---------------------------------------------------
# # TRY EXTRACTING JSON
# # ---------------------------------------------------

# try:

#     start = output_text.find("{")
#     end = output_text.rfind("}") + 1

#     json_text = output_text[start:end]

#     parsed = json.loads(json_text)

#     print("\nPARSED JSON:\n")
#     print(json.dumps(parsed, indent=4))

#     # ---------------------------------------------------
#     # FINAL JSON STRUCTURE
#     # ---------------------------------------------------

#     final_output = {
#         "image_name": image_name,
#         "image_path": IMAGE_PATH,
#         "model_name": MODEL_NAME,
#         "timestamp": timestamp,
#         "inference_time_seconds": round(
#             inference_time, 2
#         ),
#         "parsed_output": parsed
#     }

#     # ---------------------------------------------------
#     # SAVE PARSED JSON
#     # ---------------------------------------------------

#     parsed_output_path = (
#         f"outputs/{base_filename}.json"
#     )

#     with open(parsed_output_path, "w") as f:
#         json.dump(final_output, f, indent=4)

#     print(
#         f"\nParsed JSON saved to: "
#         f"{parsed_output_path}"
#     )

# except Exception as e:

#     print("\nFailed to parse JSON")
#     print(e)

#     # ---------------------------------------------------
#     # SAVE FAILURE DETAILS
#     # ---------------------------------------------------

#     failed_output_path = (
#         f"outputs/{base_filename}_failed.txt"
#     )

#     with open(failed_output_path, "w") as f:

#         f.write("FAILED TO PARSE JSON\n\n")
#         f.write(str(e))
#         f.write("\n\nRAW OUTPUT:\n\n")
#         f.write(output_text)

#     print(
#         f"\nFailure details saved to: "
#         f"{failed_output_path}"
#     )

# ---------------------------------------------------
# TOTAL EXECUTION TIME
# ---------------------------------------------------

overall_end_time = time.time()

overall_time = (
    overall_end_time - overall_start_time
)

# ---------------------------------------------------
# PERFORMANCE SUMMARY
# ---------------------------------------------------

print("\n-----------------------------------")
print(
    f"Inference Time : "
    f"{inference_time:.2f} seconds"
)
print(
    f"Total Runtime  : "
    f"{overall_time:.2f} seconds"
)
print("-----------------------------------")