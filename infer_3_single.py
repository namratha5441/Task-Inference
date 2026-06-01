import json
from PIL import Image
import torch
import time

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor
)

# ---------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------

# Uncomment one model at a time to test.
# MODEL_NAME = "google/gemma-4-E4B-it"
# MODEL_NAME = "Qwen/Qwen3.5-2B"
MODEL_NAME = "google/gemma-4-E2B-it"
# MODEL_NAME = "Qwen/Qwen3.5-0.8B"
# MODEL_NAME = "LiquidAI/LFM2.5-VL-1.6B"
# MODEL_NAME = "LiquidAI/LFM2.5-VL-450M"
# MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

overall_start_time = time.time()

print("Loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

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
    "monitor",
    "restock",
    "front_face",
    "rotate_stock",
    "check_expiry",
    "freshness_check",
    "good_assortment"
]

# ---------------------------------------------------
# PROMPT
# ---------------------------------------------------

SYSTEM_PROMPT = f"""
You are an expert retail shelf and fresh produce inspection AI.

Analyze the image carefully and use only visual evidence from the image.
Return STRICT JSON ONLY. Do not include markdown, comments, or extra text.

Main rule:
- If the goods/shelves look properly stocked and arranged, and there is no clearly visible issue, reply with "good assortment".
- Do not suggest expiry checks, freshness checks, restocking, or front-facing unless that issue is clearly visible.
- Do not list uncertain issues such as "may be empty" or "difficult to confirm".

- Strictly check for cleaning tasks

Inspection goals:
1. Describe the retail scene and important products/racks.
2. Detect empty or nearly empty racks/shelves.
3. Detect whether visible products have been sold from the front and need to be pulled forward.
4. Detect text in the image, including product labels, price tags, manufacture dates, expiry dates, best-before dates, or shelf-life text.
5. Check goods expiration from detected text when readable.
6. Classify fruit/vegetable ripeness and freshness from visible state: color, green/yellow/red/brown areas, dark spots, bruising, shriveling, mold, dryness, firmness cues, damaged skin, and overall look.
7. Check for spills, leakage, stains, trash, dust, dirt, damaged packaging residue, or other cleanliness issues.
8. If the shelf/rack is properly arranged and no issue is visible, say "good assortment" and do not add extra checks.

Allowed task names:
{ALLOWED_TASKS}

Required JSON schema:
{{
    "scene_description": "short description of the image",
    "possible_issues": [],
    "suggested_tasks": [
        {{
            "task": "one allowed task name",
            "instruction": "specific action to take",
            "evidence": "visual evidence supporting the task"
        }}
    ],
    "overall_assessment": "good assortment or short issue summary",
    "confidence": "low/medium/high" 
}}

Decision rules:
- If racks or shelf areas are clearly empty, add a restock task with instruction "refill the stock".
- If items are clearly missing from the front row or pushed back after front items were sold, add a front_face task with instruction "place the items at the front".
- If liquid spills, leakage, stains, trash, dust, dirt, or mess are clearly visible, add a clean task with instruction "clean the area".
- If products look properly stocked and arranged, set overall_assessment to "good assortment", keep possible_issues empty, and use exactly one good_assortment task.
- If text is unreadable, say not_readable instead of guessing dates or product names.
- For expiration checks, only mark expired or near_expiry when readable text supports it.
- Do not suggest check_expiry when no expiry date is readable.
- Do not suggest freshness_check unless fruits or vegetables are clearly visible.
- For fruits/vegetables, always fill produce_ripeness when visible.
- Use this ripeness guide:
  under-ripe = mostly green, pale, hard-looking, not ready.
  barely ripe = mostly final color but still slightly green/pale, few or no spots.
  ripe = expected fresh color, firm-looking, minimal blemishes, ready to sell/eat.
  very ripe = darker color, many brown/dark speckles, small bruises, use soon or sell first.
  overripe = large brown/black areas, heavy spotting, soft/shriveled look, poor quality.
  spoiled = mold, leaking liquid, collapsed texture, severe rot, unsafe-looking.
- Estimate fruit/vegetable remaining lifetime from visible color and look; use "uncertain" when the image is unclear.
- Avoid hallucinations. Be concise. Every task must be visually supported.


"""

# ---------------------------------------------------
# IMAGE PATH
# ---------------------------------------------------

IMAGE_PATH = "data/ntest13.jpeg"

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

generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

# ---------------------------------------------------
# CLEAN OUTPUT
# ---------------------------------------------------

print("\nRAW OUTPUT:\n")
print(output_text)

# Try extracting JSON
try:
    start = output_text.find("{")
    end = output_text.rfind("}") + 1

    json_text = output_text[start:end]

    parsed = json.loads(json_text)

    print("\nPARSED JSON:\n")
    print(json.dumps(parsed, indent=4))

    # Save output
    with open("outputs/result.json", "w") as f:
        json.dump(parsed, f, indent=4)

    print("\nSaved to outputs/result.json")

except Exception as e:
    print("\nFailed to parse JSON")
    print(e)


# ___________________________________________________________

overall_end_time = time.time()

overall_time = overall_end_time - overall_start_time

print("\n-----------------------------------")
print(f"Inference Time : {inference_time:.2f} seconds")
print(f"Total Runtime  : {overall_time:.2f} seconds")
print("-----------------------------------")






# "detected_objects": [
    #     {{
    #         "name": "object or product name",
    #         "location": "where it appears",
    #         "condition": "visible condition"
    #     }}
    # ],
    # "detected_text": [
    #     {{
    #         "text": "exact readable text",
    #         "location": "where the text appears",
    #         "purpose": "label/price/manufacture_date/expiry_date/best_before/unknown"
    #     }}
    # ],
    # "stock_status": {{
    #     "racks_empty": true,
    #     "racks_low_or_gaps_visible": true,
    #     "finding": "what is visible",
    #     "action": "refill the stock / monitor / no action"
    # }},
    # "front_facing_status": {{
    #     "front_items_sold_or_recessed": true,
    #     "finding": "what is visible",
    #     "action": "place the items at the front / no action"
    # }},
    # "expiration_check": {{
    #     "readable_dates_found": true,
    #     "detected_dates": [],
    #     "status": "valid/expired/near_expiry/not_readable/not_applicable",
    #     "reason": "brief reason based on readable text"
    # }},
    # "produce_lifetime_estimate": [
    #     {{
    #         "item": "fruit or vegetable name",
    #         "visual_freshness": "fresh/usable soon/overripe/spoiled/uncertain",
    #         "estimated_remaining_lifetime": "example: 1-2 days, 3-5 days, not enough evidence",
    #         "reason": "color/look based reason"
    #     }}
    # ],





# Good assortment output example:
# {{
#     "scene_description": "Store shelves with goods arranged and stocked.",
#     "possible_issues": [],
#     "suggested_tasks": [
#         {{
#             "task": "good_assortment",
#             "instruction": "good assortment",
#             "evidence": "Goods appear stocked and arranged with no clear visible issue."
#         }}
#     ],
#     "overall_assessment": "good assortment",
#     "confidence": "high"
# }}


# "produce_ripeness": [
#         {{
#             "item": "fruit or vegetable name",
#             "ripeness_stage": "under-ripe/barely ripe/ripe/very ripe/overripe/spoiled/uncertain",
#             "visual_evidence": "color, spots, bruising, shriveling, mold, or texture cues",
#             "estimated_remaining_lifetime": "example: not ready, 1-2 days, 3-5 days, use immediately, discard, uncertain",
#             "action": "wait / sell first / use soon / discount / remove / no action"
#         }}
#     ],