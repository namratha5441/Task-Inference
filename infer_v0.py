import json
from PIL import Image
import torch
import time

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration
)

# ---------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# Set this to 8, 4, or None.
# 8 and 4 use bitsandbytes quantization and are mainly supported on CUDA GPUs.
# QUANTIZATION_BITS = 8
QUANTIZATION_BITS = 4
# QUANTIZATION_BITS = None


def get_quantization_config():
    if QUANTIZATION_BITS == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True
        )

    if QUANTIZATION_BITS == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    return None

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

overall_start_time = time.time()

print("Loading model...")

quantization_config = get_quantization_config()

print(f"Model: {MODEL_NAME}")
print(f"Quantization: {QUANTIZATION_BITS or 'none'}")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config
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

output_text = processor.batch_decode(
    generated_ids[:, inputs.input_ids.shape[1]:],
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
