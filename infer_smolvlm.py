import json
import os
import sys
import time
from datetime import datetime

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# ---------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16

    if torch.backends.mps.is_available():
        return "mps", torch.float16

    return "cpu", torch.float32


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
    "good_assortment",
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
6. Estimate fruit/vegetable freshness and remaining usable lifetime from color, spots, shriveling, bruising, mold, dryness, or overall look.
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
- For fruit/vegetable lifetime, estimate from visible color and look; use "not enough evidence" when the image is unclear.
- Avoid hallucinations. Be concise. Every task must be visually supported.

"""

def extract_json(output_text):
    start = output_text.find("{")
    end = output_text.rfind("}") + 1

    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output")

    return json.loads(output_text[start:end])


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/ntest13.jpeg"

    overall_start_time = time.time()
    device, dtype = get_device_and_dtype()

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}, dtype: {dtype}")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        size={"longest_edge": 1024},
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        _attn_implementation="eager",
    ).to(device)

    print("Model loaded!")

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    ).to(device)

    print("Running inference...")
    inference_start_time = time.time()

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=768,
        do_sample=False,
    )

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    print("\nRAW OUTPUT:\n")
    print(output_text)

    os.makedirs("outputs", exist_ok=True)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"smolvlm_{image_name}_{timestamp}"

    raw_output_path = f"outputs/{base_filename}_raw.txt"
    with open(raw_output_path, "w") as f:
        f.write(output_text)

    try:
        parsed = extract_json(output_text)

        final_output = {
            "engine": "transformers",
            "image_name": image_name,
            "model_name": MODEL_NAME,
            "timestamp": timestamp,
            "inference_time_seconds": round(inference_time, 2),
            "parsed_output": parsed,
        }

        parsed_output_path = f"outputs/{base_filename}.json"
        with open(parsed_output_path, "w") as f:
            json.dump(final_output, f, indent=4)

        with open("outputs/result.json", "w") as f:
            json.dump(final_output, f, indent=4)

        print("\nPARSED JSON:\n")
        print(json.dumps(parsed, indent=4))
        print(f"\nSaved to {parsed_output_path}")
        print("Updated outputs/result.json")

    except Exception as error:
        failed_output_path = f"outputs/{base_filename}_failed.txt"
        with open(failed_output_path, "w") as f:
            f.write(str(error))

        print("\nFailed to parse JSON")
        print(error)
        print(f"Raw output saved to {raw_output_path}")

    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time

    print("\n-----------------------------------")
    print(f"Inference Time : {inference_time:.2f} seconds")
    print(f"Total Runtime  : {overall_time:.2f} seconds")
    print("-----------------------------------")


if __name__ == "__main__":
    main()
