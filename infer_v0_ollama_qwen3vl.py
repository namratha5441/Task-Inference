import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request


# ---------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------

MODEL_NAME = "qwen3-vl:2b-instruct"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

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
/no_think

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

IMAGE_PATH = "data/ntest17.jpeg"
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]


def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def check_ollama_connection():
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=10) as response:
            response_data = json.loads(response.read().decode("utf-8"))

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not connect to local Ollama at "
            f"{OLLAMA_BASE_URL}. Start Ollama first, then run "
            f"`ollama pull {MODEL_NAME}`."
        ) from error

    local_models = [
        model.get("name")
        for model in response_data.get("models", [])
    ]

    if MODEL_NAME not in local_models:
        print(
            f"Warning: {MODEL_NAME} was not found in local Ollama models. "
            f"Run `ollama pull {MODEL_NAME}` if inference fails."
        )


def call_ollama_chat(prompt, image_base64):
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64]
            }
        ],
        "options": {
            "temperature": 0,
            "num_predict": 512
        }
    }

    request = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            response_data = json.loads(response.read().decode("utf-8"))

            os.makedirs("outputs", exist_ok=True)
            with open("outputs/ollama_last_response.json", "w") as f:
                json.dump(response_data, f, indent=4)

            message = response_data.get("message", {})
            content = message.get("content", "")

            if content.strip():
                return content

            thinking = message.get("thinking", "")
            if thinking.strip():
                print(
                    "Ollama returned empty message.content but non-empty "
                    "message.thinking. Using message.thinking as fallback."
                )
                return thinking

            raise RuntimeError(
                "Ollama returned an empty response. Full response saved to "
                "outputs/ollama_last_response.json"
            )

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not run inference through local Ollama at "
            f"{OLLAMA_BASE_URL}. Make sure Ollama is running and "
            f"`ollama pull {MODEL_NAME}` has completed."
        ) from error


# ---------------------------------------------------
# RUN INFERENCE
# ---------------------------------------------------

overall_start_time = time.time()

print(f"Using Ollama model: {MODEL_NAME}")
print(f"Ollama local URL: {OLLAMA_BASE_URL}")
print(f"Image: {IMAGE_PATH}")

check_ollama_connection()

image_base64 = encode_image_base64(IMAGE_PATH)

inference_start_time = time.time()

print("Running Ollama inference...")

output_text = call_ollama_chat(
    SYSTEM_PROMPT,
    image_base64
)

inference_end_time = time.time()

inference_time = inference_end_time - inference_start_time

# ---------------------------------------------------
# CLEAN OUTPUT
# ---------------------------------------------------

print("\nRAW OUTPUT:\n")
print(output_text)

os.makedirs("outputs", exist_ok=True)

try:
    start = output_text.find("{")
    end = output_text.rfind("}") + 1

    if start == -1 or end <= start:
        raise ValueError("No JSON found in model output")

    json_text = output_text[start:end]

    parsed = json.loads(json_text)

    print("\nPARSED JSON:\n")
    print(json.dumps(parsed, indent=4))

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
