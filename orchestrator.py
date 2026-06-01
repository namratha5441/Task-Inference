import os
import json
from datetime import datetime

from PIL import Image

from agents.scene_agent import scene_agent
from agents.assessment_agent import assessment_agent
from agents.suggestion_agent import suggestion_agent
from agents.response_agent import response_agent


IMAGE_PATH = "images/test1.jpg"

image = Image.open(IMAGE_PATH).convert("RGB")

image.thumbnail((768, 768))

# =====================================================
# STEP 1 — SCENE UNDERSTANDING
# =====================================================

scene_result = scene_agent(image)

print("\nSCENE RESULT:\n")
print(scene_result)

# =====================================================
# STEP 2 — ASSESSMENT
# =====================================================

assessment_result = assessment_agent(
    image,
    scene_result["scene_description"]
)

print("\nASSESSMENT RESULT:\n")
print(assessment_result)

# =====================================================
# STEP 3 — OPTIONAL SUGGESTIONS
# =====================================================

suggestions_result = None

if assessment_result["status"] == "issue_detected":

    suggestions_result = suggestion_agent(
        image,
        assessment_result["issue_description"]
    )

    print("\nSUGGESTIONS:\n")
    print(suggestions_result)

# =====================================================
# STEP 4 — FINAL RESPONSE
# =====================================================

final_response = response_agent(
    scene_result["scene_description"],
    assessment_result,
    suggestions_result
)

print("\nFINAL RESPONSE:\n")
print(final_response)

# =====================================================
# STEP 5 — SAVE OUTPUT
# =====================================================

os.makedirs("outputs", exist_ok=True)

output = {
    "scene": scene_result,
    "assessment": assessment_result,
    "suggestions": suggestions_result,
    "final_response": final_response
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_path = f"outputs/run_{timestamp}.json"

with open(output_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"\nSaved to: {output_path}")