from models.inference import run_vlm_inference


def scene_agent(image):

    prompt = """
Analyze this image.

Describe the scene in 2 concise lines.
Focus on:
- environment
- organization
- cleanliness

Keep the response short.
"""

    result = run_vlm_inference(image, prompt)

    return {
        "scene_description": result.strip()
    }