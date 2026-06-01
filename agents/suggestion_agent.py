from models.inference import run_vlm_inference


def suggestion_agent(image, issue_description):

    prompt = f"""
Issue:
{issue_description}

Suggest 3 practical improvements.

Keep suggestions short.
"""

    result = run_vlm_inference(image, prompt)

    return {
        "suggestions": result.strip()
    }