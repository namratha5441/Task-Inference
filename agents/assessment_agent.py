import json
import re

from models.inference import run_vlm_inference


def assessment_agent(image, scene_description):

    prompt = f"""
        Scene Description:
        {scene_description}
        
        Analyze the image carefully.
        
        Determine whether:
        - everything looks good
        OR
        - issues are present
        
        Possible issues:
        - clutter
        - dirt
        - misplaced items
        - unsafe arrangement
        
        Return JSON:
        
        {{
            "status": "good OR issue_detected",
            "issue_description": "short description"
        }}
        """

    result = run_vlm_inference(image, prompt)

    try:
        json_match = re.search(r'\{.*\}', result, re.DOTALL)

        parsed = json.loads(json_match.group(0))

        return parsed

    except:

        return {
            "status": "unknown",
            "issue_description": result
        }