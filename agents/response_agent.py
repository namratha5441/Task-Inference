def response_agent(scene, assessment, suggestions=None):

    if assessment["status"] == "good":

        response = (
            f"{scene}\n\n"
            "Everything looks good."
        )

        return response

    else:

        response = (
            f"{scene}\n\n"
            f"Issue detected: {assessment['issue_description']}\n\n"
            "Would you like suggestions to improve the arrangement and cleanliness?"
        )

        return response