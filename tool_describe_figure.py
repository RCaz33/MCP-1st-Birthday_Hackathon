
import os
from openai import OpenAI
# The OpenAI library handles the API key and base URL automatically 
# after instantiation.

def thorough_picture_description(figure: str) -> str:
    """
    Generates a thorough description for a given image URL using 
    the Nebius Token Factory endpoint.

    Args:
        figure: The URL of the image to describe.

    Returns:
        The generated text description of the image.
    """

    try:
        client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY")
        )
    except Exception as e:

        return f"Error initializing OpenAI client: {e}"


    messages_payload = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Provide a very detailed, thorough, and descriptive analysis of this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": figure},
                },
            ],
        }
    ]


    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash", 
            messages=messages_payload,
            max_tokens=2048
        )
        

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            return "Could not retrieve a description from the API."

    except Exception as e:
        return f"An error occurred during the API call: {e}"
