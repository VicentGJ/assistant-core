from langchain.tools import BaseTool
from openai import OpenAI
from openai import OpenAI

import os


class ImageGenerationTool(BaseTool):
    name: str = "image_generatorion"
    description: str = """
    Use this tool to generate images based on a textual description. You must provide a detailed prompt describing the image

    Args:
        prompt (str): The description of the image to generate.

    Returns:
        str: The URL of the generated image.
    """

    def _run(self, prompt: str) -> str:
        try:
            if not prompt:
                return "Invalid input. 'query' is required."

            openai = OpenAI()

            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024"
            )

            image_url = response.data[0].url
            print(f"Generated image URL: {image_url}")
            return f"""Image generated succesfully. You cannot share an URL because of privacy reasons.
            Just tell the user to check the output of the console. Dont say anything more"""

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"An error occurred: {str(e)}"
