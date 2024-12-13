# llm_interface.py
from dotenv import load_dotenv
import os
import openai
import requests
from typing import Any, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import json

load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY")
anthropic_key = os.getenv("ANTHROPIC_KEY")

class Model(Enum):
    O1 = "o1"
    FOUR = "4"
    FOUR_O = "gpt-4o"
    CLAUDE = "claude-3-5-sonnet-20241022" 
    MINI = "gpt-4o-mini"

@dataclass
class StockRecommendation:
    ticker: str
    description: str
    potential_upside: float
    sector: str

class ModelInterface:
    def __init__(self, model_type: Model):
        self.model_type = model_type

    def generate_prompt(self, prompt: str = None, messages: list = None, params: Optional[Dict[str, Any]] = None) -> Any:
        """Sends a prompt or messages to the appropriate model API."""
        if params is None:
            params = {}

        if self.model_type in {Model.O1, Model.FOUR, Model.FOUR_O}:
            return self._request_openai(prompt=prompt, messages=messages, params=params)
        elif self.model_type == Model.CLAUDE:
            return self._request_anthropic(prompt, params)
        else:
            raise ValueError("Unsupported model type")

    def _request_openai(self, prompt: str = None, messages: list = None, params: Optional[Dict[str, Any]] = None) -> Any:
        """Handles requests to OpenAI's API."""
        if messages is None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            **params
        )

        return response.choices[0].message

    def parse_anthropic_with_openai(self, anthropic_output):
        """
        Takes unstructured stock data from Anthropic and uses OpenAI functions to parse it
        into a structured JSON format.
        """
        # Define function parameters for structured output
        function_parameters = [{
            "name": "cool_stock_info",
            "description": "Get information on 20 interesting stocks for academic research. Each stock should differ significantly, with minimal correlation among them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker": {
                                    "type": "string",
                                    "description": "The stock ticker symbol.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "A brief description of why this stock is interesting for academic research.",
                                },
                            },
                            "required": ["ticker", "description"]
                        },
                        "minItems": 20,
                        "maxItems": 20
                    },
                    "justification": {
                        "type": "string",
                        "description": "A justification of how the stocks are uncorrelated and interesting in the context of each other."
                    }
                },
                "required": ["stocks", "justification"],
                "additionalProperties": False
            }
        }]

        messages = [
            {"role": "system", "content": "You are an assistant tasked with converting unstructured stock data into a structured JSON format."},
            {"role": "user", "content": f"Please parse the following text into a JSON object with 20 'stocks' entries, each having a 'ticker' and 'description', and a 'justification' field explaining the lack of correlation.\n\n{anthropic_output}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            functions=function_parameters,
            function_call="auto"
        )

        return response.choices[0].message

    def request_openai_functions(self, messages: list, functions: list, params: Optional[Dict[str, Any]] = None, model: str = "gpt-4o-mini") -> Any:
        """Handles OpenAI function calls with custom messages and functions."""
        if params is None:
            params = {}
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call="auto",
            **params
        )
        return response.choices[0].message


    def request_anthropic_unstructured(self, function_description):
        """
        Sends a request to Anthropic Claude for information without enforcing structured output.

        Args:
            function_description (dict): A dictionary containing the description of the function and any relevant parameters.

        Returns:
            str: An unstructured response from Claude.
        """
        prompt = f"Please provide information on the following topic:\n\n{function_description['description']}"

        params = {
            "temperature": 0.7,
            "max_tokens": 4000
        }

        try:
            response = self._request_anthropic(prompt, params)

            return response

        except Exception as e:
            raise RuntimeError(f"Error in Anthropic function call: {str(e)}")

    def _request_anthropic(self, prompt: str, params: Dict[str, Any]) -> str:
        """Handles requests to Anthropic's API using the Messages format."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        payload = {
            "model": self.model_type.value,
            "max_tokens": params.get("max_tokens", 1024),
            "temperature": params.get("temperature", 0.7),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # Add system message if provided
        if params.get("system_prompt"):
            payload["system"] = params["system_prompt"]

        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Anthropic API Error: {response.status_code} - {response.text}")
        
        # Extract the content from the response
        content = response.json().get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "").strip()
        
        return ""