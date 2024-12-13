# agents/selection_agent.py
from llm_interface import ModelInterface, Model
import json

def final_selection_agent(filtered_stock_list, evaluation_feedback, unsuccessful_basket=None):
    interface = ModelInterface(model_type=Model.FOUR)

    function_parameters = [{
        "name": "finalize_stock_list",
        "description": "Finalizes the basket of 20 stocks, ensuring they meet the criteria of being interesting and uncorrelated.",
        "parameters": {
            "type": "object",
            "properties": {
                "final_stocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol."
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief reason for selection."
                            }
                        },
                        "required": ["ticker", "reason"]
                    },
                    "minItems": 20,
                    "maxItems": 20
                }
            },
            "required": ["final_stocks"],
            "additionalProperties": False
        }
    }]

    messages = [
        {
            "role": "system",
            "content": "You are the Final Selection Agent."
        },
        {
            "role": "user",
            "content": (
                "Based on the evaluation feedback, finalize a basket of 20 stocks that are interesting investments and exhibit low correlations. "
                "Ensure the portfolio is diversified across sectors and regions. For each selected stock, provide the ticker symbol and a brief justification."
            )
        },
        {
            "role": "user",
            "content": f"Filtered Stock List: {filtered_stock_list['filtered_stocks']}"
        },
        {
            "role": "user",
            "content": f"Evaluation Feedback: {evaluation_feedback}"
        }
    ]

    # Include coordinator feedback if available
    if unsuccessful_basket and 'coordinator_feedback' in unsuccessful_basket:
        messages.append({
            "role": "user",
            "content": (
                "Coordinator's previous feedback:\n"
                f"{unsuccessful_basket['coordinator_feedback']}\n"
                "Please consider this feedback when finalizing the stock list."
            )
        })

    params = {"temperature": 0.7, "max_tokens": 2000}
    response = interface.request_openai_functions(messages, function_parameters, params)

    arguments = response['function_call']['arguments']

    try:
        final_stock_list = json.loads(arguments)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("Raw arguments:", arguments)
        final_stock_list = None

    return final_stock_list
