from llm_interface import ModelInterface, Model
import json


def coordinator_agent(final_stock_list):
    interface = ModelInterface(model_type=Model.FOUR)

    function_parameters = [{
        "name": "review_final_selection",
        "description": (
            "Reviews the finalized list of 20 stocks for compliance with initial criteria of interestingness and low correlation. "
            "Confirms the selection or suggests final adjustments."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "satisfactory": {
                    "type": "boolean",
                    "description": "Indicates whether the final stock list is satisfactory."
                },
                "final_stock_list": {
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
                                "description": "Brief justification for inclusion in the final list."
                            }
                        },
                        "required": ["ticker", "reason"]
                    },
                    "description": "The finalized list of stocks."
                },
                "justification": {
                    "type": "string",
                    "description": "Explanation of the decision to confirm or adjust the stock list."
                }
            },
            "required": ["satisfactory", "final_stock_list", "justification"],
            "additionalProperties": False
        }
    }]

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Coordinator Agent responsible for reviewing the finalized stock list. "
                "Ensure that the list meets the initial criteria of interestingness and low correlation. "
                "Provide a decision on whether the list is satisfactory. If not, suggest final adjustments."
            )
        },
        {
            "role": "user",
            "content": "Please review the following finalized stock list and provide your decision."
        },
        {
            "role": "user",
            "content": f"Final Stock List: {final_stock_list['final_stocks']}"
        }
    ]

    params = {"temperature": 0.7, "max_tokens": 1500}
    response = interface.request_openai_functions(messages, function_parameters, params, "gpt-4")
    print(response)
    if 'function_call' in response:
        arguments = response['function_call']['arguments']

        try:
            coordinator_output = json.loads(arguments)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print("Raw arguments:", arguments)
            coordinator_output = None
    else:
        print("Assistant did not provide a function call.")
        coordinator_output = None

    return coordinator_output
