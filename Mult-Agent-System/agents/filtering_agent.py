
from llm_interface import ModelInterface, Model
import json



def filtering_agent(stock_list, correlation_data):
    interface = ModelInterface(model_type=Model.FOUR)

    function_parameters = {
        "name": "filter_stock_list",
        "description": "Refines the stock list by removing stocks to reduce overall portfolio correlation.",
        "parameters": {
            "type": "object",
            "properties": {
                "filtered_stocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string", "description": "The stock ticker symbol."},
                            "reason": {"type": "string", "description": "Brief reason for selection."}
                        },
                        "required": ["ticker", "reason"]
                    }
                }
            },
            "required": ["filtered_stocks"]
        }
    }

    messages = [
        {"role": "system", "content": "You are the Filtering Agent."},
        {"role": "user", "content": "Using the correlation data provided, refine the stock list to reduce overall portfolio correlation. For each highly correlated pair or group, remove the stock that is less aligned with our interestingness criteria or offers less diversification. Provide the updated list of stocks."},
        {"role": "user", "content": f"Stock List: {stock_list}"},
        {"role": "user", "content": f"Correlation Data: {correlation_data}"}
    ]

    response = interface.request_openai_functions(messages, [function_parameters])

    arguments = response['function_call']['arguments']

    filtered_stock_list = json.loads(arguments)
    return filtered_stock_list
