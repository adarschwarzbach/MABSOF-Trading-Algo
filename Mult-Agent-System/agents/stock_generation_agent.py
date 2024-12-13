# agents/stock_generation_agent.py
from llm_interface import ModelInterface, Model
import json
import pandas as pd
import os
import random

print(f"Current working directory: {os.getcwd()}")


def load_and_parse_ticker_data():
    """
    Loads ticker data from the JSON file located at a specific path and filters it for US-based exchanges.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, '../data/ticker_data.json')

        with open(file_path, 'r') as file:
            raw_data = json.load(file)

        if isinstance(raw_data, dict):  
            raw_data = [raw_data]

        df = pd.DataFrame(raw_data)

        required_columns = ['symbol', 'name', 'price', 'exchange', 'exchangeShortName', 'type']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Missing required columns in data. Expected columns: {required_columns}")

        us_exchanges = ['NYSE']
        filtered_data = df[df['exchangeShortName'].isin(us_exchanges)].to_dict(orient='records')
        return filtered_data
    except Exception as e:
        print(f"Error loading or parsing ticker data: {e}")
        return None


def divide_and_conquer_ticker_data(parsed_ticker_data, batch_size):
    """
    Divides the parsed ticker data into smaller batches of the specified size.
    """
    return [parsed_ticker_data[i:i + batch_size] for i in range(0, len(parsed_ticker_data), batch_size)]


def stock_generation_agent(criteria, unsuccessful_basket=None):
    """
    Generates a list of stocks for academic research purposes using parsed ticker data.
    Processes the data in batches using divide and conquer to handle large datasets.
    """
    parsed_ticker_data = load_and_parse_ticker_data()
    if not parsed_ticker_data:
        print("Failed to load and parse ticker data. Exiting.")
        return None

    batch_size = 1500  # Adjust based on the expected token size per entry
    ticker_batches = divide_and_conquer_ticker_data(parsed_ticker_data, batch_size)

    interface = ModelInterface(model_type=Model.FOUR)
    aggregated_results = []

    for batch_idx, ticker_batch in enumerate(ticker_batches):
        print(f"Processing batch {batch_idx + 1}/{len(ticker_batches)}...")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Stock Generation Agent for an academic research project. "
                    "Provide factual information about publicly traded companies without offering financial advice. "
                    "Only use the provided ticker data for your analysis. Ignore any external knowledge, even if you recognize a ticker. "
                    "The provided data does not represent real-world performance and must be treated as isolated."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Generate a list of up to 10 publicly traded companies from the following batch. "
                    "Each company must include the ticker symbol and a brief factual description of the company."
                )
            },
            {
                "role": "user",
                "content": f"Ticker Batch: {ticker_batch}"
            }
        ]

        if unsuccessful_basket:
            print("num_failed", len(unsuccessful_basket) )
            for unsuccessful_item in unsuccessful_basket:
                if 'justification' in unsuccessful_item:
                    feedback_content = (
                        "Previous coordinator feedback:\n"
                        f"{unsuccessful_item['justification']}\n"
                        "Please ensure this feedback is incorporated into your selection."
                    )
                    messages.append({"role": "user", "content": feedback_content})

                if 'final_stock_list' in unsuccessful_item:
                    previous_stock_list_content = "Previous Stock List (to avoid similar selections):\n"
                    for stock in unsuccessful_item['final_stock_list']:
                        previous_stock_list_content += f"- {stock['ticker']}: {stock['reason']}\n"
                    messages.append({"role": "user", "content": previous_stock_list_content})

        function_parameters = [{
            "name": "generate_stock_list",
            "description": (
                "Generates a list of up to 10 publicly traded companies for academic research purposes. "
                "Avoid providing financial advice or subjective opinions."
            ),
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
                                    "description": "The stock ticker symbol."
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Brief factual description of the company."
                                }
                            },
                            "required": ["ticker", "reason"]
                        },
                        "maxItems": 10  # Limit per batch
                    }
                },
                "required": ["stocks"],
                "additionalProperties": False
            }
        }]

        params = {"temperature": 0.7, "max_tokens": 2000}
        response = interface.request_openai_functions(messages, function_parameters, params)

        arguments = response.get('function_call', {}).get('arguments', '{}')
        try:
            stock_list = json.loads(arguments)
            if 'stocks' in stock_list:
                aggregated_results.extend(stock_list['stocks'])
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in batch {batch_idx + 1}: {e}")
            print("Raw arguments:", arguments)

    unique_stocks = {stock['ticker']: stock for stock in aggregated_results}
    final_stock_list = list(unique_stocks.values())

    print("Final Stock List:", final_stock_list)

    return final_stock_list[:50]  
