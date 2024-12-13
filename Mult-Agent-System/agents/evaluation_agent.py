from llm_interface import ModelInterface, Model
import json



def evaluation_agent(filtered_stock_list):
    interface = ModelInterface(model_type=Model.CLAUDE)

    prompt = "Evaluate the following list of stocks for diversification across sectors, geographical regions, and market capitalizations. Identify any overconcentration in specific areas and recommend substitutions to improve diversification while maintaining portfolio quality."

    prompt += "\n\nStock List:"
    for stock in filtered_stock_list['filtered_stocks']:
        prompt += f"\n- {stock['ticker']}: {stock['reason']}"

    response = interface.generate_prompt(prompt)

    return response
