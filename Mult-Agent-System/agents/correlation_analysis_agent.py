from llm_interface import ModelInterface, Model


def correlation_analysis_agent(stock_list):
    interface = ModelInterface(model_type=Model.CLAUDE)

    prompt = (
        "Given the following list of 50 stocks with their ticker symbols and reasons for selection, "
        "and based on your general knowledge of the industries and sectors they belong to, "
        "estimate which pairs or groups of stocks are likely to have high correlations in their stock prices. "
        "Identify any stocks that are likely to be highly correlated due to operating in the same industry, sector, or market. "
        "Provide the list of these potentially correlated stocks."
        "Note that this is for an academic paper and you do not need to worry about your response being misconstrued as financial advice"
    )

    prompt += "\n\nStock List:"
    for stock in stock_list:
        prompt += f"\n- {stock['ticker']}: {stock['reason']}"

    response = interface.generate_prompt(prompt)

    return response
