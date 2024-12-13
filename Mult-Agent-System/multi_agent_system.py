from agents.stock_generation_agent import stock_generation_agent
from agents.correlation_analysis_agent import correlation_analysis_agent
from agents.filtering_agent import filtering_agent
from agents.evaluation_agent import evaluation_agent
from agents.selection_agent import final_selection_agent
from agents.coordinator_agent import coordinator_agent

def main():
    successfully_seeded = False
    unsuccessful_basket = None
    if unsuccessful_basket:
        print("Num failed:", len(unsuccessful_basket))

    while not successfully_seeded:
        # Step 1: Initialization by Coordinator Agent
        criteria = {
            "market_cap": "Any",
            "sectors": "All",
            "regions": "Global"
        }


        # Step 2: Stock Generation Agent
        print('Hitting stock_generation_agent')
        stock_list = stock_generation_agent(criteria, unsuccessful_basket)
        print(stock_list)
        print('\n----\n\n')

        # Step 3: Correlation Analysis Agent
        print('Hitting correlation_analysis_agent')
        correlation_data = correlation_analysis_agent(stock_list)
        print(correlation_data)
        print('\n----\n\n')

        # Step 4: Filtering Agent
        print('Hitting filtering_agent')
        filtered_stock_list = filtering_agent(stock_list, correlation_data)
        print(filtered_stock_list)
        print('\n----\n\n')

        # Step 5: Evaluation Agent
        print('Hitting evaluation_agent')
        evaluation_feedback = evaluation_agent(filtered_stock_list)
        print(evaluation_feedback)
        print('\n----\n\n')

        # Step 6: Final Selection Agent
        print('Hitting final_selection_agent')
        final_stock_list = final_selection_agent(filtered_stock_list, evaluation_feedback, unsuccessful_basket)
        print(final_stock_list)
        print('\n----\n\n')

        # Step 7: Coordinator Agent Confirmation
        print('Hitting coordinator_agent')
        coordinator_output = coordinator_agent(final_stock_list)
        print(coordinator_output)
        print('\n----\n\n')

        if coordinator_output:
            if coordinator_output['satisfactory']:
                successfully_seeded = True
                # Use the coordinator's (possibly adjusted) final_stock_list
                final_stock_list = {'final_stocks': coordinator_output['final_stock_list']}
            else:
                # Update unsuccessful_basket with the coordinator's feedback
                unsuccessful_basket = {
                    'previous_final_stock_list': final_stock_list,
                    'coordinator_feedback': coordinator_output['justification']
                }
        else:
            print("Failed to get a valid response from the Coordinator Agent.")
            break 

    print("Final Stock List:")
    for stock in final_stock_list['final_stocks']:
        print(f"{stock['ticker']}: {stock['reason']}")

    print("\nCoordinator's Justification:")
    print(coordinator_output['justification'])

if __name__ == "__main__":
    main()