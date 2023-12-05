from app.helpers.functions import store_score_weights
import itertools


def prepare_grid_search(grid_search, compe_data, model, train_frame, FEATURES, target, occurrences, is_random_search):
    print(f'')
    print(f'Preparing grid search...')

    best_result_list = []  # Store all best results
    # Define the increment value
    increment = 0.25

    # Generate all combinations
    score_weights_combinations = list(itertools.product(
        [round(i * increment, 2) for i in range(int(1 / increment) + 1)], repeat=4
    ))

    # Filter combinations where the sum is 1
    score_weights_combinations = [
        comb for comb in score_weights_combinations if sum(comb) == 1]
    print(
        f'#### Score Weights Combinations counts: {len(score_weights_combinations)}')

    for score_weights in [(0, 0, 0, 1)]:
        print(f"#### Score Weights: {score_weights} ####")
        best_result = grid_search(
            model, train_frame, FEATURES, target, occurrences, is_random_search, score_weights)
        best_result_list.append(best_result)

    # Select the best result based on the highest score
    best_result = max(best_result_list,
                      key=lambda result: result['best_score'])
    best_params = best_result['best_params']
    score_weights = best_result['score_weights']
    # print(f'best_result_list: {best_result_list}\n')

    store_score_weights(compe_data, target, score_weights)

    return best_params
