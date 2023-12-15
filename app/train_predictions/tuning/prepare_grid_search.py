from app.helpers.functions import store_score_weights, get_score_weights
import itertools


def prepare_grid_search(grid_search, compe_data, model, train_frame, FEATURES, target, occurrences, is_random_search, run_score_weights):
    print(f'')
    print(f'Preparing grid search...')

    if run_score_weights:
        print(f'Running score weights...')
        best_result_list = []  # Store all best results
        # Define the increment value
        increment = 1.0

        # Generate all combinations
        score_weights_combinations = list(itertools.product(
            [round(i * increment, 2) for i in range(int(1 / increment) + 1)], repeat=2
        ))

        score_weights_combinations = [(0, 1)]
        # Filter combinations where the sum is 1
        score_weights_combinations = [
            comb for comb in score_weights_combinations if sum(comb) == 1]
        print(
            f'#### Score Weights Combinations counts: {len(score_weights_combinations)}')

        # accuracy, precision, f1, recall
        # for score_weights in [(0, 1)]:
        for score_weights in score_weights_combinations:
            print(f"#### Score Weights: {score_weights} ####")
            score_weights = {
                'accuracy': 0,
                'precision': 0,
                'f1': 0,
                'recall': score_weights[1],
            }
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

    else:
        # get saved weights for current competition
        score_weights = get_score_weights(compe_data, target)
        if score_weights == None:
            score_weights = {
                'accuracy': 0,
                'precision': 0,
                'f1': 0,
                'recall': 1,
            }

        best_result = grid_search(
            model, train_frame, FEATURES, target, occurrences, is_random_search, score_weights)
        best_params = best_result['best_params']

    return best_params
