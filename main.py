from train import train
from predict import predict
from app.auth.get_user_token import get_user_token
from configs.settings import EMAIL, PASSWORD
from configs.logger import Logger
from app.model_metrics import ModelMetrics
import os


def main():

    # Get the user token
    user_token = get_user_token(EMAIL, PASSWORD)

    if user_token:
        Logger.info("User token obtained successfully.")

        print('______ PREDICTIONS APP START ______\n')

        train(user_token)

        # predict(user_token)

        target = 'hda_target'
        target = 'bts_target'
        # target = 'over25_target'
        # target = 'cs_target'

        prediction_types = [
            'regular_prediction_last_5_matches_optimized',
            'regular_prediction_last_7_matches_optimized',
            'regular_prediction_last_10_matches_optimized',
            'regular_prediction_last_15_matches_optimized',
            'regular_prediction_last_20_matches_optimized',
            'regular_prediction_last_25_matches_optimized',
        ]

        model_metrics = ModelMetrics(target, prediction_types)
        comparison_results = model_metrics.compare_models()
        model_metrics.choose_best_model(comparison_results)

        print('\n______ PREDICTIONS APP END ______')


if __name__ == "__main__":
    main()
