import sys
import argparse
from train import train
from train_chained import train_chained
from combine_important_features import combine_important_features
from predict import predict
from metrics import metrics
from realtime_metrics import realtime_metrics
from app.auth.get_user_token import get_user_token
from configs.settings import EMAIL, PASSWORD
from configs.logger import Logger

def main():
    # Get the user token
    user_token = get_user_token(EMAIL, PASSWORD)

    if not user_token:
        Logger.error("Failed to obtain user token. Exiting.")
        return

    Logger.info("User token obtained successfully.")
    print('______ PREDICTIONS APP START ______\n')

    parser = argparse.ArgumentParser(description='Predictions App')
    parser.add_argument('task', choices=['train', 'train_chained', 'predict', 'metrics', 'realtime_metrics', 'combine_important_features'],
                        help='Task to perform')

    args, extra_args = parser.parse_known_args()

    if args.task == 'train':
        print('Task: Train\n')
        train(user_token, extra_args)

    elif args.task == 'train_chained':
        print('Task: Train chained\n')
        train_chained(user_token)

    elif args.task == 'predict':
        print('Task: Predict\n')
        predict(user_token)

    elif args.task == 'metrics':
        print('Task: ModelMetrics\n')
        metrics(user_token)

    elif args.task == 'realtime_metrics':
        print('Task: Realtime ModelMetrics\n')
        realtime_metrics(user_token)

    elif args.task == 'combine_important_features':
        print('Task: Combine important features\n')
        combine_important_features(user_token)

    print('\n______ PREDICTIONS APP END ______')

if __name__ == "__main__":
    main()
