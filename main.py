from train import train
from train_chained import train_chained
from combine_important_features import combine_important_features
from predict import predict
from metrics import metrics
from realtime_metrics import realtime_metrics
from app.auth.get_user_token import get_user_token
from configs.settings import EMAIL, PASSWORD
from configs.logger import Logger
import sys


def main():

    # Get the user token
    user_token = get_user_token(EMAIL, PASSWORD)

    if user_token:
        Logger.info("User token obtained successfully.")

        print('______ PREDICTIONS APP START ______\n')

        task = None
        for arg in sys.argv:
            if arg.startswith('task'):
                parts = arg.split('=')
                if len(parts) == 2:
                    task = parts[1]

        if task == None:
            print('Please choose a task to continue\n')
            return

        if task == 'train':
            print('Task: Train\n')
            train(user_token)

        elif task == 'train_chained':
            print('Task: Train chained\n')
            train_chained(user_token)

        elif task == 'predict':
            print('Task: Predict\n')
            predict(user_token)

        elif task == 'metrics':
            print('Task: ModelMetrics\n')
            metrics(user_token)

        elif task == 'realtime_metrics':
            print('Task: Realtime ModelMetrics\n')
            realtime_metrics(user_token)

        elif task == 'combine_important_features':
            print('Task: Combine important features\n')
            combine_important_features(user_token)

        else:
            print('Invalid task\n')

        print('\n______ PREDICTIONS APP END ______')


if __name__ == "__main__":
    main()
