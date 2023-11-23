from train import train
from predict import predict
from app.auth.get_user_token import get_user_token
from configs.settings import EMAIL, PASSWORD
from configs.logger import Logger


def main():

    # Get the user token
    user_token = get_user_token(EMAIL, PASSWORD)

    if user_token:
        Logger.info("User token obtained successfully.")

        print('______ PREDICTIONS APP START ______\n')

        train(user_token)

        # predict(user_token)

        print('\n______ PREDICTIONS APP END ______')


if __name__ == "__main__":
    main()
