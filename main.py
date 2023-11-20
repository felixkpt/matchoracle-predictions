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

        # class_weight = []
        # counts = 10
        # curr_val = 1
        # increment = 0.05

        # for i in range(counts):  # Assuming you have 6 classes
        #     inner_list = []
        #     for j in range(counts):
        #         class_weight_dict = {k: curr_val if k <= j else round(curr_val - increment, 3) for k in range(counts)}
        #         inner_list.append(class_weight_dict)
            
        #     class_weight.append(inner_list)

        #     curr_val += increment
        #     curr_val = round(curr_val, 3) if curr_val <= 2 else 1

        # print(class_weight)
        # return

        train(user_token)

        # predict(user_token)

        print('\n______ PREDICTIONS APP END ______')


if __name__ == "__main__":
    main()
