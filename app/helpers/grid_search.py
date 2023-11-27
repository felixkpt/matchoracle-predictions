from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from configs.settings import COMMON_FEATURES
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

_occurrences = 22


def params(train_frame, class_weight_counts=14):
    len_train = len(train_frame)

    # Setting the range for params
    class_weight = np.linspace(1.0, 2.4, class_weight_counts)
    _min_samples_splits = np.linspace(10, len_train, 50)
    _n_estimators = np.linspace(100, len_train, 10)

    min_samples_splits = []
    for x in _min_samples_splits:
        x = int(x)
        if x > int(len_train / 4):
            break
        min_samples_splits.append(x if x < len_train else len_train)

    n_estimators = []
    for x in _n_estimators:
        x = int(x)
        if x > int(len_train / 4):
            break
        n_estimators.append(x if x < len_train else len_train)

    return [class_weight, min_samples_splits, n_estimators]


def grid_search_hda(model, train_frame, target, occurrences):
    global _occurrences
    _occurrences = 333

    class_weight, min_samples_splits, n_estimators = params(
        train_frame, 20)

    _class_weight = []
    for i, x in enumerate(class_weight):

        for j in class_weight:
            res = {0: 1, 1: round(class_weight[i], 3), 2: round(j, 3)}
            _class_weight.append(res)

    # filtering where x[1] is greater than x[2],
    # this is based on the fact that our model struggles in making 1 and 2 preds, 1 being the worst
    class_weight = []
    for x in _class_weight:
        if x[1] > 1.5 and x[1] > x[2]:
            class_weight.append(x)

    len_train = len(train_frame)
    n_estimators = int(0.15 * len_train)
    min_samples_split = int(0.02 * len_train)
    min_samples_split = min_samples_split if min_samples_split > 2 else 2

    # Creating a dictionary grid for grid search
    param_grid = {
        'min_samples_split': [min_samples_split, min_samples_split * 2],
        'class_weight': class_weight,
        'n_estimators': [100],
    }

    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        # cv=StratifiedKFold(),
        n_jobs=-1,
        scoring=lambda estimator, X, y_true: scorer_hda(
            estimator, X, y_true, occurrences),
        verbose=2
    ).fit(train_frame[FEATURES], train_frame[target])

    # Extract and print the best class weight and score
    best_class_weight = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_class_weight}")
    print(f"Best score: {best_score}")

    # Create a DataFrame to store the grid search results
    # weigh_data = pd.DataFrame({
    #     'score': gridsearch.cv_results_['mean_test_score'],
    #     'weight': [1 - x for x in class_weight]
    # })

    # Set up the plot
    # sns.set_style('whitegrid')
    # plt.figure(figsize=(12, 8))

    # # Create the line plot for class weight vs. F1 score
    # sns.lineplot(x=weigh_data['weight'], y=weigh_data['score'])

    # # Add labels and ticks to the plot
    # plt.xlabel('Weight for class 1')
    # plt.ylabel('F1 score')
    # plt.xticks([round(i / 10, 1) for i in range(0, 11, 1)])
    # plt.title('Scoring for different class weights', fontsize=24)

    # # Show the plot
    # plt.show()


def grid_search_over_under(model, train_frame, target, scoring):

    class_weight, min_samples_splits, n_estimators = params(train_frame, 12)

    _class_weight = []
    for i, x in enumerate(class_weight):

        for j in class_weight:
            res = {1: round(class_weight[i], 3), 0: round(j, 3)}
            _class_weight.append(res)

    # filtering where x[1] is greater than x[2],
    # this is based on the fact that our model struggles in making 1 and 2 preds, 1 being the worst
    class_weight = []
    for x in _class_weight:
        if x[1] != x[0] and x[1] < 1.5 and x[0] < 1.5:
            class_weight.append(x)

    # Creating a dictionary grid for grid search
    param_grid = {
        'class_weight': class_weight,
    }

    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(),
        n_jobs=-1,
        scoring=scorer_over_under,
        verbose=2
    ).fit(train_frame[FEATURES], train_frame[target])

    # Create a DataFrame to store the grid search results
    # weigh_data = pd.DataFrame({
    #     'score': gridsearch.cv_results_['mean_test_score'],
    #     'weight': [1 - x for x in class_weights]
    # })

    # Set up the plot
    # sns.set_style('whitegrid')
    # plt.figure(figsize=(12, 8))

    # # Create the line plot for class weight vs. F1 score
    # sns.lineplot(x=weigh_data['weight'], y=weigh_data['score'])

    # # Add labels and ticks to the plot
    # plt.xlabel('Weight for class 1')
    # plt.ylabel('F1 score')
    # plt.xticks([round(i / 10, 1) for i in range(0, 11, 1)])
    # plt.title('Scoring for different class weights', fontsize=24)

    # Extract and print the best class weight and score
    best_class_weight = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_class_weight}")
    print(f"Best score: {best_score}")

    # Show the plot
    plt.show()


def grid_search_gg_ng(model, train_frame, target, scoring):

    class_weight, min_samples_splits, n_estimators = params(train_frame, 12)

    _class_weight = []
    for i, x in enumerate(class_weight):

        for j in class_weight:
            res = {1: round(class_weight[i], 3), 0: round(j, 3)}
            _class_weight.append(res)

    # filtering where x[1] is greater than x[2],
    # this is based on the fact that our model struggles in making 1 and 2 preds, 1 being the worst
    class_weight = []
    for x in _class_weight:
        if x[1] != x[0] and x[1] < 1.5 and x[0] < 1.5:
            class_weight.append(x)

    # Creating a dictionary grid for grid search
    param_grid = {
        'class_weight': class_weight,
    }

    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(),
        n_jobs=-1,
        scoring=scorer_gg_ng,
        verbose=2
    ).fit(train_frame[FEATURES], train_frame[target])

    # Create a DataFrame to store the grid search results
    # weigh_data = pd.DataFrame({
    #     'score': gridsearch.cv_results_['mean_test_score'],
    #     'weight': [1 - x for x in class_weights]
    # })

    # Set up the plot
    # sns.set_style('whitegrid')
    # plt.figure(figsize=(12, 8))

    # # Create the line plot for class weight vs. F1 score
    # sns.lineplot(x=weigh_data['weight'], y=weigh_data['score'])

    # # Add labels and ticks to the plot
    # plt.xlabel('Weight for class 1')
    # plt.ylabel('F1 score')
    # plt.xticks([round(i / 10, 1) for i in range(0, 11, 1)])
    # plt.title('Scoring for different class weights', fontsize=24)

    # Extract and print the best class weight and score
    best_class_weight = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_class_weight}")
    print(f"Best score: {best_score}")

    # Show the plot
    plt.show()


def grid_search_cs(model, train_frame, target, scoring):

    class_weight, min_samples_splits, n_estimators = params(
        train_frame)

    class_weight = np.linspace(1.0, 2.00, 5)

    arr = [x for x in class_weight]

    class_weight = []

    class_weight = []

    curr_val = 1
    increment = 0.1

    for i in range(6):  # Assuming you have 6 classes
        inner_list = []
        for j in range(6):
            class_weight_dict = {k: curr_val if k <=
                                 j else curr_val - increment for k in range(6)}
            inner_list.append(class_weight_dict)

        class_weight.append(inner_list)

        curr_val += increment
        curr_val = round(curr_val, 1) if curr_val <= 2 else 1

    # old code
    curr_val = 1
    increment = 1 / 60
    for i in range(0, 121):
        new_val = curr_val + increment

        obj = {
            0: new_val if i == 0 else curr_val,
            1: new_val if i == 1 else curr_val,
            2: new_val if i == 2 else curr_val,
            3: new_val if i == 3 else curr_val,
            4: new_val if i == 4 else curr_val,
            5: new_val if i == 5 else curr_val,
            6: new_val if i == 6 else curr_val,
            7: new_val if i == 7 else curr_val,
            8: new_val if i == 8 else curr_val,
            9: new_val if i == 9 else curr_val,
            10: new_val if i == 10 else curr_val,
            11: new_val if i == 11 else curr_val,
            12: new_val if i == 12 else curr_val,
            13: new_val if i == 13 else curr_val,
            14: new_val if i == 14 else curr_val,
            15: new_val if i == 15 else curr_val,
            16: new_val if i == 16 else curr_val,
            17: new_val if i == 17 else curr_val,
            18: new_val if i == 18 else curr_val,
            19: new_val if i == 19 else curr_val,
            20: new_val if i == 20 else curr_val,
            21: new_val if i == 21 else curr_val,
            22: new_val if i == 22 else curr_val,
            23: new_val if i == 23 else curr_val,
            24: new_val if i == 24 else curr_val,
            25: new_val if i == 25 else curr_val,
            26: new_val if i == 26 else curr_val,
            27: new_val if i == 27 else curr_val,
            28: new_val if i == 28 else curr_val,
            29: new_val if i == 29 else curr_val,
            30: new_val if i == 30 else curr_val,
            31: new_val if i == 31 else curr_val,
            32: new_val if i == 32 else curr_val,
            33: new_val if i == 33 else curr_val,
            34: new_val if i == 34 else curr_val,
            35: new_val if i == 35 else curr_val,
            36: new_val if i == 36 else curr_val,
            37: new_val if i == 37 else curr_val,
            37: new_val if i == 38 else curr_val,
            39: new_val if i == 39 else curr_val,
            40: new_val if i == 40 else curr_val,
            41: new_val if i == 41 else curr_val,
            42: new_val if i == 42 else curr_val,
            43: new_val if i == 43 else curr_val,
            44: new_val if i == 44 else curr_val,
            45: new_val if i == 45 else curr_val,
            46: new_val if i == 46 else curr_val,
            47: new_val if i == 47 else curr_val,
            48: new_val if i == 48 else curr_val,
            49: new_val if i == 49 else curr_val,
            50: new_val if i == 50 else curr_val,
            51: new_val if i == 51 else curr_val,
            52: new_val if i == 52 else curr_val,
            53: new_val if i == 53 else curr_val,
            54: new_val if i == 54 else curr_val,
            55: new_val if i == 55 else curr_val,
            56: new_val if i == 56 else curr_val,
            57: new_val if i == 57 else curr_val,
            58: new_val if i == 58 else curr_val,
            59: new_val if i == 59 else curr_val,
            60: new_val if i == 60 else curr_val,
            61: new_val if i == 61 else curr_val,
            62: new_val if i == 62 else curr_val,
            63: new_val if i == 63 else curr_val,
            64: new_val if i == 64 else curr_val,
            65: new_val if i == 65 else curr_val,
            66: new_val if i == 66 else curr_val,
            67: new_val if i == 67 else curr_val,
            68: new_val if i == 68 else curr_val,
            69: new_val if i == 69 else curr_val,
            70: new_val if i == 70 else curr_val,
            71: new_val if i == 71 else curr_val,
            72: new_val if i == 72 else curr_val,
            73: new_val if i == 73 else curr_val,
            74: new_val if i == 74 else curr_val,
            75: new_val if i == 75 else curr_val,
            76: new_val if i == 76 else curr_val,
            77: new_val if i == 77 else curr_val,
            78: new_val if i == 78 else curr_val,
            79: new_val if i == 79 else curr_val,
            80: new_val if i == 80 else curr_val,
            81: new_val if i == 81 else curr_val,
            82: new_val if i == 82 else curr_val,
            83: new_val if i == 83 else curr_val,
            84: new_val if i == 84 else curr_val,
            85: new_val if i == 85 else curr_val,
            86: new_val if i == 86 else curr_val,
            87: new_val if i == 87 else curr_val,
            88: new_val if i == 88 else curr_val,
            89: new_val if i == 89 else curr_val,
            90: new_val if i == 90 else curr_val,
            91: new_val if i == 91 else curr_val,
            92: new_val if i == 92 else curr_val,
            93: new_val if i == 93 else curr_val,
            94: new_val if i == 94 else curr_val,
            95: new_val if i == 95 else curr_val,
            96: new_val if i == 96 else curr_val,
            97: new_val if i == 97 else curr_val,
            98: new_val if i == 98 else curr_val,
            99: new_val if i == 99 else curr_val,
            100: new_val if i == 100 else curr_val,
            101: new_val if i == 101 else curr_val,
            102: new_val if i == 102 else curr_val,
            103: new_val if i == 103 else curr_val,
            104: new_val if i == 104 else curr_val,
            105: new_val if i == 105 else curr_val,
            106: new_val if i == 106 else curr_val,
            107: new_val if i == 107 else curr_val,
            108: new_val if i == 108 else curr_val,
            108: new_val if i == 109 else curr_val,
            110: new_val if i == 110 else curr_val,
            111: new_val if i == 111 else curr_val,
            112: new_val if i == 112 else curr_val,
            113: new_val if i == 113 else curr_val,
            114: new_val if i == 114 else curr_val,
            115: new_val if i == 115 else curr_val,
            116: new_val if i == 116 else curr_val,
            116: new_val if i == 117 else curr_val,
            118: new_val if i == 118 else curr_val,
            119: new_val if i == 119 else curr_val,
            120: new_val if i == 120 else curr_val,
        }

        curr_val += increment
        curr_val = curr_val if curr_val <= 2 else 1

        class_weight.append(obj)

    # Creating a dictionary grid for grid search
    param_grid = {
        'class_weight': class_weight,
        'min_samples_split': min_samples_splits,
        'n_estimators': n_estimators,
    }

    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=-1,
        scoring=scorer_cs,
        verbose=2
    ).fit(train_frame[FEATURES], train_frame[target])

    # Create a DataFrame to store the grid search results
    # weigh_data = pd.DataFrame({
    #     'score': gridsearch.cv_results_['mean_test_score'],
    #     'weight': [1 - x for x in class_weights]
    # })

    # Set up the plot
    # sns.set_style('whitegrid')
    # plt.figure(figsize=(12, 8))

    # # Create the line plot for class weight vs. F1 score
    # sns.lineplot(x=weigh_data['weight'], y=weigh_data['score'])

    # # Add labels and ticks to the plot
    # plt.xlabel('Weight for class 1')
    # plt.ylabel('F1 score')
    # plt.xticks([round(i / 10, 1) for i in range(0, 11, 1)])
    # plt.title('Scoring for different class weights', fontsize=24)

    # Extract and print the best class weight and score
    best_class_weight = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_class_weight}")
    print(f"Best score: {best_score}")

    # Show the plot
    plt.show()


def scorer(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    # Calculate accuracy and precision
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Combine accuracy and precision (you can adjust weights as needed)
    # Equal weights for accuracy and precision
    combined_score = 0.1 * accuracy + 0.9 * f1

    return combined_score


def scorer_hda(estimator, X, y_true, occurrences):
    h_occurance = occurrences.get(0, 0)
    d_occurance = occurrences.get(1, 0)
    a_occurance = occurrences.get(2, 0)

    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    # Calculate the proportions of predicted occurrences
    totals = len(y_pred)
    h_predicted = round((sum(1 for p in y_pred if p == 0) / totals) * 100)
    d_predicted = round((sum(1 for p in y_pred if p == 1) / totals) * 100)
    a_predicted = round((sum(1 for p in y_pred if p == 2) / totals) * 100)

    print(d_occurance, d_predicted + 2 < d_occurance)
    if h_predicted - 7 > h_occurance:
        return 0
    if d_predicted + 2 < d_occurance:
        return 0
    if d_predicted > d_occurance:
        return 0
    if a_predicted - 7 > a_occurance:
        return 0

    # Calculate the absolute differences between predicted and actual occurrences
    diff_home = abs(h_predicted - h_occurance)
    diff_draw = abs(d_predicted - d_occurance)
    diff_away = abs(a_predicted - a_occurance)

    # Calculate a similarity score based on the differences
    max_diff = max(diff_home, diff_draw, diff_away)
    natural_fit = 1 - (max_diff / 100)  # Normalize to [0, 1]

    # Calculate accuracy and precision
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    combined_score = precision

    return combined_score


def scorer_f1(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Combine accuracy and precision (you can adjust weights as needed)
    # Equal weights for accuracy and precision
    combined_score = f1

    return combined_score


def scorer_accuracy_precision_f1_score(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    # Calculate accuracy, precision, and F1 score
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Combine accuracy, precision, and F1 score (you can adjust weights as needed)
    combined_score = 0.2 * accuracy + 0.2 * precision + \
        0.6 * f1

    return combined_score


def scorer_matthews_corrcoef(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc


def scorer_matthews_corrcoef_f1(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    mcc = matthews_corrcoef(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Combine balanced_acc, and F1 score (you can adjust weights as needed)

    combined_score = mcc + (0.8 * f1)
    combined_score = combined_score if combined_score < 1 else 1

    return combined_score


def scorer_balanced_accuracy_score(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return balanced_acc


def scorer_balanced_accuracy_score_f1(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Combine balanced_acc, and F1 score (you can adjust weights as needed)
    combined_score = 0.4 * balanced_acc + \
        0.6 * f1

    return combined_score


def scorer_cs(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    mcc = matthews_corrcoef(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)

    # Combine balanced_acc, and F1 score (you can adjust weights as needed)

    combined_score = f1
    combined_score = combined_score if combined_score < 1 else 1

    return combined_score


def scorer_over_under(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    mcc = matthews_corrcoef(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Combine mcc balanced_acc, and F1 score (you can adjust weights as needed)

    combined_score = .35 * mcc + .4 * balanced_acc + .55 * f1
    combined_score = combined_score if combined_score < 1 else 1

    return combined_score


def scorer_gg_ng(estimator, X, y_true):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    mcc = matthews_corrcoef(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Combine mcc balanced_acc, and F1 score (you can adjust weights as needed)

    combined_score = .35 * mcc + .6 * balanced_acc + .55 * f1
    combined_score = combined_score if combined_score < 1 else 1

    return combined_score
