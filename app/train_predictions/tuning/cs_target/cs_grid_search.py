from sklearn.metrics import f1_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from app.train_predictions.hyperparameters.hyperparameters import hyperparameters_array_generator
import numpy as np
from itertools import product

# Set a random seed for reproducibility
np.random.seed(42)


def recursive(class_weight, outcomes, j):
    res = {}

    try:
        key = outcomes[j+1]

        for k in class_weight:
            _res = {key: round(k, 3)}
            res = {**res, **_res}
            return recursive(class_weight, outcomes, k)

    except:
        KeyError
        return res


def grid_search(model, train_frame, PREDICTORS, target, occurrences, is_random_search=False):

    n_estimators, min_samples_split, class_weight = hyperparameters_array_generator(
        train_frame, 4, 1.3, 2)

    outcomes = train_frame[target].unique()

    print(len(outcomes))

    # Count the occurrences of each class in cs_target
    class_counts = train_frame['cs_target'].value_counts()
    # Set a threshold for the minimum number of instances per class
    min_instances_threshold = 5

    # Identify classes with counts below the threshold
    classes_to_drop = class_counts[class_counts <
                                   min_instances_threshold].index

    # Filter out instances with classes to drop
    filtered_train_frame = train_frame[~train_frame['cs_target'].isin(
        classes_to_drop)]

    class_weight = np.linspace(1.0, 1.2, 2)
    __class_weight = []

    for combination in product(class_weight, repeat=int(2)):
        res = {key: round(x, 3) for key, x in zip(outcomes, combination)}
        __class_weight.append(res)

    class_weight = __class_weight

    # Creating a dictionary grid for grid search
    param_grid = {
        'random_state': [1],
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'class_weight': [{0: 1}],
        'min_samples_leaf': [1],
        'max_features': [None]
    }

    # Fitting grid search to the train data
    if not is_random_search:
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true, occurrences),
            verbose=2,
            n_jobs=-1,
        ).fit(train_frame[PREDICTORS], train_frame[target])
    else:
        gridsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true, occurrences),
            random_state=42,
            verbose=3,
        ).fit(train_frame[PREDICTORS], train_frame[target])

    # Extract and print the best class weight and score
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")

    return best_params

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


def scorer(estimator, X, y_true, occurrences):
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    # Calculate other required scorers and combine
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    combined_score = 0.5 * precision + 0.5 * f1

    return combined_score
