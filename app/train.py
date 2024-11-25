from dateutil import parser
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from app.configs.active_competitions.competitions_data import get_competitions, get_trained_competitions
from app.run_train import run_train
from app.configs.active_competitions.competitions_data import update_job_status


# Calculate from_date and to_date
TRAIN_TO_DATE = datetime.today() + relativedelta(days=-30 * 1)

async def train(user_token, prediction_type, request_data):
    print("\n............... START TRAIN PREDICTIONS ..................\n")

    # Extract values from request_data
    target = request_data.get('target')
    prefer_saved_matches = request_data.get('prefer_saved_matches', True)
    is_grid_search = request_data.get('is_grid_search', False)
    ignore_trained = request_data.get('ignore_trained', False)
    last_action_date = request_data.get('retrain_if_last_train_is_before')
    if last_action_date:
        last_action_date = parser.parse(last_action_date).strftime("%Y-%m-%d %H:%M:%S")


    print(f"Main Prediction Target: {target if target else 'all'}")
    print(f"Training to {TRAIN_TO_DATE}")

    print(f'Train/test max limit: {request_data.get("per_page", 1000)}\n')

    # If competition_id is provided, use it; otherwise, fetch from the backend API
    competition_ids = [{"id": request_data.get('competition'), "name": "N/A", "games_counts": 0}
                       ] if request_data.get('competition') is not None else get_competitions(user_token, request_data.get('per_page', 380))

    # Set last_action_date dynamically
    last_action_date = last_action_date if last_action_date is not None else (datetime.now() - timedelta(hours=24 * 7)
                                                                              ).strftime("%Y-%m-%d %H:%M:%S")
    
    if ignore_trained:
        print('Ignoring trained competitions...')
    else: 
        print(
            f"Performing train on untrained competitions or those trained on/before: {last_action_date}")

    trained_competition_ids = [] if ignore_trained else get_trained_competitions(
        last_action_date, True)
    
    arr = []
    for compe in competition_ids:
        if len(trained_competition_ids) == 0 or compe['id'] not in trained_competition_ids:
            arr.append(compe)

    print(f"Competitions info (with >={request_data.get('per_page', 380)} matches):")
    print(f"Active: {len(competition_ids)}, To be trained: {len(arr)}, Already trained: {len(competition_ids)-len(arr)}\n")

    competition_ids = arr

    # Starting points for loops
    start_from = [12, 6, 4]
    end_at = [12, 6, 4]

    for history_limit_per_match in [6, 10, 12, 15]:
        # Skip if current history_limit_per_match is less than the specified starting point
        if history_limit_per_match < start_from[0] or history_limit_per_match > end_at[0]:
            continue
        else:
            for current_ground_limit_per_match in [4, 6, 8]:
                if current_ground_limit_per_match < start_from[1] or current_ground_limit_per_match > end_at[1]:
                    continue
                else:
                    for h2h_limit_per_match in [4, 6, 8]:
                        if h2h_limit_per_match < start_from[2] or h2h_limit_per_match > end_at[2]:
                            continue
                        else:
                            # Generate prediction type based on loop parameters
                            PREDICTION_TYPE = (
                                prediction_type
                                or f"regular_prediction_{history_limit_per_match}_{current_ground_limit_per_match}_{h2h_limit_per_match}_{request_data.get('per_page', 380)}"
                            )

                            i = 0
                            # Loop over competition IDs
                            for competition in competition_ids:
                                i += 1
                                COMPETITION_ID = competition['id']

                                compe_data = {
                                    'id': COMPETITION_ID,
                                    'name': competition['name'],
                                    "games_counts": competition['games_counts'],
                                    'prediction_type': PREDICTION_TYPE
                                }
                                # Parameters for training
                                be_params = {
                                    'history_limit_per_match': history_limit_per_match,
                                    'current_ground_limit_per_match': current_ground_limit_per_match,
                                    'h2h_limit_per_match': h2h_limit_per_match,
                                    'to_date': TRAIN_TO_DATE,
                                }

                                print(
                                    f"{i}/{len(competition_ids)}. COMPETITION #{COMPETITION_ID}, ({competition['name']})")
                                print(
                                    f"***** START TRAIN PREDICTS FOR {COMPETITION_ID} *****")

                                # Start the timer
                                start_time = datetime.now()
                                
                                # Run training for the current configuration
                                run_train(user_token, compe_data=compe_data, target=target, be_params=be_params,
                                          prefer_saved_matches=prefer_saved_matches, is_grid_search=is_grid_search, per_page=request_data.get('per_page', 380), start_time=start_time)
                                
                                # End the timer
                                end_time = datetime.now()
                                duration = end_time - start_time
                                print(
                                    f"***** END TRAIN PREDICTS FOR {COMPETITION_ID} took {duration.total_seconds() / 60:.2f} minutes *****\n")

    
    job_id = request_data.get('job_id')
    if job_id:
        update_job_status(user_token, job_id, status="completed")
    
    print(f"\n....... END TRAIN PREDICTIONS, Happy coding! ........")

