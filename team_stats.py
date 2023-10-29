from composer import Composer

def calculate_team_stats(teamGames, team_id):
    if teamGames:
        for game in teamGames:
            hasResults = Composer.has_results(game)

            if hasResults:
                game['totals'] = 1  # Initialize totals to 1 for this game
                game['team_wins'] = 0
                game['draws'] = 0
                game['team_loses'] = 0
                game['goal_for'] = 0
                game['goal_against'] = 0

                if Composer.winning_side(game) == 'D':
                    game['draws'] = 1
                elif Composer.winning_side(game) == 'h' or Composer.winning_side(game) == 'a':
                    if Composer.winner_id(game) == team_id:
                        game['team_wins'] = 1
                    else:
                        game['team_loses'] = 1

                # Get goals for and goals against
                game['goal_for'] += Composer.get_scores(game, team_id)
                game['goal_against'] += Composer.get_scores(game, team_id, negate=True)
                
                # Calculate averages
                game['goal_for_avg'] = round(game['goal_for'] / game['totals'], 2)
                game['goal_against_avg'] = round(game['goal_against'] / game['totals'], 2)
                
            else:
                # If there are no results, initialize the fields to 0 or None
                game['totals'] = 0
                game['team_wins'] = 0
                game['draws'] = 0
                game['team_loses'] = 0
                game['goal_for'] = 0
                game['goal_against'] = 0
                game['goal_for_avg'] = None
                game['goal_against_avg'] = None
    return teamGames
