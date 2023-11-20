class Composer:
    @staticmethod
    def team(team, prefers=None):
        if prefers == 'short':
            return team['short_name']
        elif prefers == 'TLA':
            return team['tla']
        else:
            return team['name']

    @staticmethod
    def results(score, type='ft', show=None):
        if type == 'ft':
            h = score.get('home_scores_full_time')
            a = score.get('away_scores_full_time')
            if h is not None and a is not None:
                return h if show == 'h' else (a if show == 'a' else f'{h} - {a}')
            else:
                return '-'
        elif type == 'ht':
            h = score.get('home_scores_half_time')
            a = score.get('away_scores_half_time')
            if h is not None and a is not None:
                return h if show == 'h' else (a if show == 'a' else f'{h} - {a}')
            else:
                return '-'
        else:
            return score.get('winner', 'U')

    @staticmethod
    def winner(game, team_id):
        score = game['score']
        if not score or not score.get('winner'):
            return 'U'

        if score['winner'] == 'DRAW':
            return 'D'

        if score['winner'] == 'HOME_TEAM':
            if game['home_team_id'] == team_id:
                return 'W'
            else:
                return 'L'
        elif score['winner'] == 'AWAY_TEAM':
            if game['away_team_id'] == team_id:
                return 'W'
            else:
                return 'L'

        return 'U'

    @staticmethod
    def winning_side(game, integer=False):
        score = game['score']
        if not score or not score.get('winner'):
            return -1 if integer else 'U'

        if score['winner'] == 'DRAW':
            return 0 if integer else 'D'

        if score['winner'] == 'HOME_TEAM':
            return 1 if integer else 'h'
        elif score['winner'] == 'AWAY_TEAM':
            return 2 if integer else 'a'

        return -1 if integer else 'U'

    @staticmethod
    def goals(game, integer=False):
        score = game['score']
        if not score or not score.get('winner'):
            return -1 if integer else 'U'

         # Get the score data or provide default values if it's missing
        score_data = game.get('score', {})
        home_team_score = int(score_data.get('home_scores_full_time', 0))
        away_team_score = int(score_data.get('away_scores_full_time', 0))

        return home_team_score + away_team_score

    @staticmethod
    def gg(game, integer=False):
        score = game['score']
        if not score or not score.get('winner'):
            return -1 if integer else 'U'

         # Get the score data or provide default values if it's missing
        score_data = game.get('score', {})
        home_team_score = int(score_data.get('home_scores_full_time', 0))
        away_team_score = int(score_data.get('away_scores_full_time', 0))

        return home_team_score > 0 and away_team_score > 0

    @staticmethod
    def winner_id(game):
        score = game['score']
        if not score or not score.get('winner'):
            return None

        if score['winner'] == 'DRAW':
            return None

        if score['winner'] == 'HOME_TEAM':
            return game['home_team_id']
        elif score['winner'] == 'AWAY_TEAM':
            return game['away_team_id']

        return None

    @staticmethod
    def has_results(game):
        score = game['score']
        if not score or not score.get('winner'):
            return None

        if score['winner'] == 'DRAW':
            return True

        if score['winner'] == 'HOME_TEAM':
            return True
        elif score['winner'] == 'AWAY_TEAM':
            return True

        return None

    @staticmethod
    def get_scores(game, team_id, negate=False):
        home_team_id = game['home_team_id']
        away_team_id = game['away_team_id']

        # Get the score data or provide default values if it's missing
        score_data = game.get('score', {})
        home_team_score = score_data.get('home_scores_full_time', 0)
        away_team_score = score_data.get('away_scores_full_time', 0)

        # Calculate the scores for the specified team

        if home_team_score == away_team_score:
            scores = home_team_score
        else:
            if (negate == False):
                if team_id == home_team_id:
                    scores = home_team_score
                elif team_id == away_team_id:
                    scores = away_team_score
                else:
                    scores = 0
            else:
                if team_id == home_team_id:
                    scores = away_team_score
                elif team_id == away_team_id:
                    scores = home_team_score
                else:
                    scores = 0

        # Convert the scores to integers if they are strings
        scores = int(scores)

        return scores
