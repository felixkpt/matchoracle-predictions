def scores():
    arr = {
            # '0 - 0' =>
            0: {'hda': 1, 'h_margin': 0, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 0, 'over25': 0, 'over35': 0},
            # '0 - 1' =>
            1: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 0, 'bts_margin': 0, 'over15': 0, 'over25': 0, 'over35': 0},
            # '0 - 2' =>
            2: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 2, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 0, 'over35': 0},
            # '0 - 3' =>
            3: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 3, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 0},
            # '0 - 4' =>
            4: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 4, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '0 - 5' =>
            5: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 5, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '0 - 6' =>
            6: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 6, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '0 - 7' =>
            7: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 7, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '0 - 8' =>
            8: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 8, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '0 - 9' =>
            9: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 9, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '0 - 10' => 
            10: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 10, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '1 - 0' =>
             11: {'hda': 0, 'h_margin': 1, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 0, 'over25': 0, 'over35': 0},
            # '1 - 1' =>
             12: {'hda': 1, 'h_margin': 0, 'd_margin': 1, 'away_margin': 0, 'bts': 1, 'bts_margin': 1, 'over15': 1, 'over25': 0, 'over35': 0},
            # '1 - 2' =>
             13: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 0},
            # '1 - 3' =>
             14: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 2, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '1 - 4' =>
             15: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 3, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '1 - 5' =>
             16: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 4, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '1 - 6' =>
             17: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 5, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '1 - 7' =>
             18: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 6, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '1 - 8' =>
             19: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 7, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '1 - 9' =>
             20: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 8, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '1 - 10' =>
             21: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 9, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '2 - 0' =>
             22: {'hda': 0, 'h_margin': 2, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 0, 'over35': 0},
            # '2 - 1' =>
             23: {'hda': 0, 'h_margin': 1, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 0},
            # '2 - 2' =>
             24: {'hda': 1, 'h_margin': 0, 'd_margin': 2, 'away_margin': 0, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '2 - 3' =>
             25: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '2 - 4' =>
             26: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 2, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '2 - 5' =>
             27: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 3, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '2 - 6' =>
             28: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 4, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '2 - 7' =>
             29: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 5, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '2 - 8' =>
             30: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 6, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '2 - 9' =>
             31: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 7, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '2 - 10' =>
             32: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 8, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 0' =>
             33: {'hda': 0, 'h_margin': 3, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 0},
            # '3 - 1' =>
             34: {'hda': 0, 'h_margin': 2, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 2' =>
             35: {'hda': 0, 'h_margin': 1, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 3' =>
             36: {'hda': 1, 'h_margin': 0, 'd_margin': 3, 'away_margin': 0, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 4' =>
             37: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 5' =>
             37: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 2, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 6' =>
             39: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 3, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 7' =>
             40: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 4, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 8' =>
             41: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 5, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 9' =>
             42: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 6, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '3 - 10' =>
             43: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 7, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 0' =>
             44: {'hda': 0, 'h_margin': 4, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 1' =>
             45: {'hda': 0, 'h_margin': 3, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 2' =>
             46: {'hda': 0, 'h_margin': 2, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 3' =>
             47: {'hda': 0, 'h_margin': 1, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 4' =>
             48: {'hda': 1, 'h_margin': 0, 'd_margin': 4, 'away_margin': 0, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 5' =>
             49: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 6' =>
             50: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 2, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 7' =>
             51: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 3, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 8' =>
             52: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 4, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 9' =>
             53: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 5, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '4 - 10' =>
             54: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 6, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 0' =>
             55: {'hda': 0, 'h_margin': 5, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 1' =>
             56: {'hda': 0, 'h_margin': 4, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 1, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 2' =>
             57: {'hda': 0, 'h_margin': 3, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 3' =>
             58: {'hda': 0, 'h_margin': 2, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 4' =>
             59: {'hda': 0, 'h_margin': 1, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 5' =>
             60: {'hda': 1, 'h_margin': 0, 'd_margin': 5, 'away_margin': 0, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 6' =>
             61: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 7' =>
             62: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 2, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 8' =>
             63: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 3, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 9' =>
             64: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 4, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '5 - 10' =>
             65: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 5, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 0' =>
             66: {'hda': 0, 'h_margin': 6, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 1' =>
             67: {'hda': 0, 'h_margin': 5, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 1, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 2' =>
             68: {'hda': 0, 'h_margin': 4, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 3' =>
             69: {'hda': 0, 'h_margin': 3, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 4' =>
             70: {'hda': 0, 'h_margin': 2, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 5' =>
             71: {'hda': 0, 'h_margin': 1, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 6' =>
             72: {'hda': 1, 'h_margin': 0, 'd_margin': 6, 'away_margin': 0, 'bts': 1, 'bts_margin': 6, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 7' =>
             73: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 1, 'bts_margin': 6, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 8' =>
             74: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 2, 'bts': 1, 'bts_margin': 6, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 9' =>
             75: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 3, 'bts': 1, 'bts_margin': 6, 'over15': 1, 'over25': 1, 'over35': 1},
            # '6 - 10' =>
             76: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 4, 'bts': 1, 'bts_margin': 6, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 0' =>
             77: {'hda': 0, 'h_margin': 7, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 1' =>
             78: {'hda': 0, 'h_margin': 6, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 1, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 2' =>
             79: {'hda': 0, 'h_margin': 5, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 3' =>
             80: {'hda': 0, 'h_margin': 4, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 4' =>
             81: {'hda': 0, 'h_margin': 3, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 5' =>
             82: {'hda': 0, 'h_margin': 2, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 6' =>
             83: {'hda': 0, 'h_margin': 1, 'd_margin': 7, 'away_margin': 0, 'bts': 1, 'bts_margin': 6, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 7' =>
             84: {'hda': 1, 'h_margin': 0, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 7, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 8' =>
             85: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 1, 'bts_margin': 7, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 9' =>
             86: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 2, 'bts': 1, 'bts_margin': 7, 'over15': 1, 'over25': 1, 'over35': 1},
            # '7 - 10' =>
             87: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 3, 'bts': 1, 'bts_margin': 7, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 0' =>
             88: {'hda': 0, 'h_margin': 8, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 1' =>
             89: {'hda': 0, 'h_margin': 7, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 1, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 2' =>
             90: {'hda': 0, 'h_margin': 6, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 3' =>
             91: {'hda': 0, 'h_margin': 5, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 4' =>
             92: {'hda': 0, 'h_margin': 4, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 5' =>
             93: {'hda': 0, 'h_margin': 3, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 6' =>
             94: {'hda': 0, 'h_margin': 2, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 6, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 7' =>
             95: {'hda': 0, 'h_margin': 1, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 7, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 8' =>
             96: {'hda': 1, 'h_margin': 0, 'd_margin': 8, 'away_margin': 0, 'bts': 1, 'bts_margin': 8, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 9' =>
             97: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 1, 'bts_margin': 8, 'over15': 1, 'over25': 1, 'over35': 1},
            # '8 - 10' =>
             98: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 2, 'bts': 1, 'bts_margin': 8, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 0' =>
             99: {'hda': 0, 'h_margin': 9, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 1' => 
            100: {'hda': 0, 'h_margin': 8, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 1, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 2' => 
            101: {'hda': 0, 'h_margin': 7, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 3' => 
            102: {'hda': 0, 'h_margin': 6, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 4' => 
            103: {'hda': 0, 'h_margin': 5, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 5' => 
            104: {'hda': 0, 'h_margin': 4, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 6' => 
            105: {'hda': 0, 'h_margin': 3, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 6, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 7' => 
            106: {'hda': 0, 'h_margin': 2, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 7, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 8' => 
            107: {'hda': 0, 'h_margin': 1, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 8, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 9' => 
            108: {'hda': 1, 'h_margin': 0, 'd_margin': 9, 'away_margin': 0, 'bts': 1, 'bts_margin': 9, 'over15': 1, 'over25': 1, 'over35': 1},
            # '9 - 10' => 
            108: {'hda': 2, 'h_margin': 0, 'd_margin': 0, 'away_margin': 1, 'bts': 1, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 0' =>
            110: {'hda': 0, 'h_margin': 10, 'd_margin': 0, 'away_margin': 0, 'bts': 0, 'bts_margin': 0, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 1' => 
            111: {'hda': 0, 'h_margin': 9, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 1, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 2' => 
            112: {'hda': 0, 'h_margin': 8, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 2, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 3' => 
            113: {'hda': 0, 'h_margin': 7, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 3, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 4' => 
            114: {'hda': 0, 'h_margin': 6, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 4, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 5' => 
            115: {'hda': 0, 'h_margin': 5, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 5, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 6' => 
            116: {'hda': 0, 'h_margin': 4, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 6, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 7' => 
            116: {'hda': 0, 'h_margin': 3, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 7, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 8' => 
            118: {'hda': 0, 'h_margin': 2, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 8, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 9' => 
            119: {'hda': 0, 'h_margin': 1, 'd_margin': 0, 'away_margin': 0, 'bts': 1, 'bts_margin': 9, 'over15': 1, 'over25': 1, 'over35': 1},
            # '10 - 10' =>
            120: {'hda': 1, 'h_margin': 0, 'd_margin': 10, 'away_margin': 0, 'bts': 1, 'bts_margin': 10, 'over15': 1, 'over25': 1, 'over35': 1},
    }
    
    return arr