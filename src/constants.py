STRATEGY_IDS = {  0: 'Dove',
                  1: 'TFT',
                  2: 'Hawk',
                  3: 'RTFT', # Reverse TFT
                  4: 'GT', # Grim Trigger 
                  # 5: 'LR', # Limited Retaliation {1, 2, 3, 5, 6, 7}
                  # 6: 'SGT', # Soft Grim Trigger {1, 3, 4, 5, 6, 7}
                  # 7: 'CTFT', # Cautious TFT {0, 1, 3, 5, 7}
                  # 8: 'WSLS', # Win-Stay Lose-Shift {1, 2, 3, 4, 5}
                  5: '{5,7}', # Unknown
                  6: '{4,5,6,7}', # Unknown
                  7: '{0,2,3,4,6,7}', # Unknown
                  8: '{0,4,5,6,7}',
                  9: 'Undetermined'}

STRATEGY_CODES = { 1: [set(), {1}, {0, 1}, {0}], 
                   2: [set(), {1, 3}, {0, 1, 2, 3}, {0, 2}, {1, 2, 3}],
                   3: [set(), {1, 3, 5, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, 
                       {0, 2, 4, 6}, {1, 2, 3, 4, 5, 6, 7}, {5, 7},
                       {4, 5, 6, 7}, {0, 2, 3, 4, 6, 7},{0, 4, 5, 6, 7}]}

MAX_MEMORY_CAPACITY = 3


