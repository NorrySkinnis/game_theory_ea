STRATEGY_IDS = {  0: 'Dove',
                  1: 'TFT',
                  2: 'Hawk',
                  3: 'RTFT',
                  4: 'SGT',
                  5: 'Undetermined'}

STRATEGY_CODES = { 1: [set(), {1}, {0, 1}, {0}], 
                   2: [set(), {1, 3}, {0, 1, 2, 3}, {0, 2}, {1, 2, 3}],
                   3: [set(), {1, 3, 5, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, 
                       {0, 2, 4, 6}, {1, 2, 3, 4, 5, 6, 7}],
                   4: [set(), {1, 3, 5, 7, 9, 11, 13, 15}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                       {0, 2, 4, 6, 8, 10, 12, 14}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}]}

MAX_MEMORY_CAPACITY = 4


