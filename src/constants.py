STRATEGY_IDS = {1:{0:'Dove', 
                   1:'TFT', 
                   2:'CTFT', 
                   3:'Hawk', 
                   4:'RTFT',
                   5:'CRTFT', 
                   6:'Undetermined'},
                2:{0:'Dove', 
                   1:'TFT', 
                   2:'CTFT', 
                   3:'Hawk', 
                   4:'RTFT', 
                   5:'GT', 
                   6:'{0,2,3,4,5,6}', 
                   7:'{0,2,3}',
                   8:'{3}', 
                   9:'SG', # Silent grudge: when betrayed, will wait 1 turn and then betray back (is this the same as memory retaliation?)
                   10:'CSG', # Cautious silent grudge
                   11:'Forgiving',
                   12:'{2,3,6}',
                   13:'{0,2,3,4,5}',
                   14:'{0,2,3,5,6}',
                   15:'{1,6}',
                   16:'Undetermined'},
                3:{0:'Dove', 
                   1:'TFT', 
                   2:'CTFT', 
                   3:'Hawk', 
                   4:'RTFT', 
                   5:'GT', 
                   6:'LR', 
                   7:'SGT', 
                   8:'OTFT', 
                   9:'AFF', 
                   10:'MR', 
                   11:'{0,2,3,4,6,7}', 
                   12:'{0,4,5,6,7}', 
                   13:'{0,1,2,4,5,6,7}', 
                   14:'{0,2,4,5,6,7}', 
                   15:'OPP', 
                   16:'{0,1,4,5,6,7}', 
                   17:'{4,6,7}', 
                   18:'{5,6,7}', 
                   19:'{4,5,7}', 
                   20:'Undetermined'}
                }

# CTFT: Cautious TFT (Starts with D, then TFT)
# CRFTF: Cautious Reverse TFT (Starts with D, then RTFT)
# GT: Grim Trigger
# RTFT: Reverse TFT
# LR: Limited Retaliation
# SGT: Soft Grim Trigger
# AFF: Anti Flip Flop
# OPP: Opportunist
# OTFT: Opportunistic TFT (D)
# MR: Memory Retaliation
# Forgiving: Cooperates initially and if the opponent allied at least once
        
STRATEGY_CODES = {1: [set(), 
                      {1}, 
                      {1, 2}, 
                      {0, 1, 2}, 
                      {0},
                      {0, 2}], 
                  2: [set(), 
                      {1, 3, 6}, 
                      {1, 3, 4, 6}, 
                      {0, 1, 2, 3, 4, 5, 6}, 
                      {0, 2, 5}, 
                      {1, 2, 3, 6}, 
                      {0, 2, 3, 4, 5, 6},
                      {0, 2, 3},
                      {3},
                      {2, 3},
                      {2, 3, 4},                        
                      {3, 6},
                      {2,3,6},
                      {0,2,3,4,5},
                      {0,2,3,5,6},
                      {1,6}
                      ],
                  3: [set(), 
                      {1, 3, 5, 7}, 
                      {1, 3, 5, 7, 8}, 
                      {0, 1, 2, 3, 4, 5, 6, 7, 8}, 
                      {0, 2, 4, 6},
                      {1, 2, 3, 4, 5, 6, 7}, 
                      {1, 2, 3, 5, 6, 7},
                      {1, 3, 4, 5, 6, 7}, 
                      {0, 1, 3, 5, 7}, 
                      {5, 7}, 
                      {4, 5, 6, 7},
                      {0, 2, 3, 4, 6, 7}, 
                      {0, 4, 5, 6, 7},
                      {0, 1, 2, 4, 5, 6, 7}, 
                      {0, 2, 4, 5, 6, 7}, 
                      {0, 1, 2, 3, 4, 6, 7}, 
                      {0, 1, 4, 5, 6, 7}, 
                      {4, 6, 7}, 
                      {5, 6, 7},
                      {4, 5, 7}]}

MAX_MEMORY_CAPACITY = 3




