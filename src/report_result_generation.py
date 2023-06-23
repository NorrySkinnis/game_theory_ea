# import required module
import os
import numpy as np
from strategy_detector import StrategyDetector
from constants import STRATEGY_IDS
from player import Player
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

def count_from_data():
    # assign directory
    directory = f'./src/figures/data'
    
    # iterate over files in
    # that directory

    for filename in os.listdir(directory):
        strat_count_1 = {key: 0 for key in STRATEGY_IDS[1]}
        strat_count_2 = {key: 0 for key in STRATEGY_IDS[2]}
        f = os.path.join(directory, filename)
        # checking if it is a file
        data = np.load(f)
        sim = float(filename[filename.index('_sim_')+5])
        mc = float(filename[filename.index('_mc_')+4])
        e = float(filename[filename.index('_e_')+3:filename.index('_mr_')])
        mr = float(filename[filename.index('_mr_')+4:filename.index('_cr_')])
        cr = float(filename[filename.index('_cr_')+4:filename.index('.npy')])
        for i in range(data.shape[0]):
            if mc == 1:
                strat_count_1[i] += data[i,0]
            elif mc == 2:
                strat_count_2[i] += data[i,0]

def count_from_generated(mc, f):
    strat_count = {key: 0 for key in STRATEGY_IDS[mc]}
    n_players = 100000
    detector = StrategyDetector()
    players = [Player(identifier=id, n_matchups=60, n_games=15, memory_capacity=mc) 
                    for id in range(n_players)]

    for player in players:
        detector.detect_strategy(player, False)
        strat_count[player.strategy] += 1

    print('mc == ' + str(mc) + ':')
    for strat in STRATEGY_IDS[mc]:
        print(str(STRATEGY_IDS[mc][strat]) + ' & ' + str(100*strat_count[strat]/n_players) + ' \\\\\\hline')

def count_mutations(mc, mr):
    strat_count = {key: {key: 0 for key in STRATEGY_IDS[mc]} for key in STRATEGY_IDS[mc]}
    mutation_matrix = np.zeros((len(STRATEGY_IDS[mc]), len(STRATEGY_IDS[mc])))
    detector = StrategyDetector()
    n_players = 100000
    players = [Player(identifier=id, n_matchups=60, n_games=15, memory_capacity=mc) 
                    for id in range(n_players)]
    
    for player in players:
        detector.detect_strategy(player, False)
        before = player.strategy
        player.brain.mutate(mr)
        detector.detect_strategy(player, False)
        after = player.strategy
        strat_count[before][after] +=1

    for before in STRATEGY_IDS[mc]:
        total_before = sum(strat_count[before].values())
        # print(str(STRATEGY_IDS[mc][before]) + ': --->')
        for after in STRATEGY_IDS[mc]:
            total_after = strat_count[before][after]
            percentage = round(100*total_after/total_before, 1)
            mutation_matrix[before][after] = percentage
            # print('\t' + str(STRATEGY_IDS[mc][after]) + ':' + str(percentage))
    labels = [str(x) for x in STRATEGY_IDS[mc]]  # replace this with your labels

    plt.figure(figsize=(10, 7))  # Optional: You can set the figure size
    sns.heatmap(mutation_matrix, annot=True, cmap='YlGnBu', fmt=".2f", xticklabels=labels, yticklabels=labels)

    plt.title('Transition Matrix: Memory Capacity = ' + str(mc) + ', Mutation Rate = ' + str(mr))
    plt.xlabel('After Mutation')
    plt.ylabel('Original Strategy')

    plt.savefig(f'./src/figures/matrices/matrix_mc_{mc}_mr_{mr}.png')

def cosine_similarity(mc):
    detector = StrategyDetector()
    n_players = 100000
    players = [Player(identifier=id, n_matchups=60, n_games=15, memory_capacity=mc) 
                    for id in range(n_players)]
    vector_len = len(players[0].brain.W1.flatten()) + len(players[0].brain.W2.flatten()) + len(players[0].brain.Wb1.flatten()) + len(players[0].brain.Wb2.flatten())
    strat_means = {key: np.zeros(vector_len) for key in STRATEGY_IDS[mc]}
    strat_counts =  {key: 0 for key in STRATEGY_IDS[mc]}
    
    for player in players:
        detector.detect_strategy(player, False)
        strat = player.strategy
        vector = np.concatenate((player.brain.W1.flatten(), player.brain.W2.flatten(), player.brain.Wb1.flatten(), player.brain.Wb2.flatten()))
        strat_means[strat] += vector
        strat_counts[strat] += 1

    for strat in strat_means:
        strat_means[strat] /= strat_counts[strat]

    cosim_matrix = np.zeros((len(STRATEGY_IDS[mc]), len(STRATEGY_IDS[mc])))
    for strat1 in strat_means:
        for strat2 in strat_means:
            if strat1 == 7:
                print(strat_means[strat1], strat_means[strat2])
            cosim_matrix[strat1, strat2] = 1 - spatial.distance.cosine(strat_means[strat1], strat_means[strat2])

    plt.figure(figsize=(10, 7))  # Optional: You can set the figure size
    labels = [str(x) for x in STRATEGY_IDS[mc]]  # replace this with your labels
    sns.heatmap(cosim_matrix, annot=True, cmap='YlGnBu', fmt=".2f", xticklabels=labels, yticklabels=labels)

    plt.title('Cosine Similarity Matrix: Memory Capacity = ' + str(mc))
    plt.xlabel('Strategy')
    plt.ylabel('Strategy')

    plt.savefig(f'./src/figures/matrices/cosine_{mc}_.png')

mcs = [1,2]
# mrs = [0.1,0.5,1]

for mc in mcs:
    cosine_similarity(mc)
    # for mr in mrs:
    #     count_mutations(mc, mr)

