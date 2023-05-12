from environment import Environment as env

if __name__=='main':
    n_games = 100
    n_matchups = 100
    n_generations = 100

    env = env()
    players = env.run(n_games=n_games, 
                      n_matchups=n_matchups, 
                      n_generations=n_generations)

