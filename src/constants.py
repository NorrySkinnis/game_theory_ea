import torch

STRATS = {
			0: 'TitForTat',
			1: "Dove",
			2: "Hawk",
			3: "Random/Undetermined",
            4: "GrimTrigger"
		}

ACTIONS = {"C": 0, "D": 1}

MAX_MEMORY_CAPACITY = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
