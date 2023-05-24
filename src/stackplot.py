import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def plot(data, name, title='', save=True):
	plt.figure()
	generations = np.arange(data.shape[0])
	y = [[] for i in strats.keys()]
	for gen_data in data:
		c = Counter(gen_data)
		for k in strats.keys():
			if k not in c.keys():
				c[k] = 0
		for k in c.keys():
			y[k].append(c[k])

	# print(generations.shape)
	# print(y.shape)
	plt.stackplot(generations, y)
	plt.title(title)
	plt.xlabel('Generations')
	plt.ylabel('Relative strategy distribution (%)') # make labels so it lines up with colour
	plt.legend([strats[k] for k in strats.keys()])

	if save:
		plt.savefig('figures/'+name+'.png')
	else:
		plt.show()


if __name__ == '__main__':
	# generate some dummy data
	generations = 100
	gen_size = 100
	strats = {0: 'TitForTat', 1: "Dove", 2: "Hawk", 3: "Random"}
	dummy_data = np.random.choice(list(strats.keys()), (generations, gen_size), p=[0.25, 0.25, 0.25, 0.25])

	plot(dummy_data, "dummy_data", title="Distribution over time per strategy")
