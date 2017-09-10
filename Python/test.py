from network import *

network = Network(2, 2, 2)
network.setWeights([[0.15, 0.2], [0.25, 0.3]], [[0.4, 0.45], [0.5, 0.55]])
network.setBiases(0.35, 0.6)

for i in range(0, 20000):
	network.activate([0.05, 0.1])
	network.train([0.01, 0.99])

output = network.activate([0.05, 0.1])
print output
