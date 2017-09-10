import random
import math

class Network:
	
	def __init__(self, *args):
		#print "Creating network..."
		self.layers = []
		self.output = []
		self.learningRate = 0.5
		self.totalError = None
		
		#Create neurons in layers
		for i in range(0, len(args)):
			self.layers.append([])
			if i == 0:
				layerId = 'i'
			elif i < len(args) - 1:
				layerId = 'h'
			else:
				layerId = 'o'
			for j in range(0, args[i]):
				id_ = layerId + str(j + 1)
				self.layers[i].append(Neuron(id_))
		
		#Create synapses between neurons
		l = 0
		for i in range(1, len(self.layers)):
			for j in range(0, len(self.layers[i])):
				for k in range(0, len(self.layers[i - 1])):
					l += 1
					Synapse('w' + str(l), self.layers[i - 1][k], self.layers[i][j], random.random())
	
	def setWeights(self, *args):
		#print "Setting weights..."
		for i in range(0, len(args)):
			for j in range(0, len(args[i])):
				for k in range(0, len(args[i][j])):
					synapse = self.layers[i+1][j].inputSynapses[k]
					weight = args[i][j][k]
					synapse.weight = weight
					#print "%s weight set to %s" % (synapse.id_, weight)
	
	def setBiases(self, *args):
		#print "Setting biases..."
		for i in range(0, len(args)):
			for j in range(0, len(self.layers[i+1])):
				neuron = self.layers[i+1][j]
				neuron.bias = args[i]
				#print "%s bias set to %s" % (neuron.id_, args[i])
	
	def activate(self, values):
		#print "Activating network..."
		#Put value in input neurons
		for i in range(0, len(self.layers[0])):
			self.layers[0][i].value = values[i]
			#print "%s set to %s" % (self.layers[0][i].id_, values[i])
		
		#Forward pass through network
		for i in range(1, len(self.layers)):
			for j in range(0, len(self.layers[i])):
				neuron = self.layers[i][j]
				
				#Get net input
				net = neuron.bias
				for k in range(0, len(neuron.inputSynapses)):
					synapse = neuron.inputSynapses[k]
					net += synapse.getValue()
				
				#Activation function (sigmoid)
				out = (1 / (1 + math.exp(-1 * net)))
				
				neuron.value = out
				#print "%s set to %s" % (neuron.id_, out)
		
		#Return outputs
		output = []
		for i in range(0, len(self.layers[len(self.layers) - 1])):
			output.append(self.layers[len(self.layers) - 1][i].value)
		self.output = output
		return output
		
	def train(self, target):
		#print "Training network..."
		#Get total error
		self.totalError = 0;
		for i in range(0, len(self.output)):
			self.totalError += 0.5 * (target[i] - self.output[i]) ** 2
		
		#print "Total error at %s" % self.totalError
		
		#Backward pass
		for i in range(len(self.layers) - 1, -1, -1):
			for j in range(0, len(self.layers[i])):
				neuron = self.layers[i][j]
				if i == (len(self.layers) - 1):
					neuron.delta = -1*(target[j] - neuron.value) * neuron.value * (1 - neuron.value)
					#print "%s delta set to %s" % (neuron.id_, neuron.delta)
				elif i != 0:
					neuron.delta = 0;
					for k in range(0, len(neuron.outputSynapses)):
						synapse = neuron.outputSynapses[k]
						neuron.delta += synapse.weight * synapse.toNeuron.delta
					neuron.delta *= neuron.value * (1 - neuron.value)
					#print "%s delta set to %s" % (neuron.id_, neuron.delta)
				
				#Update weights from delta values
				for k in range(0, len(neuron.outputSynapses)):
					synapse = neuron.outputSynapses[k]
					synapse.weight = synapse.weight - self.learningRate * (synapse.toNeuron.delta * synapse.fromNeuron.value)
					#print "%s weight changed to %s" % (synapse.id_, synapse.weight)
		

class Neuron:
	def __init__(self, id_):
		#print "Created neuron %s" % (id_)
		self.id_ = id_;
		self.value = 0;
		self.delta = 0;
		self.inputSynapses = [];
		self.outputSynapses = [];
		self.bias = 0;

class Synapse:
	def __init__(self, id_, fromNeuron, toNeuron, weight):
		#print "Created synapse %s from neuron %s to neuron %s" % (id_, fromNeuron.id_, toNeuron.id_)
		self.id_ = id_
		self.weight = weight
		self.fromNeuron = fromNeuron
		self.toNeuron = toNeuron
		
		fromNeuron.outputSynapses.append(self)
		toNeuron.inputSynapses.append(self)
	
	def getValue(self):
		return self.fromNeuron.value * self.weight

""" Example:
network = Network(2, 2, 2)
network.setWeights([[0.15, 0.2], [0.25, 0.3]], [[0.4, 0.45], [0.5, 0.55]])
network.setBiases(0.35, 0.6)

for i in range(0, 20000):
	network.activate([0.05, 0.1])
	network.train([0.01, 0.99])

output = network.activate([0.05, 0.1])
print output
"""
