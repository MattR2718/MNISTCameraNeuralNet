import numpy
import scipy.special as sps
import matplotlib.pyplot as mp
import time as t

#neural network class definintion
class neuralNet:
    #init method
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        #set number of nodes in each layer
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        
        #set the learning rate
        self.lRate = learningRate
        
        #create weight matricies
        #wih = weight input->hidden
        #who = weight hidden->output
        #simple method
        #self.wih = np.random.rand(self.hNodes, self.iNodes) - 0.5
        #self.who = np.random.rand(self.oNodes, self.hNodes) - 0.5
        #more sophisticated method
        self.wih = numpy.random.normal(0.0, pow(self.iNodes, -0.5), (self.hNodes, self.iNodes))
        self.who = numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.oNodes, self.hNodes))
        
        #activation function is the sigmoid function
        self.activationFunction = lambda x: sps.expit(x)
    #train method
    def train(self, inputList, targetList):
        #convert input list into 2d array
        inputs = numpy.array(inputList, ndmin=2).T
        targets = numpy.array(targetList, ndmin=2).T
        
        #calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from the hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        
        #calculate signals into output layer
        outputInputs = numpy.dot(self.who, hiddenOutputs)
        #calculate signals emerging from output layer
        outputOutput = self.activationFunction(outputInputs)
        
        #error is (tagret - actual)
        outputErrors = targets - outputOutput
        #hidden layer error is the outputErrors, split by weights, recombined at hidden nodes
        hiddenErrors = numpy.dot(self.who.T, outputErrors)
        
        #update the weights for the links between the hidden and output layers
        self.who += self.lRate * numpy.dot((outputErrors * outputOutput * (1.0 - outputOutput)), numpy.transpose(hiddenOutputs))
        #update the weights for the links between the input and hidden layers
        self.wih += self.lRate * numpy.dot((hiddenErrors * hiddenOutputs * (10 - hiddenOutputs)), numpy.transpose(inputs))
    #query method to test input images on trained net
    def query(self, inputList):
        #convert input list into 2d array
        inputs = numpy.array(inputList, ndmin=2).T
        
        #calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from the hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        
        #calculate signals into output layer
        outputInputs = numpy.dot(self.who, hiddenOutputs)
        #calculate signals emerging from output layer
        outputOutput = self.activationFunction(outputInputs)
        
        return outputOutput

def run():
	pStart = t.time()
	#create a neural net
	#number of nodes
	inputNodes = 784
	hiddenNodes = 100
	outputNodes = 10
	#learning rate
	learningRate = 0.3
	#create instance of neural net
	n = neuralNet(inputNodes, hiddenNodes, outputNodes, learningRate)
	print("Neural Net Created")


	#get train data
	#Choose either small 100 image set to test that program is working or use full dataset
	#dataFile = open("mnistDataset/mnist_train_100.csv", 'r')
	dataFile = open("mnistDataset/mnist_train.csv", 'r')
	datalist = dataFile.readlines()
	dataFile.close()
	print("Training Data got")

	trStart = t.time()
	#train net
	for record in datalist:
	    allValues = record.split(',')
	    scaledInput = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
	    targets = numpy.zeros(outputNodes) + 0.01
	    targets[int(allValues[0])] = 0.99
	    n.train(scaledInput, targets)
	print("Net trained")
	trEnd = t.time()

	#get test data
	dataFile = open("number.csv", 'r')
	testData = dataFile.readlines()
	dataFile.close()
	
	print(type(testData))
	
	teStart = t.time()
	#test neural network
	#allValues = testData.split(',')
	
	#print('-------------')
	#for h in testData:
	#	print(h)
	#	print('########################')
	#print('-------------')	
	
	allValues = testData[0].split(',')
	#print(correctLabel, " correct label")
	scaledInput = (numpy.asfarray(allValues) / 255.0 * 0.99) + 0.01
	output = n.query(scaledInput)
	label = numpy.argmax(output)
	print('Network Guess: ', label)
	#print(scorecard)
	teEnd = t.time()


	pEnd = t.time()
	print("Total train time: ", trEnd - trStart, "s")
	print("Total test time: ", teEnd - teStart, "s")
	print("Total program time: ", pEnd - pStart, "s")
