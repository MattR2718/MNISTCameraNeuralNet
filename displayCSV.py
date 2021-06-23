import numpy
import matplotlib.pyplot as mp

#get train data
#dataFile = open("mnistDataset/mnist_train.csv", 'r')
dataFile = open("number.csv", 'r')
datalist = dataFile.readlines()
dataFile.close()
print("CSV got")

#plot data
#allValues = datalist[24].split(',')
allValues = datalist[0].split(',')
#imageArray = numpy.asfarray(allValues[1:]).reshape((28,28))
imageArray = numpy.asfarray(allValues).reshape((28,28))

for y in range(784):
	if y % 28 == 0:
		print()
	else:
		print(allValues[y], ", ", end="")

print("\n--------------------------------------------------------")
pixels = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,""^`'. "
print(pixels)
print("123456789012345678901234567890")
for y in range(784):
	if y % 28 == 0:
		print()
	else:
		print(pixels[(len(pixels)-1) - round(int(allValues[y])/len(pixels))], end="")

mp.imshow(imageArray, cmap='Greys', interpolation='None')
mp.show()
print("Data Plotted")

