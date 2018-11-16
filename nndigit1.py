import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt

train_epoch_lt=[]
train_err_lt=[]
test_epoch_lt=[]
test_err_lt=[]
Xtrain=[]
Ytrain=[]
Xtest=[]
Ytest=[]

plt.ion()

def sigmoid(z):
	return  float(1.0 / float((1.0 + math.exp(-1.0*z))))

def sigmoidDerivative(z):
	return  z*(1-z)

#***********************************************************************************************
def initialize_network():
	input_neurons=len(Xtrain[0])
	hidden_neurons=32
	output_neurons=10
	n_hidden_layers=4
	net=list()
	for h in range(n_hidden_layers):
		if(h!=0):
			input_neurons=len(net[-1])

		hidden_layer=[{'weights':np.random.uniform(low=-2,high=2,size=input_neurons)} for i in range(hidden_neurons)]
		net.append(hidden_layer)
	output_layer=[{'weights':np.random.uniform(low=-2,high=2,size=hidden_neurons)} for i in range(output_neurons)]
	net.append(output_layer)
	return net


#***********************************************************************************************
def forward_propagation(net,input):
	row=input
	# print(row)
	for layer in net:
		prev_input=np.array([])
		for neuron in layer:
			# print("row  ",row)
			sum=neuron['weights'].T.dot(row)
			# print(neuron['weights'])
			# print("sum ",sum)
			result=sigmoid(sum)
			# print("result=", result)
			neuron['result']=result
			prev_input=np.append(prev_input,[result])
		row=prev_input
		# print("row",row)
	return row

#***********************************************************************************************

def back_propagation(net,row,expected):
	for i in reversed(range(len(net))):
		layer=net[i]
		errors=np.array([])
		if(i==len(net)-1):
			results=[neuron['result'] for neuron in layer]
			errors=np.array(expected)-np.array(results)
			# print("errors",errors)
		else:
			for j in range(len(layer)):
				herror=0
				nextlayer=net[i+1]
				for neuron in nextlayer:
					herror+=(neuron['weights'][j]*neuron['delta'])
				errors=np.append(errors,[herror])
		for j in range(len(layer)):
			neuron=layer[j]
			# print("neuron[result]",neuron['result'])
			neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])


#***********************************************************************************************

def updateWeights(net,input,lrate):
	for i in range(len(net)):
		inputs= input
		if(i!=0):
			inputs=[neuron['result']for neuron in net[i-1]]
		for neuron in net[i]:
			for j in range(len(inputs)):
				# print("weights",neuron['weights'][j])
				# print("delta",neuron['delta'])
				neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]
				# print("updateWeights",neuron['weights'][j])
				# print("input", inputs[j])

#***********************************************************************************************

def training(net,epochs,lrate,n_outputs):
	errors=[]
	for epoch in range(epochs):
		train_sum_error=0
		test_sum_error=0
		for i,row in enumerate(Xtrain):
			row=np.asarray(row,dtype=float)
			train_outputs=forward_propagation(net,row)
			expected=[0.0 for i in range(n_outputs)]
			for j in range(0,10):
				if(Ytrain[i][j]==1):
					expected[j]=1
			for j in range(len(expected)):
				train_sum_error+=(expected[j]-train_outputs[j])**2
			# print("sum_error",sum_error)
			back_propagation(net,row,expected)
			updateWeights(net,row,0.25)

		train_epoch_lt.append(epoch+1)
		train_err_lt.append(train_sum_error/len(Xtrain))
		
		test_sum_error=0
		for i, row in enumerate(Xtest):
			row=np.asarray(row)
			test_outputs=forward_propagation(net,row)
			for j in range(len(Ytest[i])):
				test_sum_error+=(Ytest[i][j]-test_outputs[j])**2
		test_epoch_lt.append(epoch+1)
		test_err_lt.append(test_sum_error/len(Xtest))
		if (epoch+1)%1==0:
			print('epoch=%d,train_error=%f , test_error=%f'%(epoch+1,train_sum_error/len(Xtrain),test_sum_error/len(Xtest)))
		
		
	return errors

#***********************************************************************************************
def predict(network,row):
	outputs=forward_propagation(net,row)
	return outputs
#******************************************************************************
def feature_extractor(inputset):
	X1=[]
	Y1=[]
	inputlen=len(inputset)
	for index in range(inputlen):
		s1=inputset[index].strip('\n')
		img1 = s1.split(' ')
		digit1=int(img1[0])
		img1= img1[1:]
		for i in range(0,len(img1)):
			img1[i]=int(float(img1[i])+1)
			
		feature1=img1
		X1.append(feature1)
		label=[0 for i in range(0,10)]
		label[digit1]=1
		Y1.append(label)
	
	return X1,Y1
#***********************************************************************************************
fp= open("test.txt","r")#/content/drive/My Drive/colab/
st=fp.readlines()
length=len(st)
lt=[]
lt=random.sample(range(0,length),length)
train_ind=lt[:int(0.8*(length))]
test_ind=lt[int(0.8*(length)):]
trainingset=[]
testingset=[]
for ind in train_ind:
	trainingset.append(st[ind])
for ind in test_ind:
	testingset.append(st[ind])

fp.close()

Xtrain,Ytrain=feature_extractor(trainingset)
Xtest,Ytest=feature_extractor(testingset)


net=initialize_network()
errors=training(net,100,0.2,10)


plt.title("Error v/s Epochs")
plt.xlabel("No of Epoch")
plt.ylabel("Mean Square Error")
plt.xlim(1,100)
plt.ylim(0,1)
plt.plot(train_epoch_lt,train_err_lt,color='blue',label='Training Error')
plt.plot(test_epoch_lt,test_err_lt,color='red',label='Testing Error')
plt.legend(loc='upper right')
plt.show()
plt.pause(5667)
