import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt

epoch_lt=[]
err_lt=[]
Xtrain=[]
Ytrain=[]

plt.ion()

def sigmoid(z):
	return  float(1.0 / float((1.0 + math.exp(-1.0*z))))

def sigmoidDerivative(z):
	return  z*(1-z)

#***********************************************************************************************
def initialize_network():
	input_neurons=len(Xtrain[0])
	hidden_neurons=32
	output_neurons=2
	n_hidden_layers=4
	net=list()
	for h in range(n_hidden_layers):
		if(h!=0):
			input_neurons=len(net[-1])

		hidden_layer=[{'weights':np.random.uniform(low=-1,high=1,size=input_neurons)} for i in range(hidden_neurons)]
		net.append(hidden_layer)
	output_layer=[{'weights':np.random.uniform(low=-1,high=1,size=hidden_neurons)} for i in range(output_neurons)]
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
			errors=expected-np.array(results)
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
		sum_error=0
		for i,row in enumerate(Xtrain):
			row=np.asarray(row,dtype=float)
			outputs=forward_propagation(net,row)
			expected=[0.0 for i in range(n_outputs)]
			expected[Ytrain[i]]=1
			# print("expected= ", expected," output= ",outputs)
			# print(len(expected))
			# sum_error+=sum([(expected[j]-outputs[j])**2 for j in range(len(expected))])
			for j in range(len(expected)):
				sum_error+=(expected[j]-outputs[j])**2
			# print("sum_error",sum_error)
			back_propagation(net,row,expected)
			updateWeights(net,row,0.25)
		epoch_lt.append(epoch+1)
		err_lt.append(sum_error/len(Xtrain))
		if (epoch+1)%1==0:
			print('epoch=%d,error=%f'%(epoch+1,sum_error/len(Xtrain)))
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
		if(digit1==4 or digit1==1):
			# ct1+=1
			img1= img1[1:]
			for i in range(0,len(img1)):
				img1[i]=int(float(img1[i])+1)
				
			
			feature1=img1
			X1.append(feature1)
			if(digit1==1):
				Y1.append(1)
			else:
				Y1.append(0)
	
	return X1,Y1
#***********************************************************************************************
fp= open("test.txt","r")#/content/drive/My Drive/colab/
st=fp.readlines()
length=len(st)
lt=[]
lt=random.sample(range(0,length),length)
train_ind=lt[:int(0.7*(length))]
test_ind=lt[int(0.7*(length)):]
trainingset=[]
testingset=[]
for ind in train_ind:
	trainingset.append(st[ind])
for ind in test_ind:
	testingset.append(st[ind])
Xtrain,Ytrain=feature_extractor(trainingset)
# plt.imshow(np.array(Xtrain[0]).reshape(16,16))
# print(Xtrain[0])
Xtest,Ytest=feature_extractor(testingset)
fp.close()

net=initialize_network()
# print(net)
errors=training(net,20,0.25,2)
# pred=predict(net,np.array([1,0]))
print("Epoch",epoch_lt)
print("error ",err_lt)

plt.title("Error v/s Epochs")
plt.xlabel("No of Epoch")
plt.ylabel("Mean Square Error")
plt.xlim(1,50)
plt.ylim(0,1)
plt.plot(epoch_lt,err_lt,label='Training Error')
plt.legend(loc='upper right')

plt.show()
plt.pause(5667)



# output=np.argmax(pred)
# print(output)

