import numpy as np
import math
import os
import matplotlib.pyplot as plt


XORdata=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
x=XORdata[:,0:2]
y=XORdata[:,-1]
epoch_lt=[]
err_lt=[]

def sigmoid(z):
	return  float(1.0 / float((1.0 + math.exp(-1.0*z))))

def sigmoidDerivative(z):
	return  z*(1-z)

#***********************************************************************************************
def initialize_network():
	input_neurons=len(x[0])
	hidden_neurons=input_neurons+1
	output_neurons=2
	n_hidden_layers=1
	net=list()
	for h in range(n_hidden_layers):
		if(h!=0):
			input_neurons=len(net[-1])

		hidden_layer=[{'weights':np.random.uniform(size=input_neurons)} for i in range(hidden_neurons)]
		net.append(hidden_layer)
	output_layer=[{'weights':np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
	net.append(output_layer)
	return net


#***********************************************************************************************
def forward_propagation(net,input):
	row=input
	for layer in net:
		prev_input=np.array([])
		for neuron in layer:
			sum=neuron['weights'].T.dot(row)
			result=sigmoid(sum)
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
		else:
			for j in range(len(layer)):
				herror=0
				nextlayer=net[i+1]
				for neuron in nextlayer:
					herror+=(neuron['weights'][j]*neuron['delta'])
				errors=np.append(errors,[herror])
		for j in range(len(layer)):
			neuron=layer[j]
			neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])


#***********************************************************************************************

def updateWeights(net,input,lrate):
	for i in range(len(net)):
		inputs= input
		if(i!=0):
			inputs=[neuron['result']for neuron in net[i-1]]
		for neuron in net[i]:
			for j in range(len(inputs)):
				neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]
				# print("delta",neuron['delta'])

#***********************************************************************************************

def training(net,epochs,lrate,n_outputs):
	errors=[]
	for epoch in range(epochs):
		sum_error=0
		for i,row in enumerate(x):
			outputs=forward_propagation(net,row)
			expected=[0.0 for i in range(n_outputs)]
			expected[y[i]]=1
			sum_error+=sum([(expected[j]-outputs[j])**2 for j in range(len(expected))])
			back_propagation(net,row,expected)
			updateWeights(net,row,0.05)
		
		# plt.title("Error v/s Epochs")
		# plt.xlabel("No of Epoch")
		# plt.ylabel("Mean Square Error")
		# plt.xlim(1,10000)
		# plt.ylim(0,3)
		# plt.scatter(epoch+1,sum_error,color = 'blue',marker='.')
		# plt.show()
		epoch_lt.append(epoch+1)
		err_lt.append(sum_error)
		if (epoch+1)%1000==0:
			print('epoch=%d,error=%f'%(epoch+1,sum_error))
	return errors

#***********************************************************************************************
def predict(network,row):
	outputs=forward_propagation(net,row)
	return outputs

#***********************************************************************************************
net=initialize_network()
# print(net)
errors=training(net,100000,0.05,2)
pred=predict(net,np.array([1,0]))
# print(pred)

plt.title("Error v/s Epochs")
plt.xlabel("No of Epoch")
plt.ylabel("Mean Square Error")
plt.plot(epoch_lt,err_lt,color="blue",label='Training Error')
plt.legend(loc='upper right')
plt.xlim(1,100000)
plt.ylim(0,3)
plt.show()




# output=np.argmax(pred)
# print(output)
