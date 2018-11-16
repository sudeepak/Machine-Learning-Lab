import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import *

thetaFinal=[]
iterlist=[]
errLtTrain=[]
errLtTest=[]
iteration_no=0
#******************************************************************************
def plotData(X,Y):
	for i in range(len(X)):
		if(Y[i]==1):
			plt.scatter(X[i][1],X[i][2],color = 'green',marker='.')
		else:
			plt.scatter(X[i][1],X[i][2],color = 'blue',marker='.')
#*************************************************************************
def plotGraph(theta):
	x = [i/100 for i in range (0,101)]
	x = np.array(x)
	equation = '-' + str(theta[0]/theta[2])+ '-' + str(theta[1]/theta[2]) + "*x"
	y = eval(equation)
	plt.plot(x,y)
#*************************************************************************
def Sigmoid(z):
	return  float(1.0 / float((1.0 + math.exp(-1.0*z))))

#****************************************************************************
def Hypothesis(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)

#************************************************************************************
def wrong_preditions(Xtest,Ytest,thetaFinal,testlen):
	m=len(Ytest)
	err=0
	for i in range(m):
		xi = Xtest[i]
		hi = Hypothesis(thetaFinal,xi)
		if(hi>=0.5 and Ytest[i]!=1 or hi<0.5 and Ytest[i]!=0):
	 	 	err+=1
	return err

#************************************************************************************
def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	err=0
	error=0
	m=len(Y)
	for i in range(m):
		xi = X[i]
		hi = Hypothesis(theta,xi)
		if(hi>=0.5 and Y[i]!=1 or hi<0.5 and Y[i]!=0):
	 	 	err+=1
		if  ((Y[i] == 1)and (hi!=0)):
			error = Y[i] * math.log(hi)
		elif ((Y[i] == 0)and (hi!=1)):
			error += (1-Y[i]) * math.log(1-hi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	return J
#****************************************************************************************

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
#def Cost_Function_Derivative(X,Y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothesis(theta,X[i])
		error = (hi - Y[i])*xij
		sumOfErrors += error

	
	const = float(alpha)/float(m)
	J = const * sumOfErrors
	return J
	# gradient=[0 for i in range(len(X[0]))]
	# gradient=np.asarray(gradient)
	# for i in range(m):
	# 	z = 0
	# 	for j in range(len(theta)):
	# 		z += Y[i]*X[i][j]*theta[j]
	# 	res1=Sigmoid(-z)
	# 	const=(Y[i]*-1)*res1/m
	# 	gradient=np.add(gradient,np.asarray(X[i])*const)
	# return list(gradient)



#****************************************************************************

def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	CFDerivative=[]
	# CFDerivative = Cost_Function_Derivative(X,Y,theta,0,m,alpha)
	for j in range(len(theta)):
		CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		#new_theta_value = theta[j] - alpha*CFDerivative[j]
		new_theta.append(new_theta_value)
	return new_theta

#*******************************************************************************

def Logistic_Regression(Xtrain,Ytrain,Xtest,Ytest,alpha,theta,num_iters):
	testlen= len(Ytest)
	trainlen=len(Ytrain)
	global errLtTest,errLtTrain,iteration_no,iterlist,thetaFinal

	for i in range(num_iters):
		iteration_no+=1
		new_theta = Gradient_Descent(Xtrain,Ytrain,theta,trainlen,alpha)
		theta = new_theta
		err1=0
		err2=0
		err1=Cost_Function(Xtrain,Ytrain,theta,trainlen)
		err2=Cost_Function(Xtest,Ytest,theta,testlen)
		errLtTrain.append(err1)
		errLtTest.append(err2)
		iterlist.append(iteration_no)
		figure(2)
		scatter(iteration_no,err1,color='blue',marker='.',s=1)
		scatter(iteration_no,err2,color='green',marker='.',s =1)
		print(iteration_no," ",err1," ",err2)
		# plotGraph(theta)
		x = [i/100 for i in range (0,101)]
		x = np.array(x)
		equation = '-' + str(theta[2]/theta[1])+ "*x*x"+'-' + str(theta[0]/theta[1])
		y = eval(equation)
		y=np.sqrt(y)
		figure(1)
		p=plt.plot(x,y)
		pause(0.0000000000001)
		if(i==0 or i!=num_iters-1):
		 	 p.pop(0).remove()
	thetaFinal=theta
	print("Total Wrong outputs for Testing Data ",testlen)
	err=wrong_preditions(Xtest,Ytest,thetaFinal,testlen)
	print(err)
	print("percentage Error = ")
	print(err/testlen)

#***********************************************************************************

def feature_extractor(inputset,type):
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
			lt1=[float(i) for i in img1]

			
			output1= np.asarray(lt1, dtype=int) 
			output1=output1.reshape(16,16)
			r1,c1=output1.shape
			for i in range(r1):
				for j in range(c1):
					output1[i][j]=output1[i][j]+1

			avgBrightness1=0
			for i in range(r1):
				for j in range(c1):
					avgBrightness1+= output1[i][j]
			avgBrightness1/=256

			tempx1=np.zeros_like(output1)

			for i in range(r1):
				for j in range(c1):
					tempx1[i][j]=output1[i][16-1-j]
			
			symmetryx1=0
			for i in range(r1):
				for j in range(c1):
					symmetryx1+=abs(output1[i][j]-tempx1[i][j])
			symmetryx1/=256
			x11=avgBrightness1
			x12=symmetryx1

			feature1=[1,x11*x11,x12*x12]
			X1.append(feature1)
			if(digit1==1):
				Y1.append(1)
			else:
				Y1.append(0)
	# if(type==1):
		
	# 	for i in range(len(X1)):
	# 		if(Y1[i]==1):
	# 			plt.scatter(X1[i][1],X1[i][2],color = 'green',marker='.')
	# 		else:
	# 			plt.scatter(X1[i][1],X1[i][2],color = 'blue',marker='.')
	
	return X1,Y1


#**********************************************************************************************

fp= open("test.txt","r")
st=fp.readlines()
length=len(st)
trainingset=st[:int(0.7*(length))]
testingset=st[int(0.7*(length)):]
ct1=0
Xtrain=[]
Ytrain=[]
Xtest=[]
Ytest=[]
# f1=figure(1)
Xtrain,Ytrain=feature_extractor(trainingset,1)
Xtest,Ytest=feature_extractor(testingset,2)
fp.close()

alpha=0.1
theta=[0,0,0]
theta=np.asarray(theta)
f1=figure(1)
title("Logistic Regression")
for i in range(len(Xtrain)):
	if(Ytrain[i]==1):
		plt.scatter(Xtrain[i][1],Xtrain[i][2],color = 'green',marker='.')
	else:
		plt.scatter(Xtrain[i][1],Xtrain[i][2],color = 'blue',marker='.')

f2=figure(2)
xlim(0,10000)
ylim(0,2)

Logistic_Regression(np.asarray(Xtrain),np.asarray(Ytrain),np.asarray(Xtest),np.asarray(Ytest),alpha,theta,10000)

print(thetaFinal)

		





# f2=figure(2)
# xlim(0,10000)
# ylim(0,2)
title("Error v/s iterations")
plt.xlabel("No of iterations")
plt.ylabel("Error")
plt.legend(loc='upper right')
plt.plot(iterlist,errLtTest,color="blue",label='Testing Error')
plt.plot(iterlist,errLtTrain,color="green",label='Training Error')
# plt.legend(loc='upper right')
# plt.xlabel("No of iterations")
# plt.ylabel("Error")
show()

