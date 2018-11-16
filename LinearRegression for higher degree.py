import numpy as np
from matplotlib import pyplot as plt
import random as rn
import math
# equation of line : y = ax+b

def gen_points():
	global xcor,ycor
	m = rn.uniform(-1,1)
	ite = 50
	noise = 100
	intercept = 10
	for i in range(ite):
		x = rn.uniform(-500,500)
		# y = m*x+intercept
		y=math.sin(math.radians(x))
		xcor.append(x)
		ycor.append(y+rn.uniform(-noise,noise))
		
def calculate_reg_wts(i):
	global xcor,ycor,lis
	wt_list = []
	#new_column = np.array([[xcor[j]**i] for j in range(len(xcor))])
	#lis = np.append(lis,new_column, axis = 1)
	x = np.asmatrix(lis)
	y = np.asmatrix(ycor)
	y = np.transpose(y)
	a = x.transpose()*x
	i = np.identity(i+1)
	i = np.asmatrix(i);
	a = a + 3*i
	b = x.transpose()*y
	print(a)
	c = np.linalg.inv(a)
	c = c*b
	wt_list = c.tolist()
	return wt_list
	
def calculate_wts(i):
	global xcor,ycor,lis
	wt_list = []
	new_column = np.array([[xcor[j]**i] for j in range(len(xcor))])
	lis = np.append(lis,new_column, axis = 1)
	x = np.asmatrix(lis)
	y = np.asmatrix(ycor)
	y = np.transpose(y)
	a = x.transpose()*x
	b = x.transpose()*y
	c = np.linalg.inv(a)
	c = c*b
	wt_list = c.tolist()
	return wt_list

def plot_points():
	global xcor,ycor
	for i in range(len(xcor)):
		plt.scatter(xcor[i],ycor[i],color = 'red',marker='+')
		plt.pause(.000001)		

def plot_curve(wt_list,flg):
	global xcor,ycor,lis
	print(lis)
	x = [i for i in range(-1001,1001)]
	x = np.array(x)
	equation = str(wt_list[0][0]) + '+'
	var = "x"
	op = "*"
	for i in range(1,len(wt_list)):
		wt = wt_list[i][0]
		equation = equation + str(wt) + '*' + ((var+op)*i)[:-1] + '+'
	equation = equation[:-1]	
	print(equation)
	y = eval(equation)
	#plt.pause(0.7)
	if(flg==1):
		plt.plot(x,y,color="blue",label="without Regularization")
	else:
		plt.plot(x,y,color="green",label="with Regularization")
	plt.legend(loc='upper right')

#plt.ion()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression for Higher Degree Curves")
plt.xlim(-600,600)
plt.ylim(-200,200)
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')
xcor=[]
ycor=[]
gen_points()
plot_points()
#plot_line(a,b)

lis = [[1] for i in range(len((xcor)))]
lis = np.array(lis)
#np.reshape(lis,(len(xcor),1))
degree = 17
for i in range(1,degree+1):
	plt.title("Linear Regression for Higher Degree Curve( Degree %d)" %(i))
	ch=input("Want to plot curve for next higher Degree[y/n]\n")
	if(ch=='y' or ch=='Y'):
		wt_list = calculate_wts(i)
		plot_curve(wt_list,1)
		wt_list = calculate_reg_wts(i)
		plot_curve(wt_list,0)
		plt.pause(1)
		if(i!=0 and i!=degree ):
			plt.gca().lines[-1].remove()
			plt.gca().lines[-1].remove()

plt.show()	

