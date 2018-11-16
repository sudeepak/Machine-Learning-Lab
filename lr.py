import numpy as np
from matplotlib import pyplot as plt
import random as rn
# equation of line : y = ax+b

def gen_points():
	global xcor,ycor
	m = rn.uniform(-10,10)
	ite = 20
	noise = 150
	intercept = 10
	for i in range(ite):
		x = rn.uniform(-100,100)
		y = m*x+intercept
		xcor.append(x)
		ycor.append(y+rn.uniform(-noise,noise))

def calculate_coefficients():
	global xcor,ycor
	xcor = np.array(xcor)
	ycor = np.array(ycor)
	sumx = np.sum(xcor)
	sumy = np.sum(ycor)
	sumxy = np.sum(xcor*ycor)
	sumxx = np.sum(xcor*xcor)
	n = np.size(xcor)
	a = ((sumy*sumxx)-(sumx*sumxy))/(n*sumxx-sumx*sumx)
	b = (n*sumxy-sumx*sumy)/(n*sumxx-sumx*sumx)
	return (b,a)

def plot_points():
	global xcor,ycor
	for i in range(len(xcor)):
		plt.scatter(xcor[i],ycor[i],color = 'red',marker='.',linewidths=1)
		#plt.pause(.000001)		

def plot_line(a,b):
	global xcor,ycor
	x = [i for i in range(-1001,1001)]
	x = np.array(x)
	equation = str(a)+'*x +'+str(b)
	y = eval(equation)
	plt.plot(x,y)
	plt.show()

#plt.ion()
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-500,500)
plt.ylim(-500,500)
plt.title("Linear Regression")
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')
xcor=[]
ycor=[]
gen_points()
a,b = calculate_coefficients()
#print("a={} b={}".format(a,b))
plot_points()
plot_line(a,b)
