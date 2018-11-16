import matplotlib.pyplot as plt
import random as rn
import numpy as np

n=40
train_set=[]
x1_0=[]
x2_0=[]
x1_1=[]
x2_1=[]
weights=[]
def gen_points():
	global n,train_set
	for i in range(n):
		x=rn.uniform(-6,6)
		y=rn.uniform(-10,3)
		z=1
		train_set.append([x,y,z])
		x=rn.uniform(7,15)
		y=rn.uniform(1,10)
		z=-1
		train_set.append([x,y,z])

def plot_graph(dataset):
                global data,weights
                print(1, weights)
                for t in dataset:
                    if(t[-1] == -1):
                            x1_0.append(t[0])
                            x2_0.append(t[1])
                    else:
                            x1_1.append(t[0])
                            x2_1.append(t[1])

                plt.scatter(x1_0, x2_0, color= "green",label= "0", marker= ".")
                plt.scatter(x1_1, x2_1, color= "blue",  label= "1",marker= ".")
                plt.legend(loc='upper right')

line = -1
def plot_line():
                xs = np.arange(-6, 15, 0.01)
                global weights, line
                if line != -1:
                        line.pop(0).remove()
                line = plt.plot(xs, (-weights[2]-xs*weights[0])/weights[1])
                plt.pause(0.1)
                plt.draw()
def calculate_weight(dataset):
        global weights,data
        weights = [0 for i in range(len(dataset[0]))]
        flag=0
        while 1:
                flag=0
                for t in (data):
                        t=np.array(t)
                        #np.reshape(t,3,1)
                        temp=np.dot(weights,t)
         #               print(temp)
                        if(temp<=0):
                                print("temp", temp, weights)
                                flag=1
                                weights=weights+t
                                #print(weights)
                                break
                plot_line()
                #print(weights)
                # if(flag==0):
                #         break
        #print("Prince\n")  
        #print(weights)
        # plot_graph(train_set)
        return weights

gen_points()
plot_graph(train_set)

plt.xlim((-6, 20))
plt.ylim((-20, 30))
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("Perceptron Learning Algorithm")
#train_set=[ [0,0,1], [0,1,1],[1,0,-1],[1,1,-1]]
data=np.array(train_set)
print(data)
for i in range(0,len(data)):
        if(data[i][2]==-1):
                data[i][0]=-data[i][0]
                data[i][1]=-data[i][1]
weight=calculate_weight(train_set)

