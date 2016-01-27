import numpy as np
from functions import *

states = ['hot', 'cold', 'small', 'medium', 'large' ]
values= ['yes','maybe','no']
#State Transition Probabilities [[h->h,c->h,s->h,m->h,l->h], [],[],[],[]]
T = np.matrix([[0.6,0.8,0.,0.,0.],[0.4,0.2,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,1.,0.2,0.2],[0.,0.,0.,0.8,0.8]]) # T[i,j] means transfer TO i FROM j !!
# State -> Observation "emission" probabilities  [[H->A,L->A],[H->C,L->C],[etc..],[]]
E = np.matrix([[0.9,0.,0.3,0.4,0.8],[0.1,0.1,0.3,0.3,0.1],[0.,0.9,0.4,0.3,0.1]])
# E[t,k] is the prob of emitting the t'th observation FROM the k'th state
print("T = \n" + str(T))
print("")
print("E = \n" +str(E))
#Data
observations = ['no','no','no','no','no', 'no','no']


#Convert data
y = []
for value in observations:
    if value == 'yes':
        y.append(0)
    elif value == 'maybe':
        y.append(1)
    elif value == 'no':
        y.append(2)
    else:
        break
        print("invalid measurement array. please leave!")

# Uniform 'beleif' distribution
p = [0.2,0.2,0.2,0.2,0.2] #hot,cld,sml,med,lrg

#Initialize beleif dist (given first obs) and 'update-memory' for backtrack
# i.e. for t = 0
V = [[]]
B = [[]]
for i in range(len(states)):
    V[0].append(p[i]*E[y[0],i]) #prior x emission prob
    B[0].append(i)
V[0] = normalize(V[0])
print(V[0])
print(B[0])
#The real shit goes down
for t in range(1,len(observations)): #for each observed state (time step)
    value = y[t]
    V.append([])
    B.append([])
    for i in range(len(states)): #update each element in 'belief' distribution
        transfer = []
        for k in range(len(states)):
            transfer.append( V[t-1][k]*E[value,k]*T[i,k] )
        V[t].append(max(transfer))
        B[t].append(np.argmax(transfer))
    V[t] = normalize(V[t])
    print(V[t])
    # print("normalized sum: " + str(sum(V[t])) )
    print(B[t])
# print(B)
# print(V)

#Backtrack to get most probable path
index = np.argmax(V[len(observations)-1])
bestpath = [index] # len(observations)-1 'th element
for t in range(len(observations)-2 , -1,-1):
    bestpath = [B[t+1][bestpath[0]]] + bestpath
print("best path is " + str(bestpath))
