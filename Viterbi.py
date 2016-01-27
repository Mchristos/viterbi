import numpy as np
from functions import *

def Viterbi(states, values, T, E, observations):
    #Convert observations to numbers
    y = []
    for obs in observations:
        for i in range(len(values)):
            if obs == values[i]:
                y.append(i)


    # Uniform 'beleif' distribution
    p = []
    for i in range(len(states)):
        p.append(1./len(states))

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
        #print(V[t])
        #print("This should equal 1: normalized sum = " + str(sum(V[t])) )
        #print(B[t])


    #Backtrack to get most probable path
    index = np.argmax(V[len(observations)-1])
    bestpath = [index] # len(observations)-1 'th element
    for t in range(len(observations)-2 , -1,-1):
        bestpath = [B[t+1][bestpath[0]]] + bestpath
    statepath = []
    for j in bestpath:
        statepath.append(states[j])

    return V , statepath
