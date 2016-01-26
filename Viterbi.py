import numpy as np

states = ['high', 'low']
values= ['A','C', 'G','T']
#State Transition Probabilities [[H->H,L->H], [H->L, L->L]]
T = np.matrix([[0.5,0.4],[0.5,0.6]]) # T[i,j] means transfer TO i FROM j !!
# State -> Observation "emission" probabilities  [[H->A,L->A],[H->C,L->C],[etc..],[]]
E = np.matrix([[0.2,0.3],[0.3,0.2],[0.3,0.2],[0.2,0.3]])
# E[t,k] is the prob of emitting the t'th observation FROM the k'th state

#Data
observations = ['G','G','C','A','C','T','G','A','A']
#Convert data
y = []
for measure in observations:
    if measure == 'A':
        y.append(0)
    elif measure == 'C':
        y.append(1)
    elif measure == 'G':
        y.append(2)
    elif measure == 'T':
        y.append(3)
    else:
        break
        print("invalid measurement array. please leave")

# Uniform 'beleif' distribution
p = [0.5,0.5]

#Initialize beleif dist (given first obs) and 'update-memory' for backtrack
V = [[]]
B = [[]]
for i in range(len(states)):
    V[0].append(np.log(p[i]) + np.log(E[y[0],i]))
    B[0].append(i)

#The real shit goes down
for t in range(1,len(observations)): #for each observed state (time step)
    measure = y[t]
    V.append([])
    B.append([])
    for i in range(len(states)): #update each element in 'belief' distribution
        transfer = []
        for k in range(len(states)):
            transfer.append( V[t-1][k] +np.log(E[measure,k]) + np.log(T[i,k]) )
        V[t].append(max(transfer))
        B[t].append(np.argmax(transfer))
print(B)
print(V)

#Backtrack to get most probable path
index = np.argmax(V[len(observations)])
bestpath = [index]
# for t in range()
