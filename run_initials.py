import Viterbi as vt
import numpy as np

# states = ['hot', 'cold', 'small', 'medium', 'large' ]
# values= ['yes','maybe','no']
# #State Transition Probabilities [[h->h,c->h,s->h,m->h,l->h], [],[],[],[]]
# T = np.matrix([[0.6,0.8,0.,0.,0.],[0.4,0.2,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,1.,0.2,0.2],[0.,0.,0.,0.8,0.8]]) # T[i,j] means transfer TO i FROM j !!
# # State -> Observation "emission" probabilities  [[H->A,L->A],[H->C,L->C],[etc..],[]]
# E = np.matrix([[0.9,0.,0.3,0.4,0.8],[0.1,0.1,0.3,0.3,0.1],[0.,0.9,0.4,0.3,0.1]])
# # E[t,k] is the prob of emitting the t'th observation FROM the k'th state
# observations = ['no','no','no','no','no', 'no','no']

# #DNA example
# states = ['H','L']
# values = ['A','C','G','T']
# T = np.matrix([[0.5,0.4],[0.5,0.6]])
# E = np.matrix([[0.2,0.3],[0.3,0.2],[0.3,0.2],[0.2,0.3]])
# observations = ['G','G','C','A','C','T','G','A','A']


path = vt.Viterbi(states,values,T,E,observations)[1]
print(path)
