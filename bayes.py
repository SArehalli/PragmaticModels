import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import math 

class model:

    def __init__(self, 
                 worlds, lexicon,
                 prior_worlds, 
                 semantics, cost,
                 rationality):

        self.prior_worlds = prior_worlds     # prior over possible worlds: p(w)
        self.lexicon = lexicon               # set of all utterances in common ground
        self.worlds = worlds                 # set of all worlds able to be expressed
        self.semantics = semantics           # world x utterance -> {0,1}; 
                                             # True iff the utterance describes the world
        self.rationality = rationality       # rationality parameter lambda. 
                                             # rationality -> infty means chooses optimally 
        self.cost = cost                     # utterance -> num 
                                             # cost to utter the utterance
    # Literal Listener
    # return p(world | utterance) = p(world | [[utterance]] = true)
    def LiteralListener(self, world, utterance):

        # = p(world ^ [[utterance]] = true)/p([[utterance]] == true)
        # p([[utterance]] == true) = sum_worlds p(world)P([[utterance]] = true | world)
        p_utterance = sum(map(lambda w : self.semantics(w, utterance) * self.prior_worlds(world), 
                              self.worlds))
        
        return self.prior_worlds(world)/p_utterance \
               if self.semantics(world, utterance)  \
               else 0

    # Pragmatic Speaker
    # return p(utterance | world) = exp(rationality * (utility - cost(utterance)))
    def makePragSpeaker(self, listener):

        # utility(u | world) ~ p_l
        utility = lambda w, u : listener(w, u)

        # the soft-max utility  
        softmax = lambda u, w: math.exp(self.rationality * (utility(w, u) - self.cost(u)))

        # and the normed distribution p(u | w)
        def speaker(utterance, world):
            norm = sum(map(lambda u : softmax(u, world), 
                           self.lexicon)) 
            return softmax(utterance, world)/norm

        return speaker

    # Pragmatic Listener
    # return p(world | utterance) ~ p_s(utterance | world) p(world)
    def makePragListener(self, speaker):

        def listener(world, utterance):
            # Just Bayes Rule
            norm = sum(map(lambda w : speaker(utterance, w) * self.prior_worlds(w), 
                           self.worlds))
            return (speaker(utterance, world) * self.prior_worlds(world))/norm
        
        return listener

    def getPragPair(self, level):
        listeners = []
        speakers = []
        listeners.append(self.LiteralListener)
        for i in range(level):
            speakers.append(self.makePragSpeaker(listeners[-1]))
            listeners.append(self.makePragListener(speakers[-1]))
        return listeners[-1], speakers[-1]

    def getPragListener(self, level):
        return self.getPragPair(level)[0]

    def getPragSpeaker(self, level):
        return self.getPragPair(level)[1]

if __name__ == "__main__":
    # Test if we can match Frank and Goodman 2013
    test_mod = model(worlds = ["blue circle", "green circle", "green square"], # Objects
                     lexicon = ["blue", "green", "circle", "square"],          # Labels
                     prior_worlds = lambda w : 1/3,                             # Uniform prior (no concern for salience,
                     semantics = lambda w, u : 1 if u in w else 0,             # if the label is in the name, then they match. easy
                     cost = lambda u : 0,                                       # Cost function - currently uniform 0,
                     rationality = 4)                                          # Rationality
    L1 = test_mod.getPragListener(1)

    sns.barplot(test_mod.worlds, list(map(lambda w : L1(w, "circle"), test_mod.worlds)))
    sns.plt.show()

