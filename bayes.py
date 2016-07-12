import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math 
import functools

class model:

    def __init__(self, 
                 worlds, lexicon,
                 prior_worlds, 
                 semantics, cost,
                 rationality, utility,
                 contexts = [None], 
                 prior_contexts = None):

        self.prior_worlds = prior_worlds     # prior over possible worlds: p(w)
        self.lexicon = lexicon               # set of all utterances in common ground
        self.worlds = worlds                 # set of all worlds able to be expressed
        
        self.semantics = semantics           # world x utterance x context -> {0,1}; 
                                             # True iff the utterance describes the world
        self.rationality = rationality       # rationality parameter lambda. 
                                             # rationality -> infty means chooses optimally 
        self.utility = utility               # function U(l, w, c, u) where l is the P_l(w | u, c)
        self.cost = cost                     # utterance -> num 
                                             # cost to utter the utterance

        self.contexts = contexts             # (optional contexts)
        self.prior_contexts = lambda x : 1/len(contexts) if prior_contexts is None else prior_contexts


    # Literal Listener
    # return p(world | utterance, context) = p(world | [[utterance]]^context = true)
    @functools.lru_cache(maxsize=None)
    def LiteralListener(self, world, context, utterance):

        p_utterance = sum(map(lambda w : self.semantics(w, utterance, context) * self.prior_worlds(world), 
                              (w for w in self.worlds)))

        #If p_utterance = 0, then return 0 - don't divide by 0
        if p_utterance == 0:
            return 0
        
        return self.prior_worlds(world)/p_utterance \
               if self.semantics(world, utterance, context)  \
               else 0

    # Pragmatic Speaker
    # return p(utterance | world) = exp(rationality * (utility - cost(utterance)))
    def makePragSpeaker(self, listener):

        # and the normed distribution p(u | w, V)
        def speaker(utterance, world, context):

            l = lambda w, c, u : listener(w, c, u)
            # the soft-max utility  
            softmax = lambda u, w, c: math.exp(self.rationality * (self.utility(l, w, c, u) - self.cost(u)))

            norm = sum(map(lambda u : softmax(u, world, context), 
                       self.lexicon)) 

            return softmax(utterance, world, context)/norm

        return speaker

    # Pragmatic Listener
    # return p(world, context | utterance) ~ p_s(utterance | world, context) p(world ^ context)
    def makePragListener(self, speaker):

        def listener(world, context, utterance):
            # Just Bayes Rule
            norm = sum(map(lambda x : speaker(utterance, x[0], x[1]) * self.prior_worlds(x[0]), 
                           ((w, c) for w in self.worlds for c in self.contexts)))
            return (speaker(utterance, world, context) * self.prior_worlds(world))/norm
        
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
        if level == 0:
            return self.LiteralListener
        return self.getPragPair(level)[0]

    def getPragSpeaker(self, level):
        return self.getPragPair(level)[1]

if __name__ == "__main__":
    # Test if we can match Frank and Goodman 2012
    test_mod = model(worlds = ["blue circle", "green circle", "green square"], # Objects
                     lexicon = ["blue", "green", "circle", "square"],          # Labels
                     prior_worlds = lambda w : 1/3,                            # Uniform prior (no concern for salience),
                     utility = lambda l, w, c, u : l(w, c, u),
                     semantics = lambda w, u, c : 1 if u in w else 0,          # if the label is in the name, then they match. easy
                     cost = lambda u : 0,                                      # Cost function - currently uniform 0,
                     rationality = 4)                                          # Rationality
    L1 = test_mod.getPragListener(1)

    #sns.barplot(test_mod.worlds, list(map(lambda w : L1(w, None, "circle"), test_mod.worlds)))


    # Test if we can match Lassiter and Goodman 2013
    from scipy.stats import norm 
    precision = 20
    mean = 0.5
    stddev = 0.3

    test_mod = model(worlds = [x/precision for x in range(0, precision + 1)],               # Height of subject
                     lexicon = ["", "tall", "short"],                                       # Labels
                     contexts = [x/precision for x in range(0, precision + 1)],             # Height of subject
                     prior_worlds = lambda w : norm.pdf(w, mean, stddev)/(precision + 1),   # some sort of prior ???
                     utility = lambda l, w, c, u : math.log(l(w, c, u)) if l(w, c, u) != 0 else -10000,
                     semantics = lambda w, u, c : 0 if ((u == "tall" and w < c) or \
                                                        (u == "short" and w > c)) \
                                                    else 1,                                 # 
                     cost = lambda u : 0.6 * len(u.split()),                                # Cost function - currently uniform 0
                     rationality = 4)                                                       # Rationality

    L0 = test_mod.getPragListener(0)
    S1 = test_mod.getPragSpeaker(1)
    L1 = test_mod.getPragListener(1)

    domain = [(x,y) for x in test_mod.worlds for y in test_mod.contexts]
    #joint_dist = list(zip(domain, list(map(lambda x : L1(x[0], x[1], "tall"), 
    #                                       domain))))
    
    # Plot marginals

    p0_world = [sum([L0(world, context, "tall") * test_mod.prior_contexts(context) for context in test_mod.contexts]) for world in test_mod.worlds]
    p0_context = [sum([L0(world, context, "tall") * test_mod.prior_contexts(context) for world in test_mod.worlds]) for context in test_mod.contexts]
    p1_world = [sum([L1(world, context, "tall") for context in test_mod.contexts]) for world in test_mod.worlds]
    p1_context = [sum([L1(world, context, "tall") for world in test_mod.worlds]) for context in test_mod.contexts]

    context = [[L1(world, context, "tall") for world in test_mod.worlds] for context in test_mod.contexts]
    for x in context:
        print(("{:.2f}  " * 10).format(*x))

    input()

    f, ((ax01, ax11, ax21), (ax02, ax12, ax22)) = plt.subplots(2,3, sharey="row")

    sns.barplot(test_mod.worlds, list(map(test_mod.prior_worlds, test_mod.worlds)), ax = ax01)
    sns.barplot(test_mod.contexts, list(map(test_mod.prior_contexts, test_mod.contexts)), ax = ax02)
    sns.barplot(test_mod.worlds, p0_world, ax = ax11)
    sns.barplot(test_mod.contexts, p0_context, ax = ax12)
    sns.barplot(test_mod.worlds, p1_world, ax = ax21)
    sns.barplot(test_mod.contexts, p1_context, ax = ax22)
    sns.plt.show()

    """ # Joint distribution plotting
    xs = []
    ys = []
    for entry in joint_dist:
        (x,y) = entry[0]
        num_coord = int(1000 * entry[1])
        xs += [x for i in range(num_coord)]
        ys += [y for i in range(num_coord)]
    xs = pd.Series(xs, name="height")
    ys = pd.Series(ys, name="threshold")
    sns.jointplot(xs, ys, xlim=(0,1), ylim=(0,1), kind="kde")
    """
