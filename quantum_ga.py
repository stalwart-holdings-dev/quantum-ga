# Quantum Assisted Genetic Algoritm.
import numpy as np
from copy import deepcopy
from dimod import ExactSolver, AdjArrayBQM
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system.composites import ReverseAdvanceComposite

solutiondict = {}

def SolveLargeBQM(largebqm, smallbqms, bqmindexdict, gaparamsdict):
    # largebqm is an AdjArrayBQM object representing the bqm to be solved
    # smallbqms is a dictionary where each element is an AdjArrayBQM object
    # containing a partition of largebqm
    # bqmindexdict contains a mapping between the index of a variable in
    # the partition bqm and the index of this variable in the unpartitioned 
    # problem, i.e. if
    # bqmindexdict[j][i] = k
    # then the k-th variable in the j-th partitioned BQM 
    # corresponds to the i-th variable in the global BQM
    # gaparamsdict is a dictionary with the following entries:
    # - nvars: number of variables of the large bqm
    # - gapopsize: population size for the genetic algorithm
    # - gagencount: number if generations to evolve
    # - tgtenergy: a target level of energy to stop the evolution if reached
    # - forwardmutate: True if the mutation operator is a forward anneal, False if the mutation operator is a reverse anneal
    # - initialreads: number of anneals for the problem initialisation
    # - mutationreads: number of anneals for the mutation operator
    # - mutationtime: qpu time for a mutation (hold time for a reverse anneal or anneal time for a forward anneal)
    # - annealtime: anneal time for a forward anneal
    # - reinitstate: boolean indicating if states should be reinitialised during a reverse anneal
    # - mutationct: number of small bqms to be included in one mutation operation
    # - mutationprob: probability that someone who is not elite will mutate
    # - eliteratio: fraction of lowest energy solutions that will carry on to the next generation without being mutated
    # - breedratio: fraction of a generation that is comprised of offsprings of the previous generation
    # - randomtiebreak: True if intersecting variables between bqms should be chosen randomly when joining. can speed up the convergence of 
    #                really large problems but makes each iteration slightly more expensive. should not be used when the bqms are completely
    #                disjoint as there will be no benefit whatsoever but there will still be a performance cost from the random shuffle.
    
    nvars = gaparamsdict['nvars']
    tiebreak = gaparamsdict['randomtiebreak']
    mutationct = gaparamsdict['mutationct']
    mutationprob = gaparamsdict['mutationprob']
    initialreads = gaparamsdict['initialreads']
    
    global sampler_reverse
    global reverse_schedule
    global forwardmutate
    global mutationreads
    global mutationtime
    global reinitstate
    global annealtime
    global gapopsize
    global sampler

    sampler = EmbeddingComposite(DWaveSampler())
    gapopsize = gaparamsdict['gapopsize']
    annealtime = gaparamsdict['annealtime']
    reinitstate = gaparamsdict['reinitstate']
    mutationtime = gaparamsdict['mutationtime']
    forwardmutate = gaparamsdict['forwardmutate']
    mutationreads = gaparamsdict['mutationreads']
    eliteendindex = int(gapopsize*gaparamsdict['eliteratio'])
    parentsindex = int(gapopsize*(1-gaparamsdict['breedratio']))
    revsampler = DWaveSampler(solver=dict(qpu=True, max_anneal_schedule_points__gte=4))
    sampler_reverse = EmbeddingComposite(revsampler)
    reverse_schedule = [[0,1],[3,0.45],[mutationtime+3,0.45],[mutationtime+6,1]]
    
    samples = []
    variables = []
    for i in range(len(smallbqms)):
        quantumsol = sampler.sample(smallbqms[i], num_reads=initialreads, annealing_time=annealtime)
        sortedrecord = quantumsol.record
        sortedrecord.sort(order='energy')
        samples.append(sortedrecord.sample)
        variables.append(quantumsol.variables)
        
    
    population = []         # an array of gapopsize numpy arrays, each being one individual solution
    sortedfitness = []      # a list of two-element tuples containing in the first element of the tuple
                            # the fitness sorted in ascending order (i.e. zero index is the lowest) and in the
                            # second element of the tuple the index in the list population that this fitness
                            # value refers to.
    initialise(population, sortedfitness, samples, tiebreak, nvars, largebqm, bqmindexdict, gapopsize)
    for i in range(gaparamsdict['gagencount']):
        print ('Current Lowest Energy = ' + str(sortedfitness[0][0]))
        if sortedfitness[0][0] <= gaparamsdict['tgtenergy']:
            break
        nextpopulation = []
        nextfitness = []
        mutationsamples = []
        revcounter = []
        fillmutationsamples(mutationsamples, revcounter, population[sortedfitness[0][1]], smallbqms, bqmindexdict, variables)

        for j in range(0,eliteendindex):
            # add non-mutated elites to nextpopulation
            newindividual = deepcopy(population[sortedfitness[j][1]])
            nextpopulation.append(newindividual)
            addtofitnesslist(nextfitness, largebqm, newindividual)
        for j in range(eliteendindex, 2*eliteendindex):
            # add mutated elites to nextpopulation
            newindividual = deepcopy(population[sortedfitness[j][1]])
            mutate(newindividual, 1, mutationct, bqmindexdict, mutationsamples, revcounter)
            nextpopulation.append(newindividual)
            addtofitnesslist(nextfitness, largebqm, newindividual)
        for j in range(2*eliteendindex,parentsindex):
            # add remaining non-elites with probabilistic mutations
            newindividual = deepcopy(population[sortedfitness[j-eliteendindex][1]])
            mutate(newindividual, mutationprob, mutationct, bqmindexdict, mutationsamples, revcounter)
            nextpopulation.append(newindividual)
            addtofitnesslist(nextfitness, largebqm, newindividual)
        for j in range(len(revcounter)):
            revcounter[j] = 0
        for j in range(parentsindex,gapopsize):
            # add breeds with appropriate mutations
            offspring = breed(population, bqmindexdict, tiebreak)
            mutate(offspring, mutationprob, mutationct, bqmindexdict, mutationsamples, revcounter)
            nextpopulation.append(offspring)
            addtofitnesslist(nextfitness, largebqm, offspring)
        nextfitness.sort(key=lambda x: x[0])
        population = nextpopulation
        sortedfitness = nextfitness
    
    return({'lastpopulation': population, 'sortedfitness': sortedfitness, 'ngen': i})

def fillmutationsamples(mutationsamples, revcounter, individual, bqms, bqmindexdict, variables):
    for i in range(len(bqms)):
        if forwardmutate:
            sampleset = sampler.sample(bqms[i], num_reads=mutationreads, annealing_time=annealtime)
        else:
            i_sample = samplefromindividual(individual, bqmindexdict[i])
            init_samples = dict(zip(variables[i], i_sample))
            sampleset = sampler_reverse.sample(bqms[i],
                                anneal_schedule=reverse_schedule,
                                initial_state=init_samples,
                                num_reads=mutationreads,
                                reinitialize_state=reinitstate)
        sortedrecord = sampleset.record
        sortedrecord.sort(order='energy')
        mutationsamples.append(sortedrecord.sample)
        revcounter.append(0)

def mutate(individual, mutationprob, mutationct, bqmindexdict, mutationsamples, revcounter):
    #if np.random.randint(10000) >= 10000*mutationprob:
        # do nothing (individual was not drawn to be mutated)
        # the two comment lines above were added just for clarity
    if np.random.randint(10000) < 10000*mutationprob:
        mutationidx = np.random.randint(len(bqmindexdict), size=mutationct)
        for i in mutationidx:
            writemutation(individual, mutationsamples[i][revcounter[i]], bqmindexdict[i])
            revcounter[i] = revcounter[i]+1

def samplefromindividual(individual, bqmindexdict):
    # builds a sample array for a bqm given the individual and the bqmindexdict
    # bqmindexdict contains a mapping between the index of a variable in
    # the partition bqm and the index of this variable in the unpartitioned 
    # problem, i.e. if
    # bqmindexdict[j][i] = k
    # then the k-th variable in the j-th partitioned BQM 
    # corresponds to the i-th variable in the global BQM
    sample = np.zeros(shape=len(bqmindexdict),dtype=np.int8)
    for key in bqmindexdict:
        sample[bqmindexdict[key]] = individual[key]
    return(sample)

def writemutation(individual, reverseresults, bqmindexdict):
    # writes the results from the reverse anneal onto the individual
    for key in bqmindexdict:
        individual[key] = reverseresults[bqmindexdict[key]]

def breed(population, bqmindexdict, randomtiebreak):
    # creates one large solution by doing a join of bqms from the population
    popsize = len(population)
    breedorder = createjoinorder(bqmindexdict, randomtiebreak)
    offspring = np.zeros(shape=len(population[0]),dtype=np.int8)
    for i in breedorder:
        parent = np.random.randint(popsize)
        for key in bqmindexdict[i]:
            offspring[key] = population[parent][key]
    return(offspring)

def join(samples, joinindex, bqmindexdict, randomtiebreak, nvars):
    # creates one large solution by combining the solutions of individual bqms given in samples
    # each item of samples should be a record.sample list of numpy arrays returned by the BQM solver
    # joinindex specifies which sample should used for the join
    joinresult = np.zeros(shape=nvars,dtype=np.int8)
    joinorder = createjoinorder(bqmindexdict, randomtiebreak)
    for i in joinorder:
        for key in bqmindexdict[i]:
            joinresult[key] = samples[i][joinindex][bqmindexdict[i][key]]
    return(joinresult)

def createjoinorder(bqmindexdict, randomtiebreak):
    if randomtiebreak == True:
        # shuffles the small bqms to ensure a diversity of scenarios over generations for the variables that 
        # are part of the intersection between bqms
        bqmindices = list(range(len(bqmindexdict)))
        joinorder = []
        startinglen = len(bqmindices)
        for i in range(startinglen):
            itemtopop = np.random.randint(len(bqmindices))
            joinorder.append(bqmindices.pop(itemtopop))
        return(joinorder)
    else:
        return(list(range(len(bqmindexdict))))
    
def initialise(population, sortedfitness, samples, randomtiebreak, nvars, largebqm, bqmindexdict, gapopsize):
    # creates first population from inital forward anneals getting only the top gapopsize reads
    for i in range(gapopsize):
        population.append(join(samples, i, bqmindexdict, randomtiebreak, nvars))
        addtofitnesslist(sortedfitness, largebqm, population[i])
    sortedfitness.sort(key=lambda x: x[0])

def addtofitnesslist(fitnesslist, largebqm, individual):
    index = len(fitnesslist)
    fitnesstuple = (calcenergy(largebqm,individual),index)
    fitnesslist.append(fitnesstuple)

def calcenergy(bqm, individual):
    # stores in memory the calculation of the energy into a dictionary to avoid repeated recalculations
    global solutiondict
    individualstring = np.array_str(individual)
    if individualstring in solutiondict:
        return(solutiondict[individualstring])
    else:
        energy = bqm.energy(individual,dtype=np.int8)
        solutiondict[individualstring] = energy
        return(energy)

