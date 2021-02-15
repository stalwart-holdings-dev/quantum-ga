# work in progress. this file contains only the first prototype
# more documentation can be found here:
# https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/composites.html#reverse-anneal
# https://cloud.dwavesys.com/leap/learning/

import numpy as np
from dimod import ExactSolver, AdjArrayBQM
from dwave.system import DWaveSampler, EmbeddingComposite
from helpers.schedule import make_reverse_anneal_schedule 

global solutiondict = {}

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
    # - holdtime: hold time for a reverse anneal
    # - annealtime: anneal time for a forward anneal
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
    gapopsize = gaparamsdict['gapopsize']
    mutationprob = gaparamsdict['mutationprob']
    eliteendindex = int(gapopsize*gaparamsdict['eliteratio'])
    parentsindex = int(gapopsize*(1-gaparamsdict['breedratio']))
    revsampler = DWaveSampler(solver=dict(qpu=True, max_anneal_schedule_points__gte=4))
    max_slope = 1.0/auxsampler.properties['annealing_time_range'][0]
    
    global sampler_reverse
    global reverse_schedule
    sampler = EmbeddingComposite(DWaveSampler())
    sampler_reverse = ReverseAdvanceComposite(revsampler)    
    reverse_schedule = make_reverse_anneal_schedule(s_target=0.45, hold_time=gaparamsdict['holdtime'], ramp_up_slope=max_slope)
    
    samples = []
    variables = []
    for i in range(len(problemdata['bqms'])):
        quantumsol = sampler.sample(problemdata['bqms'][i], num_reads=gapopsize, annealing_time=gaparamsdict['annealtime'])
        samples.append(quantumsol.record.sample)
        variables.append(quantumsol.variables)
    
    population = []         # an array of gapopsize numpy arrays, each being one individual solution
    sortedfitness = []      # a list of two-element tuples containing in the first element of the tuple
                            # the fitness sorted in ascending order (i.e. zero index is the lowest) and in the
                            # second element of the tuple the index in the list population that this fitness
                            # value refers to.
    initialise(population, sortedfitness, samples, tiebreak, nvars, largebqm, bqmindexdict)
    for i in range(gaparamsdict['gagencount']):
        if sortedfitness[0][0] <= gaparamsdict['tgtenergy']:
            break
        nextpopulation = []
        nextfitness = []
        for j in range(0,eliteendindex):
            # add elites to nextpopulation
            nextpopulation.append(population[sortedfitness[j][1]])
            addtofitnesslist(nextfitness, largebqm, population[j])
        for j in range(eliteendindex,parentsindex):
            # add remaining non-elites with appropriate mutations
            nextpopulation.append(mutate(population[sortedfitness[j][1]], mutationprob, mutationct, bqms, bqmindexdict, variables))
            addtofitnesslist(nextfitness, largebqm, population[j])
        for j in range(parentsindex,gapopsize):
            # add breeds with appropriate mutations
            offspring = breed(population, bqmindexdict, tiebreak)
            nextpopulation.append(mutate(offspring, mutationprob, mutationct, bqms, bqmindexdict, variables))
            addtofitnesslist(nextfitness, largebqm, population[j])
        nextfitness.sort(key=lambda x: x[0])
        population = nextpopulation
        sortedfitness = nextfitness
    
    return({'lastpopulation': population, 'popindices': popindices, 'sortedfitness': sortedfitness, 'ngen': i)

def mutate(individual, mutationprob, mutationct, bqms, bqmindexdict, variables):
    # reverse anneals mutationct bqms chosen randomly if this individual is drawn to be mutated
    if np.random.randint(10000) > 10000*mutationprob:
        # do nothing (individual was not drawn to be mutated)
    else:
        mutationidx = np.random.randint(len(bqms), size=mutationct)
        for i in mutationidx:
            init_samples = dict(zip(variables[i], samplefromindividual(individual, bqmindexdict[j])))
            sampleset = sampler_reverse.sample(bqms[i],
                                   anneal_schedules=reverse_schedule,
                                   initial_state=init_samples,
                                   num_reads=1,
                                   reinitialize_state=False)
            writemutation(individual, sampleset.record.sample[0], bqmindexdict)

def samplefromindividual(individual, bqmindexdict):
    # builds a sample array for a bqm given the individual and the bqmindexdict
    # bqmindexdict contains a mapping between the index of a variable in
    # the partition bqm and the index of this variable in the unpartitioned 
    # problem, i.e. if
    # bqmindexdict[j][i] = k
    # then the k-th variable in the j-th partitioned BQM 
    # corresponds to the i-th variable in the global BQM
    sample = numpy.zeros(shape=len(bqmindexdict),dtype=int8)
    for key in bqmindexdict:
        sample[key] = individual[bqmindexdict[key]]
    return(sample)

def writemutation(individual, reverseresults, bqmindexdict):
    # writes the results from the reverse anneal onto the individual
    for key in bqmindexdict:
        individual[bqmindexdict[key]] = reverseresults[key]

def breed(population, bqmindexdict, randomtiebreak):
    # creates one large solution by doing a join of bqms from the population
    popsize = len(population)
    breedorder = createjoinorder(bqmindexdict, randomtiebreak)
    offspring = numpy.zeros(shape=len(population[0]),dtype=int8)
    for i in breedorder:
        parent = np.randint(popsize)
        for key in bqmindexdict[i]:
            offspring[key] = population[parent][key]
    return(offspring)

def join(samples, joinindex, bqmindexdict, randomtiebreak, nvars):
    # creates one large solution by combining the solutions of individual bqms given in samples
    # each item of samples should be a record.sample list of numpy arrays returned by the BQM solver
    # joinindex specifies which sample should used for the join
    joinresult = numpy.zeros(shape=nvars,dtype=int8)
    joinorder = createjoinorder(bqmindexdict, randomtiebreak)
    for i in joinorder:
        for key in bqmindexdict[i]:
            joinresult[key] = samples[i][joinindex][bqmindexdict[i][q]]
    return(joinresult)

def createjoinorder(bqmindexdict, randomtiebreak):
    if randomtiebreak == True:
        # shuffles the small bqms to ensure a diversity of scenarios over generations for the variables that 
        # are part of the intersection between bqms
        bqmindices = list(range(len(bqmindexdict)))
        joinorder = []
        startinglen = len(bqmindices)
        for i in range(startinglen):
            itemtopop = np.randint(len(bqmindices))
            joinorder.append(bqmindices.pop(itemtopop))
        return(joinorder)
    else:
        return(list(range(len(bqmindexdict))))
    
def initialise(population, sortedfitness, samples, tiebreak, nvars, largebqm, bqmindexdict):
    # creates first population from inital forward anneals
    initorder = createjoinorder(bqmindexdict, randomtiebreak)
    for i in range(samples):
        population.append(join(samples, i, bqmindexdict, randomtiebreak, nvars))
        addtofitnesslist(sortedfitness, largebqm, population[i])
    sortedfitness.sort(key=lambda x: x[0])

def addtofitnesslist(fitnesslist, largebqm, individual):
    index = len(fitnesslist)
    fitnesstuple = (calclargeenergy(largebqm,individual),index)
    fitnesslist.append(fitnesstuple)

def calclargeenergy(largebqm, individual):
    # stores in memory the calculation of the energy into a dictionary to avoid repeated recalculations
    if individual in solutiondict:
        return(solutiondict[individual])
    else:
        energy = largebqm.energy(individual,dtype=int8)
        solutiondict[individual] = energy
        return(energy)
