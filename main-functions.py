# work in progress. this file contains only the first prototype

# https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/composites.html#reverse-anneal
# https://cloud.dwavesys.com/leap/learning/

import numpy as np
from dimod import ExactSolver, AdjArrayBQM
from dwave.system import DWaveSampler, EmbeddingComposite
from helpers.schedule import make_reverse_anneal_schedule 

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
    # - gapopsize: population size for the genetic algorithm
    # - gagencount: number if generations to evolve
    # - tgtenergy: a target level of energy to stop the evolution if reached
    # - holdtime: hold time for a reverse anneal
    # - annealtime: anneal time for a forward anneal
    # - mutationct: number of small bqms to be included in one mutation operation
    # - mutationprob: probability that someone who is not elite will mutate
    # - eliteratio: fraction of lowest energy solutions that will carry on to the next generation without being mutated
    # - breedratio: fraction of a generation that is comprised of offsprings of the previous generation
    
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
    energies = []
    variables = []
    for i in range(len(problemdata['bqms'])):
        quantumsol = sampler.sample(problemdata['bqms'][i], num_reads=n, annealing_time=gaparamsdict['annealtime'])
        samples.append(quantumsol.record.sample)
        energies.append(quantumsol.record.energy)
        variables.append(quantumsol.variables)
    
    population = []         # an array of gapopsize numpy arrays, each being one individual solution
    sortedfitness = []      # an array containing the fitness sorted in ascending order (i.e. zero index is the lowest)
    popindices = []         # defined such that sortedfitness[k] is the fitness of indivitual population[popindices[k]]
    initialise(population, popindices, sortedfitness, samples)
    for i in range(gaparamsdict['gagencount']):
        if sortedfitness[0] <= gaparamsdict['tgtenergy']:
            break
        nextpopulation = []
        for j in range(0,eliteendindex):
            # add elites to nextpopulation
            nextpopulation.append(population[popindices[j]])
        for j in range(eliteendindex,parentsindex):
            # add remaining non-elites with appropriate mutations
            nextpopulation.append(mutate(population[popindices[j]], mutationprob, mutationct, bqms, bqmindexdict, variables))
        for j in range(parentsindex,gapopsize):
            # add breeds with appropriate mutations
            nextpopulation.append(mutate(breed(population), mutationprob, mutationct, bqms, bqmindexdict, variables))
        sortfitness(nextpopulation, popindices, sortedfitness)
        population = nextpopulation
    
    return({'lastpopulation': population, 'popindices': popindices, 'sortedfitness': sortedfitness, 'ngen': i)

def mutate(individual, mutationprob, mutationct, bqms, bqmindexdict, variables):
    # reverse anneals mutationct bqms chosen randomly if this individual is drawn to be mutated
    if np.random.randint(10000) > 10000*mutationprob:
        # do nothing (individual was not drawn to be mutated)
    else:
        mutationidx = np.random.randint(len(bqms), size=mutationct)
        for i in mutationidx:
            init_samples = dict(zip(variables[i], samplefromindividual(individual, bqmindexdict)))
            sampleset = sampler_reverse.sample(bqms[i],
                                   anneal_schedules=reverse_schedule,
                                   initial_state=init_samples,
                                   num_reads=1,
                                   reinitialize_state=False)
            writemutation(individual, sampleset.record.sample[0], bqmindexdict)

# Functions to do:
# samplefromindividual (builds a sample array for a bqm given the individual and the bqmindexdict)
# writemutation (writes the results from the reverse anneal onto the individual)
# join (creates one large solution by combining all bqms with random tie breaks for the common points)
# breed (creates one large solution by doing a join of bqms from the lowest energy solutions)
# initialise (creates first population from inital forward anneals)
# sortfitness (returns the fitness array sorted in ascending order)
#
# Internals to do:
# solutiondict (to avoid repeated energy calcs)
# memory saving data structures for solutions

