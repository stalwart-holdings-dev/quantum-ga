# work in progress. this file contains only the first prototype

# https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/composites.html#reverse-anneal
# https://cloud.dwavesys.com/leap/learning/

import numpy
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
    
    auxsampler = DWaveSampler(solver=dict(qpu=True, max_anneal_schedule_points__gte=4))
    max_slope = 1.0/auxsampler.properties["annealing_time_range"][0]
    reverse_schedule = make_reverse_anneal_schedule(s_target=0.45, hold_time=80, ramp_up_slope=max_slope)
    
    sampler = EmbeddingComposite(DWaveSampler())
    
    quantumsols = []
    quantumenergies = []
    for i in range(len(problemdata['bqms'])):
        quantumsol = sampler.sample(problemdata['bqms'][i], num_reads=n, annealing_time=30)
        quantumsols.append(quantumsol.record.sample)
        quantumenergies.append(quantumsol.record.energy)
    #TBD  init_samples = dict(zip(forward_answer.variables, forward_answer.record[i5].sample))
    sampler_reverse = ReverseAdvanceComposite(auxsampler)    
    sampleset = sampler_reverse.sample(bqm,
                                   anneal_schedules=reverse_schedule,
                                   initial_state=init_samples,
                                   num_reads=1,
                                   reinitialize_state=False)

# Functions to do:
# mutate (reverse anneals mutationct bqms chosen randomly)
# join (creates one large solution by combining all bqms with random tie breaks for the common points)
# breed (creates one large solution by doing a join of bqms from the lowest energy solutions)
# Internals to do:
# solutiondict (to avoid repeated energy calcs)
# memory saving data structures for solutions

#print("Lowest energy found:", sampleset.record.energy[0])
        
