import logging
import os
from multiprocessing import Pool

from rlenvs.environment import assess_perf

from .ga import mutate, niched_crossover, niched_selection, standard_selection
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .init import init_pop
from .rng import seed_rng

_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])


class PPL:
    def __init__(self, env, encoding, hyperparams_dict):
        self._env = env
        self._encoding = encoding
        self._default_actions = self._env.action_space
        register_hyperparams(hyperparams_dict)
        seed_rng(get_hp("seed"))
        self._pop = None
        self._gen_num = 0

    @property
    def pop(self):
        return self._pop

    def init(self):
        self._pop = init_pop(self._encoding, self._default_actions)
        self._eval_pop_fitness(self._pop)
        return self._pop

    def run_gen(self):
        self._gen_num += 1

        use_niched_selection = \
            (self._gen_num < get_hp("num_niched_select_gens"))
        if use_niched_selection:
            logging.info(f"Using niched selection for gen {self._gen_num}")
            parents = niched_selection(self._pop, self._default_actions)
        else:
            logging.info(f"Using standard selection for gen {self._gen_num}")
            parents = standard_selection(self._pop)

        offspring = niched_crossover(parents, self._default_actions,
                                     self._encoding)
        for child in offspring:
            mutate(child, self._encoding)

        new_pop = offspring
        self._eval_pop_fitness(new_pop)
        self._pop = new_pop
        return self._pop

    def _eval_pop_fitness(self, pop):
        # process parallelism for fitness eval
        num_rollouts = get_hp("num_rollouts")
        gamma = get_hp("gamma")
        with Pool(_NUM_CPUS) as pool:
            results = pool.starmap(self._eval_indiv_fitness,
                                   [(indiv, num_rollouts, gamma)
                                    for indiv in pop])
        for (indiv, result) in zip(pop, results):
            indiv.fitness = result.perf
            indiv.time_steps_used = result.time_steps_used

    def _eval_indiv_fitness(self, indiv, num_rollouts, gamma):
        return assess_perf(self._env, indiv, num_rollouts, gamma)
