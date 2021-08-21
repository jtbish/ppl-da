import copy
import math

from .rule import Rule
from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .indiv import Indiv
from .rng import get_rng
from .util import get_niche

_MIN_TOURN_SIZE = 2


def niched_selection(pop, default_actions):
    """Select parents from niches: niches picked at random to preserve approx.
    uniform distribution of niche representation."""
    niche_map = {da: get_niche(pop, da) for da in default_actions}

    parents = []
    num_to_select = get_hp("pop_size")
    for _ in range(num_to_select):
        default_action = get_rng().choice(default_actions)
        niche = niche_map[default_action]
        parents.append(_tournament_selection(niche))
    return parents


def standard_selection(pop):
    """Standard selection: niches combined into homogeneous pop."""
    parents = []
    num_to_select = get_hp("pop_size")
    for _ in range(num_to_select):
        parents.append(_tournament_selection(pop))
    return parents


def _tournament_selection(indiv_set):
    # tourn size scales according to num indivs in the set being selected in:
    # so whether operating in whole pop or single niche, selection pressure is
    # dynamic
    tourn_size = max(_MIN_TOURN_SIZE,
                     math.ceil(get_hp("tourn_percent") * len(indiv_set)))
    best = _select_random_indiv(indiv_set)
    for _ in range(_MIN_TOURN_SIZE, (tourn_size + 1)):
        indiv = _select_random_indiv(indiv_set)
        if indiv.fitness > best.fitness:
            best = indiv
    return best


def _select_random_indiv(indiv_set):
    idx = get_rng().randint(0, len(indiv_set))
    return indiv_set[idx]


def niched_crossover(parents, default_actions, encoding):
    pop_size = get_hp("pop_size")
    assert len(parents) == pop_size

    offspring = []
    # first split up parents into their niches
    niches = [get_niche(parents, da) for da in default_actions]

    # now for each niche, pair off the parents and attempt crossover.
    # note that the selection order is already randomised so can just pair the
    # parents off in order that they appear, don't need to re-randomise the
    # pairing order
    for niche in niches:
        num_parents_in_niche = len(niche)
        for idx in range(0, num_parents_in_niche, 2):
            parent_a = niche[idx]
            try:
                parent_b = niche[idx + 1]
            except IndexError:
                # odd number of parents, so last parent has no "mate"
                parent_b = None

            if parent_b is not None:
                # can do crossover since there is a pair of parents
                (child_a, child_b) = _crossover(parent_a, parent_b, encoding)
                offspring.append(child_a)
                offspring.append(child_b)
            else:
                # no crossover, just clone the lone parent to produce a single
                # child
                child = copy.deepcopy(parent_a)
                offspring.append(child)

    assert len(offspring) == pop_size
    return offspring


def _crossover(parent_a, parent_b, encoding):
    parent_a = copy.deepcopy(parent_a)
    parent_b = copy.deepcopy(parent_b)
    if get_rng().random() < get_hp("p_cross"):
        (child_a,
         child_b) = _uniform_crossover(parent_a, parent_b,
                                       encoding)
    else:
        (child_a, child_b) = (parent_a, parent_b)
    return (child_a, child_b)


def _uniform_crossover(parent_a, parent_b, encoding):
    """Uniform crossover with swapping acting on individual genes of rules."""
    assert parent_a.default_action == parent_b.default_action
    assert parent_a.selectable_actions == parent_b.selectable_actions
    n = get_hp("indiv_size")

    parent_a_alleles = []
    for rule in parent_a.rules:
        parent_a_alleles.extend(rule.condition.alleles)
        parent_a_alleles.append(rule.action)

    parent_b_alleles = []
    for rule in parent_b.rules:
        parent_b_alleles.extend(rule.condition.alleles)
        parent_b_alleles.append(rule.action)

    # (2 alleles for interval on each dim) + action
    alleles_per_cond = (2 * len(encoding.obs_space))
    alleles_per_rule = (alleles_per_cond + 1)
    total_alleles = (n * alleles_per_rule)
    assert len(parent_a_alleles) == total_alleles
    assert len(parent_b_alleles) == total_alleles

    child_a_alleles = parent_a_alleles
    child_b_alleles = parent_b_alleles
    for idx in range(0, total_alleles):
        if get_rng().random() < get_hp("p_cross_swap"):
            _swap(child_a_alleles, child_b_alleles, idx)

    default_action = parent_a.default_action
    selectable_actions = parent_a.selectable_actions

    def _reassemble_child(alleles):
        rules = []
        cond_start_idxs = [i * alleles_per_rule for i in range(0, n)]
        for cond_start_idx in cond_start_idxs:
            cond_end_idx = (cond_start_idx + alleles_per_cond)
            cond_alleles = alleles[cond_start_idx:cond_end_idx]
            action = alleles[cond_end_idx]
            cond = Condition(cond_alleles, encoding)
            rules.append(Rule(cond, action))
        assert len(rules) == n
        return Indiv(rules, default_action, selectable_actions)

    child_a = _reassemble_child(child_a_alleles)
    child_b = _reassemble_child(child_b_alleles)
    return (child_a, child_b)


def _swap(seq_a, seq_b, idx):
    seq_a[idx], seq_b[idx] = seq_b[idx], seq_a[idx]


def mutate(indiv, encoding):
    """Mutates condition and action of rules contained within indiv by
    resetting them in Rule object."""
    for rule in indiv.rules:
        cond_alleles = rule.condition.alleles
        mut_cond_alleles = encoding.mutate_condition_alleles(cond_alleles)
        mut_cond = Condition(mut_cond_alleles, encoding)
        mut_action = _mutate_action(rule.action, indiv.selectable_actions)
        rule.condition = mut_cond
        rule.action = mut_action


def _mutate_action(action, selectable_actions):
    if get_rng().random() < get_hp("p_mut"):
        other_actions = list(set(selectable_actions) - {action})
        return get_rng().choice(other_actions)
    else:
        return action
