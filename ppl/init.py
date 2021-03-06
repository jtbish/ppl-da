from .rule import Rule
from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .indiv import Indiv
from .rng import get_rng


def init_pop(encoding, default_actions):
    return [
        _init_indiv(encoding, default_actions)
        for _ in range(get_hp("pop_size"))
    ]


def _init_indiv(encoding, default_actions):
    num_rules = get_hp("indiv_size")
    default_action = get_rng().choice(default_actions)
    selectable_actions = tuple(set(default_actions) - {default_action})
    assert len(selectable_actions) >= 1
    rules = [
        _init_rule(encoding, selectable_actions) for _ in range(num_rules)
    ]
    return Indiv(rules, default_action, selectable_actions)


def _init_rule(encoding, selectable_actions):
    condition = _init_rule_condition(encoding)
    action = _init_rule_action(selectable_actions)
    return Rule(condition, action)


def _init_rule_condition(encoding):
    return Condition(alleles=encoding.init_condition_alleles(),
                     encoding=encoding)


def _init_rule_action(selectable_actions):
    return get_rng().choice(selectable_actions)
