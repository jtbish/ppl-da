from .error import UnsetPropertyError
from .inference import infer_action


class Indiv:
    def __init__(self, rules, default_action, selectable_actions):
        self._rules = list(rules)
        self._default_action = default_action
        self._selectable_actions = selectable_actions
        self._fitness = None
        self._time_steps_used = None

    @property
    def rules(self):
        return self._rules

    @property
    def default_action(self):
        return self._default_action

    @property
    def selectable_actions(self):
        return self._selectable_actions

    @property
    def fitness(self):
        if self._fitness is None:
            raise UnsetPropertyError
        else:
            return self._fitness

    @fitness.setter
    def fitness(self, val):
        self._fitness = val

    @property
    def time_steps_used(self):
        """Time steps used on *most recent* fitness assessment."""
        if self._time_steps_used is None:
            raise UnsetPropertyError
        else:
            return self._time_steps_used

    @time_steps_used.setter
    def time_steps_used(self, val):
        self._time_steps_used = val

    def select_action(self, obs):
        """Performs inference on obs using rules + default action to predict an action;
        i.e. making Indiv act as a policy."""
        return infer_action(self, obs)
