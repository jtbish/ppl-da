def get_niche(indiv_set, default_action):
    return [
        indiv for indiv in indiv_set if indiv.default_action == default_action
    ]
