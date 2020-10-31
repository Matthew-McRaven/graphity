
def add_agent_attr(policy_based=False, allow_callback=False, allow_update=False):
    def deco(cls):
        attrs = {}
        attrs['policy_based'] = policy_based
        attrs['allow_callback'] = allow_callback
        attrs['allow_update'] = allow_update
        for attr in attrs:
            setattr(cls,attr, attrs[attr])
        return cls
    return deco