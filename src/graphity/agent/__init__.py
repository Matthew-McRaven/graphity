__all__ = ['det', 'pg']
def add_agent_attr(policy_based=False, allow_update=False):
    def deco(cls):
        attrs = {}
        attrs['policy_based'] = policy_based
        attrs['allow_update'] = allow_update
        # By default, a model should not be recurrent.
        attrs['recurrent'] = lambda x: False
        for attr in attrs:
            setattr(cls,attr, attrs[attr])
        return cls
    return deco