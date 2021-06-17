__all__ = ['det']

"""
Contains all the agents we can use.
"""

def add_agent_attr(policy_based=False, allow_update=False):
    """
    A decorator that automatically adds attributes to a class.
    These attributes are useful for detecting if an agent is machine-learnable.
    
    :param policy_based: Is the agent powered by a policy-base method? If so, the policy must be tracked in replay memory.
    :param allow_update: Can the agent be trained? As of 20210617, no agents are trainable.
    """
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