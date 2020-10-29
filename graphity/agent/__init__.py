
def add_agent_attr(probability_based=False, allow_callback=False, allow_update=False):
    def deco(cls):
        attrs = {}
        attrs['probability_based'] = probability_based
        attrs['allow_callback'] = allow_callback
        attrs['allow_update'] = allow_update
        for attr in attrs:
            setattr(cls,attr, attrs[attr])
        return cls
    return deco