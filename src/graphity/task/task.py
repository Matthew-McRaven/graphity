# Describes a family of related tasks
# Single task object is not shared between episodes within an epoch.
# Otherwise, replay buffers will overwrite each other.
class Task():
    def __init__(self,  device="cpu"):
        self._device = device
        
    # Let the task control which device it wants to be scheduled on.
    @property
    def device(self):
        return self._device
    @device.setter
    def device(self, value):
        self._device = value