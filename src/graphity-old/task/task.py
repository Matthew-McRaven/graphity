# Describes a family of related tasks
# Single task object is not shared between episodes within an epoch.
# Otherwise, replay buffers will overwrite each other.
class Task():
    class Definition:
        def __init__(self, ctor, **kwargs):
            self.task_ctor = ctor
            self.task_kwargs = kwargs
        def instance(self):
            return self.task_ctor(**self.task_kwargs)
    def __init__(self,  device="cpu", batch_size=10):
        self._device = device
        self._batch_size = batch_size
        
    # Let the task control which device it wants to be scheduled on.
    @property
    def device(self):
        return self._device
    @device.setter
    def device(self, value):
        self._device = value

    # Let the task control the sampled batch size
    @property
    def batch_size(self):
        return self._batch_size
    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value