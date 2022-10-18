## GPU Utilization Profiling Usage:
``` python
from utilization import gpuUtilization, UtilizationContext, UtilizationTrainContext

...
with UtilizationContext(args.get_util) as context:

    for epoch in epoch:
        for sample in samples:
            with UtilizationTrainContext(context):
                ...
                # this is a pass
                model.train()
                forward()
                backward()
                ...
```
Recording result:
```python
exp_recorder.record("utilization", context.getAvg())
```
Also, need to increase iteration number for get speed a little bit (5->~18)
