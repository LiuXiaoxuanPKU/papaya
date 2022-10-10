import json
import matplotlib.pyplot as plt
import numpy as np

vision_data = {}
vision_bz = 64
model_names = {
    'resnet50' : 'ResNet-50, 2015',
    'resnet152' : 'ResNet-152, 2015',
    'resnet101' : 'ResNet-101, 2016',
    'wide_resnet50_2' : 'Wide-ResNet-50, 2016',
    'wide_resnet101_2' : 'Wide-ResNet-101, 2016',
    'bert-large-cased' : 'Bert-Large, 2017',
    'bert-base-cased' : 'Bert-Base, 2017',
    'roberta-base' : 'RoBERTa-Base, 2018',
    'roberta-large' : 'RoBERTa-Large, 2018',
}

vision_mem_dir = "resnet/v100/results/mem_results.json"
with open(vision_mem_dir, 'r') as f:
    lines = f.readlines()

for line in lines:
    obj = json.loads(line)
    network, bz, activation, model, workspace = \
        obj['network'], obj['batch_size'], obj['activation'], obj['model_only'], obj['workspace']
    if bz != vision_bz:
        continue
    vision_data[network] = [activation / 1000, model / 1000, workspace / 1000]


language_data = {}
language_bz = 36
language_mem_dir = "text_classification_fp16/v100/results/mem_results.json"
with open(language_mem_dir, 'r') as f:
    lines = f.readlines()

for line in lines:
    obj = json.loads(line)
    network, bz, activation, model, workspace = \
        obj['network'], obj['batch_size'], obj['activation'], obj['model_size'], obj['workspace_size']
    if bz != language_bz:
        continue
    language_data[network] = [activation / 1000, model / 1000, workspace / 1000]


print(language_data)

labels = list(vision_data.keys()) + list(language_data.keys())
labels = [model_names[l] for l in labels]
activations = [x[0] for x in vision_data.values()] + [x[0] for x in language_data.values()]
model = [x[1] for x in vision_data.values()] + [x[1] for x in language_data.values()]
workspace = [x[2] for x in vision_data.values()] + [x[2] for x in language_data.values()]

print(activations)
print(model)
print(workspace)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 4)
width = 0.5

ax.bar(labels, activations, width, label='Activation')
ax.bar(labels, model, width, bottom=activations, label='Model-related')
ax.bar(labels, workspace, width, bottom=np.array(activations) + np.array(model), label='Workspace')

ax.set_ylabel('Memory (GB)')
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
# ax.set_title('Scores by group and gender')
ax.legend(ncol=3, loc='upper left')
plt.tight_layout()
fig.savefig("graphs/background/memory.pdf")


