import json
import matplotlib.pyplot as plt
import numpy as np

vision_data = {}
vision_bz = 80
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
    'swin-large' : 'Swin-Large, 2021',
    'gpt-small' : 'GPT3-Small, 2020',
    'gpt-medium' : 'GPT3-Medium, 2020'
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
language_mem_dir = "text_classification_fp16/v100/results/mem_results_category.json"
with open(language_mem_dir, 'r') as f:
    lines = f.readlines()

for line in lines:
    obj = json.loads(line)
    network, bz, activation, model, workspace = \
        obj['network'], obj['batch_size'], obj['activation'], obj['model_size'], obj['workspace_size']
    if bz != language_bz:
        continue
    language_data[network] = [activation / 1000, model / 1000, workspace / 1000]


swin_data = {}

swin_mem_dir = "Swin-Transformer/v100/results/mem_results_background.json"
with open(swin_mem_dir, 'r') as f:
    lines = f.readlines()

for line in lines:
    obj = json.loads(line)
    network = "swin-large"
    if 'activation' not in obj:
        continue
    bz, activation, model, workspace = \
         obj['batch_size'], obj['activation'], obj['model_size'], obj['workspace']
    if bz != language_bz:
        continue
    swin_data[network] = [activation / 1000, model / 1000, workspace / 1000]

gpt_bz = 6
gpt_data = {
    "gpt-small" : [2.6, 2.1, 0.1],
    "gpt-medium" : [5.5, 6.1, 0.1]
}



labels = list(vision_data.keys()) + list(language_data.keys()) + list(swin_data.keys()) + list(gpt_data.keys())
labels = [model_names[l] for l in labels]
activations = [x[0] for x in vision_data.values()] + \
              [x[0] for x in language_data.values()] + \
              [x[0] for x in swin_data.values()] + \
              [x[0] for x in gpt_data.values()]
model = [x[1] for x in vision_data.values()] + \
        [x[1] for x in language_data.values()]  + \
        [x[1] for x in swin_data.values()] + \
        [x[1] for x in gpt_data.values()]
workspace = [x[2] for x in vision_data.values()] + \
            [x[2] for x in language_data.values()]  + \
            [x[2] for x in swin_data.values()] + \
            [x[2] for x in gpt_data.values()]

print(activations)
print(model)
print(workspace)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 4)
width = 0.5

ax.bar(labels, activations, width, label='Activation', color='#2596be')
ax.bar(labels, model, width, bottom=activations, label='Model-related', color='orange')
ax.bar(labels, workspace, width, bottom=np.array(activations) + np.array(model), label='Workspace', color='green')

label_size = 13
pre_size = 0
plt.text(pre_size - 0.4, 13.2, f"BS = {vision_bz}", fontsize=label_size)
pre_size += len(vision_data)
plt.axvline(x = pre_size - 0.5, color = 'b', linestyle="--", linewidth=2)
plt.text(pre_size +  - 0.4, 13.2, f"BS = {language_bz}", fontsize=label_size)
pre_size += len(language_data) + len(swin_data)
plt.axvline(x = pre_size - 0.5, color = 'b', linestyle="--", linewidth=2)
plt.text(pre_size - 0.4, 13.2, f"BS = {gpt_bz}", fontsize=label_size)


ax.set_yticks([3, 6, 9, 12], fontsize=14)
ax.set_ylabel('Memory (GB)', size = 16)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(0.06, 1.02, 1, 0.2))
plt.tight_layout()
fig.savefig("graphs/background/memory.pdf")


