import torch
import sys
import onnxruntime as ort
from utils.other import dirjoin
from models.model import CFGCNN
from torchvision import datasets
from utils.transforms import default_transform
from data.data import RexailDataset
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functools import partial
import random
from typing import Dict, Any
import json
from pathlib import Path

def dict_to_json(data: Dict[str, Any], *, indent: int = 2, sort_keys: bool = False) -> str:
    path = Path("idx_to_skl.json")
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys, ensure_ascii=True)

INFERENCE_DIR = "~/newdata"
TRAIN_DIR = "~/newdata"

STATE_DICT_DIR = "state_dicts"
MODEL_CONFIG_DIR = "configs/architecture"

MODEL_CONFIG_NAME, STATE_DICT_FILE_NAME = sys.argv[1], sys.argv[2]

train_dataset = RexailDataset(root=TRAIN_DIR,
                            transform=None,
                            pre_transform=None,
                            class_decider=partial(RexailDataset.filter_by_min,
                                                    threshold=750),
                            max_class_size=750,
                            ratio=90,
                            complement_ratio=False,
                            storewise=False)

test_dataset = RexailDataset(root=TRAIN_DIR,
                            transform=None,
                            pre_transform=None,
                            class_decider=partial(RexailDataset.filter_by_min,
                                                    threshold=750),
                            max_class_size=750,
                            ratio=90,
                            complement_ratio=True,
                            storewise=False)

c = test_dataset.classes
d = {i:c[i] for i in range(len(c))}
dict_to_json(d)

"""IMG_FILE_PATH, TARGET = random.choice(test_dataset.samples)
print(IMG_FILE_PATH)"""

IMG_FILE_PATH, TARGET = '/Users/ethanelkayam/downloads/1743331342026_1_835.jpg', 0
if any(t[0] == IMG_FILE_PATH for t in test_dataset.samples):
    print('YES, testing')
elif any(t[0] == IMG_FILE_PATH for t in train_dataset.samples):
    print('YES, training')
else:
    print('NO')

STATE_DICT_PATH = dirjoin(STATE_DICT_DIR,STATE_DICT_FILE_NAME)

model = CFGCNN(cfg_name=MODEL_CONFIG_NAME, cfg_dir=MODEL_CONFIG_DIR)
state_dict = torch.load(STATE_DICT_PATH, weights_only=True, map_location=torch.device('cpu'))

new_state_dict = {}
for key, value in state_dict.items():
    new_key = key[7:] if key.startswith('module.') else key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.eval()

classes = test_dataset.classes

transform = default_transform(resize=(256,256),crop_size=(224,224), mean=[0.46422, 0.45539, 0.44493], std=[0.25619, 0.24720, 0.26809], dtype=torch.float)
X = transform(datasets.folder.default_loader(IMG_FILE_PATH))

y = model(X.unsqueeze(0))
p = torch.softmax(y, dim=1)
class_to_prob = {cls: prob for cls, prob in zip(classes, p.squeeze(0).tolist())}
real_class = classes[TARGET]

session = ort.InferenceSession("RV1-1-2_4.onnx", providers=['CPUExecutionProvider'])
print(session.get_inputs()[0].shape)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

outputs = session.run([output_name], {input_name: (X.unsqueeze(0)).numpy()})
result = outputs[0]
print(torch.argmax(y))
print(y.shape)
print(result)
print(y)

sorted_results = sorted(class_to_prob.items(), key=itemgetter(1), reverse=True)
print(sorted_results)


img = mpimg.imread(IMG_FILE_PATH)

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("Prediction:")

plt.imshow(img)

plt.axis('off')  

plt.text(10, 10, f'Predicted: {sorted_results[0][0]} with probability: {sorted_results[0][1]}, \n real target: {real_class}', color='white', fontsize=12, ha='left', va='top')

plt.show()
