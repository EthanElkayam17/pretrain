import torch
import sys
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


INFERENCE_DIR = "~/data"
TRAIN_DIR = "~/data"

STATE_DICT_DIR = "state_dicts"
MODEL_CONFIG_DIR = "configs/architecture"

MODEL_CONFIG_NAME, STATE_DICT_FILE_NAME = sys.argv[1], sys.argv[2]

test_dataset = RexailDataset(root=INFERENCE_DIR,
                                        transform=None,
                                        pre_transform=None,
                                        class_decider=partial(RexailDataset.filter_by_min,
                                                              threshold=500),
                                        max_class_size=500,
                                        ratio=90,
                                        complement_ratio=True,
                                        storewise=True)

IMG_FILE_PATH, TARGET = random.choice(test_dataset.samples)
STATE_DICT_PATH = dirjoin(STATE_DICT_DIR,STATE_DICT_FILE_NAME)

model = CFGCNN(cfg_name=MODEL_CONFIG_NAME, cfg_dir=MODEL_CONFIG_DIR)
state_dict = torch.load(STATE_DICT_PATH, weights_only=True, map_location=torch.device('cpu'))

new_state_dict = {}
for key, value in state_dict.items():
    new_key = key[7:] if key.startswith('module.') else key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.half()
model.eval()

classes = test_dataset.classes

transform = default_transform(resize=(146,146),crop_size=(128,128), mean=[0.46422, 0.45539, 0.44493], std=[0.25619, 0.24720, 0.26809])
X = transform(datasets.folder.default_loader(IMG_FILE_PATH))

y = model(X.unsqueeze(0))
p = torch.softmax(y, dim=1)
class_to_prob = {cls: prob for cls, prob in zip(classes, p.squeeze(0).tolist())}
real_class = classes[TARGET]

sorted_results = sorted(class_to_prob.items(), key=itemgetter(1), reverse=True)
print(sorted_results)


img = mpimg.imread(IMG_FILE_PATH)
plt.imshow(img)

plt.axis('off')  

plt.text(10, 10, f'Predicted: {sorted_results[0][0]} with probability: {sorted_results[0][1]}, \n real target: {real_class}', color='white', fontsize=12, ha='left', va='top')

plt.show()
