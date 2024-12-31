import torch
import sys
from utils.other import dirjoin
from models.model import CFGCNN
from torchvision import datasets
from utils.transforms import default_transform
from utils.data import RexailDataset
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


INFERENCE_DIR = ""
TRAIN_DIR = ""

STATE_DICT_DIR = "engine/state_dicts"
MODEL_CONFIG_DIR = "configs/architecture"

MODEL_CONFIG_NAME, STATE_DICT_FILE_NAME, IMG_FILE_NAME = sys.argv[1], sys.argv[2], sys.argv[3]
IMG_FILE_PATH = dirjoin(INFERENCE_DIR,IMG_FILE_NAME)
STATE_DICT_PATH = dirjoin(STATE_DICT_DIR,STATE_DICT_FILE_NAME)

model = CFGCNN(cfg_name=MODEL_CONFIG_NAME, cfg_dir=MODEL_CONFIG_DIR)
model.load_state_dict(torch.load(STATE_DICT_PATH ,weights_only=True))
classes, _ = RexailDataset.find_classes(TRAIN_DIR)

transform = default_transform()
X = transform(datasets.folder.default_loader(IMG_FILE_PATH))
X

y = model(X)
p = torch.nn.Softmax(y,dim=1)
class_to_prob = {cls: prob for cls, prob in zip(classes, p.squeeze(0).tolist())}

sorted_results = sorted(class_to_prob.items(), key=itemgetter(1), reverse=True)
print(sorted_results)


img = mpimg.imread(IMG_FILE_PATH)
plt.imshow(img)

plt.axis('off')  

plt.text(10, 10, f'Predicted: {sorted_results[0][0]} with probability: {sorted_results[0][1]}', color='white', fontsize=12, ha='left', va='top')

plt.show()
