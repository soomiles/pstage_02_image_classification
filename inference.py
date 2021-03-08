import random
import os
from importlib import import_module

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from dataset import TestDataset
from model import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    seed_everything(42)

    data_dir = '/mnt/ssd/data/mask/mask_final'  # os.environ['SM_CHANNEL_EVAL']
    model_dir = './results'  # os.environ['SM_CHANNEL_MODEL']
    output_dir = './outputs'  # os.environ['SM_OUTPUT_DATA_DIR']

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model_name = "VGG19"
    best_checkpoint_path = os.path.join(model_dir, 'best_checkpoint.ckpt')

    batch_size = 64
    num_workers = 8
    num_classes = 18

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- model
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    checkpoint = torch.load(best_checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f'Model Loaded: {os.path.basename(best_checkpoint_path)}')

    for status in ['public', 'private']:
        img_root = os.path.join(data_dir, status, 'images')
        info_path = os.path.join(data_dir, status, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

        preds = []
        for idx, images in enumerate(loader):
            images = images.to(device)
            with torch.no_grad():
                pred = model(images)
                pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
        info['ans'] = preds
        info.to_csv(os.path.join(output_dir, f'{status}.csv'), index=False)
    print(f'Inference Done!')
