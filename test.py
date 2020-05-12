import argparse
import better_exceptions
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from model.dex_models import Age
from defaults import _C as cfg
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom datasets
from load_csv import CustomDatasetFromImages


class ImageFolderWithPaths(dsets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        filename = os.path.split(path)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (filename[-1],))
        return tuple_with_path



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default=R'G:\GoogleDrive\Avlab\program\Self-Attention-GAN-Experiment\generate\test', help="Data root directory")
    parser.add_argument("--resume", type=str, default='./MORPH_256/checkpoint/epoch008_0.15920_2.5270.pth', help="Model weight to be tested")
    parser.add_argument("--save_path", type=str, default='../Self-Attention-GAN-Experiment/evaluate/age', help="Result path")
    parser.add_argument("--csv_path", type=str, default='../Self-Attention-GAN-Experiment/evaluate/age', help="Result path")
    parser.add_argument("--version", type=str, default=None, help="Result path")
    parser.add_argument("--target", type=str, default=None, help="Result path")
    parser.add_argument("--set", type=str, default=None, help="Result path")
    args = parser.parse_args()
    return args

def predict(validate_loader, model, device, args):

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    df = pd.DataFrame()
    filenames = []
    preds = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y, name) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)
                # compute output
                x = (x + 1) / 2
                x = x.clamp_(0, 1)
                outputs = model(x)
                filenames.append(name)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    filenames = np.concatenate(filenames, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    data = list(zip(filenames.tolist(), ave_preds.tolist()))
    df = df.append(data, ignore_index=True)
    df.columns = ['name', 'age']
    save_name = os.path.split(args.test_dir)[-1]
    df.to_csv(os.path.join(save_dir ,'{}_{}_m{:.2f}_s{:.2f}.csv'.format(args.version, save_name, ave_preds.mean(), ave_preds.std())), index=False)
    print('Testing mean age={}, stdv={}'.format(ave_preds.mean(), ave_preds.std()))

def main():
    args = get_args()

    cfg.freeze()

    # create model
    model = Age()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomCrop(cfg.MODEL.IMG_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if not args.set is None:
        csv_dir = os.path.join(args.csv_path, args.set, 'val_src_{}.csv'.format(args.target))
        dataset = CustomDatasetFromImages(args.test_dir, csv_dir, transform=transform)
    else:
        dataset = ImageFolderWithPaths(args.test_dir, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=cfg.TEST.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=cfg.TEST.WORKERS,
                                              drop_last=False)


    print("=> start predicting")
    predict(test_loader, model, device, args)
    print(f"Finish!")


if __name__ == '__main__':
    main()
