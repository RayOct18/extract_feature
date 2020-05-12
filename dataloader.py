import torchvision.transforms as transforms
import torch
import torchvision.datasets as dset
import torchvision.utils as vutils
import math
from load_csv import CustomDatasetFromImages


def loadImgs(dataset_path=None, csv_dir=None, img_size=224, batchSize=32, num_workers=4, shuffle=False, drop_last=False):
    if csv_dir:
        dataset = CustomDatasetFromImages(dataset_path=dataset_path,
                                          csv_path=csv_dir,
                                          transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.RandomCrop(img_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ]))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                                 shuffle=True,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 drop_last=drop_last
                                                 )
        data_len = dataset.__len__()

    else:
        dataloader = None

    return dataloader
