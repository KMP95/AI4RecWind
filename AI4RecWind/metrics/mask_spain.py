import os
import xarray as xr
import torch

def load_mask(args):
    spain_mask = xr.open_dataset(os.path.join(args.mask_root, args.spain_mask))["ws"].values
    spain_mask = torch.tensor(spain_mask).to('cpu')

    return spain_mask
