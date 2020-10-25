import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch
import torchvision.transforms as T
import cv2
import pandas as pd
from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from collections import Counter
import torch.nn as nn
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]
def in_extensions(filename, extensions):
    return any(filename.endswith(extension) for extension in extensions)
def find_valid_files(dir, extensions=None, max_dataset_size=float("inf")):
    """
    Get all the images recursively under a dir.
    Args:
        dir:
        extensions: specific extensions to look for. else will use IMG_EXTENSIONS
        max_dataset_size:

    Returns: found files, where each item is a tuple (id, ext)

    """
    if isinstance(extensions, str):
        extensions = [extensions]
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if in_extensions(fname, extensions if extensions else IMG_EXTENSIONS):
                path = os.path.join(root, fname)
                images.append(path)
    return images[: min(max_dataset_size, len(images))]
def per_channel_transform(input_tensor, transform_function):
    """
    Randomly transform each of n_channels of input data.
    Out of place operation
    :param input_tensor: must be a numpy array of size (n_channels, w, h)
    :param transform_function: any torchvision transforms class
    :return: transformed pt tensor
    """
    input_tensor = input_tensor.numpy()
    tform_input_np = np.zeros(shape=input_tensor.shape, dtype=input_tensor.dtype)
    n_channels = input_tensor.shape[0]
    for i in range(n_channels):
        tform_input_np[i] = np.array(
            transform_function(Image.fromarray(input_tensor[i]))
        )
    return torch.from_numpy(tform_input_np)
def to_onehot_tensor(sp_matrix, n_labels):
    """
    convert sparse scipy labels matrix to onehot pt tensor of size (n_labels,H,W)
    Note: sparse tensors aren't supported in multiprocessing https://github.com/pytorch/pytorch/issues/20248

    :param sp_matrix: sparse 2d scipy matrix, with entries in range(n_labels)
    :return: pt tensor of size(n_labels,H,W)
    """
    sp_matrix = sp_matrix.tocoo()
    indices = np.vstack((sp_matrix.data, sp_matrix.row, sp_matrix.col))
    indices = torch.LongTensor(indices)
    values = torch.Tensor([1.0] * sp_matrix.nnz)
    shape = (n_labels,) + sp_matrix.shape
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape)).to_dense()
def decompress_cloth_segment(fname, n_labels):
    """
    Load cloth segmentation sparse matrix npz file and output a one hot tensor.
    :return: tensor of size(H,W,n_labels)
    """
    try:
        data_sparse = load_npz(fname)
    except Exception as e:
        print("Could not decompress cloth segment:", fname)
        raise e
    return to_onehot_tensor(data_sparse, n_labels)
def remove_top_dir(dir, n=1):
    """
    Removes the top level dirs from a path
    Args:
        dir:
        n:

    Returns:

    """
    parts = dir.split(os.path.sep)
    top_removed = os.path.sep.join(parts[n:])
    return top_removed
def get_dir_file_extension(dir, check=5):
    """
    Guess what extensions are for all files in a dir.
    Args:
        dir:
        check:

    Returns:

    """
    exts = []
    for root, _, fnames in os.walk(dir, followlinks=True):
        for fname in fnames[:check]:
            ext = os.path.splitext(fname)[1]
            exts.append(ext)
    if len(exts) == 0:
        raise ValueError(f"did not find any files under dir: {dir}")
    return Counter(exts).most_common(1)[0][0]
def remove_extension(fname):
    return os.path.splitext(fname)[0]
def get_corresponding_file(original, target_dir, target_ext=None):
    """
    Say an original file is
        dataroot/subject/body/SAMPLE_ID.jpg

    And we want the corresponding file
        dataroot/subject/cloth/SAMPLE_ID.npz

    The corresponding file is in target_dir dataroot/subject/cloth, so we replace the
    top level directories with the target dir

    Args:
        original:
        target_dir:
        target_ext:

    Returns:

    """
    # number of top dir to replace
    num_top_parts = len(target_dir.split(os.path.sep))
    # replace the top dirs
    top_removed = remove_top_dir(original, num_top_parts)
    target_file = os.path.join(target_dir, top_removed)
    # extension of files in the target dir
    if not target_ext:
        target_ext = get_dir_file_extension(target_dir)
    # change the extension
    target_file = remove_extension(target_file) + target_ext
    return target_file

class WarpDataset(Dataset):
    def __init__(self, is_train=True):
        self.cloth_dir = "./data/texture/cloth"
        extensions = [".npz"]
        self.cloth_files = find_valid_files(self.cloth_dir, extensions)
        self.cloth_transform = T.RandomOrder([
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=20
            ),
            T.RandomPerspective()
        ])
        self.body_dir = "./data/texture/body"
        self.normalize_body = T.Normalize([0.06484050184440379, 0.06718090599394404, 0.07127327572275131],
                                          [0.2088075459038679, 0.20012519201951368, 0.23498672043315685])
        self.is_train = is_train

    def __len__(self):
        return len(self.cloth_files)

    def _perform_cloth_transform(self, cloth_tensor):
        """ Either does per-channel transform or whole-image transform """
        return per_channel_transform(cloth_tensor, self.cloth_transform)

            # return self.input_transform(cloth_tensor)
    def _load_cloth(self, index):
        """
        Loads the cloth file as a tensor
        """
        cloth_file = self.cloth_files[index]
        target_cloth_tensor = decompress_cloth_segment(
            cloth_file, 19
        )
        if self.is_train:
            # during train, we want to do some fancy transforms
            input_cloth_tensor = target_cloth_tensor.clone()

            # apply the transformation for input cloth segmentation
            input_cloth_tensor = self._perform_cloth_transform(input_cloth_tensor)

            return cloth_file, input_cloth_tensor, target_cloth_tensor
        else:
            # during inference, we just want to load the current cloth
            return cloth_file, target_cloth_tensor, target_cloth_tensor
    def _load_body(self, index):
        """ Loads the body file as a tensor """
        if self.is_train:
            # use corresponding strategy during train
            cloth_file = self.cloth_files[index]
            body_file = get_corresponding_file(cloth_file, self.body_dir)
        else:
            # else we have to load by index
            body_file = self.body_files[index]
        as_pil_image = Image.open(body_file).convert("RGB")
        body_tensor = self.normalize_body(T.ToTensor()(as_pil_image))
        return body_file, body_tensor
    def __getitem__(self, index):
        cloth_file, input_cloth_tensor, target_cloth_tensor = self._load_cloth(index)
        body_file, body_tensor = self._load_body(index)
        input_cloth_tensor = nn.functional.interpolate(
            input_cloth_tensor.unsqueeze(0), size=128
        ).squeeze()
        if self.is_train:
            target_cloth_tensor = nn.functional.interpolate(
                target_cloth_tensor.unsqueeze(0), size=128
            ).squeeze()
        body_tensor = nn.functional.interpolate(
            body_tensor.unsqueeze(0),
            size=128,
            mode="bilinear",  # same as default for torchvision.resize
        ).squeeze()
        return {
            "body_paths": body_file,
            "bodys": body_tensor,
            "cloth_paths": cloth_file,
            "input_cloths": input_cloth_tensor,
            "target_cloths": target_cloth_tensor,
        }

if __name__ == "__main__":
    warp_dataset = WarpDataset()
    print(len(warp_dataset))
    warp_loader = DataLoader(warp_dataset, 1)
    for data in warp_loader:
        print(data)
        exit()