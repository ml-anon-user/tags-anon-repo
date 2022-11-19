import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from plyfile import PlyData

class PointCloudWithNormalsDataset(Dataset):

    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_normals = []
        self.pointcloud_names = []
        self.resolution = resolution
        self.split = split
        
        self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
        for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
            if split == 'train':
                if fn[-3:] != 'ply':
                    continue
                pcl_path = os.path.join(self.pcl_dir, fn)
                if not os.path.exists(pcl_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)

                pcl_all = PlyData.read(pcl_path)['vertex'].data.tolist()
                pcl = torch.from_numpy(np.asarray(pcl_all, dtype=np.float32)[:, :3])
                pcl_normal = torch.from_numpy(np.asarray(pcl_all, dtype=np.float32)[:, 3:6])
                self.pointcloud_normals.append(pcl_normal)
            else:
                if fn[-3:] != 'xyz':
                    continue
                pcl_path = os.path.join(self.pcl_dir, fn)
                if not os.path.exists(pcl_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)
                pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            
            self.pointclouds.append(pcl)                    
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)

    def __str__(self):
        return "Dataset with resolution: {}".format(self.resolution)

    def __getitem__(self, idx):
        if self.split == 'train':
            data = {
                'pcl_clean': self.pointclouds[idx].clone(), 
                'pcl_clean_normals': self.pointcloud_normals[idx].clone(), 
                'name': self.pointcloud_names[idx]
            }
        else:
            data = {
                'pcl_clean': self.pointclouds[idx].clone(), 
                'name': self.pointcloud_names[idx]
            }
        if self.transform is not None:
            data = self.transform(data)

        return data

