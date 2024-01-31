import torch
from torch.utils.data import Dataset

class CircleDataset(Dataset):
    def __init__(self,
                 n_samples:int=1000,
                 n_points:int=32
                 ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_points = n_points
        self.data = self.sample_on_circle()
    
    def sample_on_circle(self):
        radius = torch.ones(self.n_samples, self.n_points)
        center = torch.zeros(self.n_samples, self.n_points, 2)

        theta = 2 * torch.pi * torch.rand(self.n_samples, self.n_points)
        sorted = torch.argsort(theta, dim=-1).unsqueeze(1)

        # Convert spherical coordinates to Cartesian coordinates
        x = radius * torch.cos(theta) + center[:, :, 0]
        y = radius * torch.sin(theta) + center[:, :, 1]

        points = torch.stack([x, y], dim=1)
        points = torch.take_along_dim(points, sorted, dim=-1)
        points += torch.randn_like(points) * 0.005
        return points.permute(0,2,1)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, index) -> torch.Tensor:
        data = self.data[index, :, :]
        return data
    
class SphereDataset(Dataset):
    def __init__(self,
                 n_samples:int=1000,
                 n_points:int=32
                 ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_points = n_points
        self.data = self.sample_on_sphere()
    
    def sample_on_sphere(self):
        radius = torch.rand(self.n_samples, 1) + 1
        center = torch.rand(self.n_samples, 1)

        phi = 2 * torch.pi * torch.rand(self.n_samples, self.n_points)
        cos_theta = 2 * torch.rand(self.n_samples, self.n_points) - 1
        theta = torch.acos(cos_theta)

        # Convert spherical coordinates to Cartesian coordinates
        x = radius * torch.sin(theta) * torch.cos(phi) + center[0]
        y = radius * torch.sin(theta) * torch.sin(phi) + center[1]
        z = radius * torch.cos(theta) + center[2]

        points = torch.stack([x, y, z], dim=1)
        return points
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index, :]