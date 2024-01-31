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
        self.points, self.radius = self.sample_on_circle()
    
    def sample_on_circle(self):
        radius = torch.randint(high=2, size=(self.n_samples, self.n_points)) / 2. + 0.5 # Radius .5 or 1 (between 0 and 1)
        center = torch.zeros(self.n_samples, self.n_points, 2)

        theta = 2 * torch.pi * torch.rand(self.n_samples, self.n_points)

        # Convert spherical coordinates to Cartesian coordinates
        x = radius * torch.cos(theta) + center[:, :, 0]
        y = radius * torch.sin(theta) + center[:, :, 1]

        points = torch.stack([x, y], dim=1)
        #points += torch.randn_like(points) * 0.005
        return points.permute(0,2,1), radius
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.points[index, :, :], self.radius[index, :]
