from typing import Iterable, Optional
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import Resize, InterpolationMode
import torch.nn as nn
import os
from scipy.ndimage import binary_erosion
from preprocessing import gaussian_blur
from model_temp import TemperatureNet, LipschitzLinear, gaussian_blur


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = r'C:\Users\guoli\Documents\Coding\python stuff\climate net\final'


class Sobel(nn.Module):
    """https://github.com/chaddy1004/sobel-operator-pytorch/blob/master/model.py"""

    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)
        self.to(DEVICE)

    def forward(self, img):
        return self.filter(img)


class PrecipitationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_sobel = False

        d = 2
        hidden_dims = 5
        self.elevation_diffs = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 1)
            LipschitzLinear(hidden_dims, 2),
            nn.Tanhshrink()
        )
        d = 1
        hidden_dims = 3
        self.latitude = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 1)
            LipschitzLinear(hidden_dims, 3),
            nn.Tanhshrink()
        )
        d = 5
        hidden_dims = 3
        self.gwi = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 1)
            LipschitzLinear(hidden_dims, 2),
            nn.Tanhshrink()
        )
        d = 3
        hidden_dims = 2
        self.dwi = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 2)
            LipschitzLinear(hidden_dims, 2),
            nn.Tanhshrink()
        )
        d = 3
        hidden_dims = 2
        self.coastline = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 2)
            LipschitzLinear(hidden_dims, 2),
            nn.Tanhshrink()
        )
        d = 9
        hidden_dims = 5
        self.coastline = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 2)
            LipschitzLinear(hidden_dims, 4),
            nn.Tanhshrink()
        )

        d = 19
        hidden_dims = 6
        self.wind_water = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 3)
            LipschitzLinear(hidden_dims, 5),
            nn.Tanhshrink()
        )

        d = 22
        hidden_dims = 8
        self.prec_sum = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 12)
            LipschitzLinear(hidden_dims, 1),
        )

        d = 22
        hidden_dims = 8
        self.prec = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 12)
            LipschitzLinear(hidden_dims, 12)
        )

        d = 3
        hidden_dims = 3
        self.conv = nn.Sequential(
            nn.BatchNorm2d(d),
            nn.Conv2d(d, hidden_dims, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(hidden_dims, hidden_dims, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_dims, 1, kernel_size=3, padding=1),
        )

        self.to(DEVICE)
        self.apply(self.initialize_weights)

    def initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight)

    def to2d(self, x: Tensor) -> Tensor:
        D, H, W = x.shape
        return torch.permute(x, (1, 2, 0)).reshape(H * W, D)

    def to3d(self, x: Tensor) -> Tensor:
        D = x.shape[1]
        return torch.permute(torch.reshape(x, (180, 360, D)), (2, 0, 1))

    def forward(self, x: Tensor, temp: Tensor, resize: Optional[Resize] = None) -> tuple[Tensor, Tensor]:
        """
        x is [D_in, H, W]
        temp is [12, H, W]
        output is [D_out, H, W]

        [elevation, latitude, land, gwi (5), dwi (3), dli, itcz, wind_onshore_offshore,
         water_current_temperature, closest_coast, continentality, ediff (4), west, east]
        """
        # Elevation differences
        latitude = x[None, 1]
        rising_left = x[[21, 23]]
        rising_right = x[[22, 24]]
        boundary_left_down = gaussian_blur((latitude >= 0) & (latitude < 30), 5)
        boundary_left_up = gaussian_blur((latitude < 0) & (latitude >= -30), 5)
        boundary_right_down = gaussian_blur((latitude < -20) & (latitude >= -60), 5)
        boundary_right_up = gaussian_blur((latitude >= 20) & (latitude < 60), 5)
        left_down = boundary_left_down * gaussian_blur(torch.relu(-rising_right), 1)
        left_up = boundary_left_up * gaussian_blur(torch.relu(rising_left), 1)
        right_down = boundary_right_down * gaussian_blur(torch.relu(-rising_left), 1)
        right_up = boundary_right_up * gaussian_blur(torch.relu(rising_right), 1)
        left_down = self.to2d(left_down)
        left_up = self.to2d(left_up)
        right_down = self.to2d(right_down)
        right_up = self.to2d(right_up)
        elevation_diffs = torch.concatenate([self.elevation_diffs(left_down),
                                             self.elevation_diffs(right_down),
                                             self.elevation_diffs(left_up),
                                             self.elevation_diffs(right_up)], dim=1)  # (180 * 360, 8)
        # Wind, water
        latitude = self.latitude(self.to2d(x[None, 1]))  # (180 * 360, 3)
        gwi = self.gwi(self.to2d(x[[3, 4, 5, 6, 7]]))  # (180 * 360, 2)
        dwi = self.dwi(self.to2d(x[[8, 9, 10]]))  # (180 * 360, 2)
        y = torch.concatenate([self.to2d(x[[13, 14, 15, 25, 26]]), gwi, dwi], dim=1)
        coastline = self.coastline(y)  # (180 * 360, 4)
        y = torch.concatenate([self.to2d(x[[11, 12, 25, 26]]), elevation_diffs, latitude, coastline], dim=1)
        wind_water = self.wind_water(y)  # (180 * 360, 5)

        # Precipitation (bounded by temperature)
        continentality = torch.max(temp, dim=0, keepdim=True).values - torch.min(temp, dim=0, keepdim=True).values
        temp_avg = torch.mean(temp, dim=0, keepdim=True)
        l1 = torch.maximum(torch.tensor(0.), ((temp_avg + 100) / 16.57227) ** 5 + 1)
        l2 = torch.sigmoid(temp_avg / 2 - 6.351945) * 12500 + 7500
        max_prec_sum = torch.empty_like(temp_avg, device=temp_avg.device)
        mask = temp_avg < 0
        max_prec_sum[mask] = l1[mask]
        max_prec_sum[~mask] = l2[~mask]
        del l1, l2, mask

        y = torch.concatenate([self.to2d(x[[0, 1, 11, 12]]), wind_water, latitude, elevation_diffs,
                               self.to2d(continentality), self.to2d(temp_avg)], dim=1)
        prec_sum = self.prec_sum(y - torch.log(self.to2d(max_prec_sum)))
        prec_sum = torch.sigmoid(prec_sum + self.to2d(self.conv(x[[0, 1, 2]][None])[0])) * self.to2d(max_prec_sum)
        prec = torch.softmax(self.prec(y), dim=1) * prec_sum
        prec_sum = torch.sum(prec, dim=1, keepdim=True)

        f = self.to3d if resize is None else lambda x: resize(self.to3d(x))
        return f(prec), f(prec_sum)

    @torch.no_grad()
    def predict(self, x: np.ndarray, temp: np.ndarray) -> np.ndarray:
        self.eval()
        x = torch.from_numpy(x).float().to(DEVICE)
        temp = torch.from_numpy(temp).float().to(DEVICE)
        return torch.concatenate(self.forward(x, temp), dim=0).cpu().numpy()

    def log(self, x: Tensor) -> Tensor:
        values = torch.empty_like(x, device=x.device)
        zero = torch.tensor(0., device=x.device)
        values[x < 0] = torch.minimum(zero, -torch.log(-x[x < 0] + 1))
        values[x >= 0] = torch.maximum(zero, torch.log(x[x >= 0] + 1))
        return values

    def loss_function(self, y: Tensor, t: Tensor, weight: Tensor, mask: Tensor) -> Tensor:
        if len(y.shape) == 3:
            y = y[None]
            t = t[None]
        weight = weight[None]
        mask = mask[None]
        factor = 100
        loss = torch.mean(mask * weight * factor * (torch.log(y + 1) - torch.log(t + 1)) ** 4)

        if self.use_sobel:
            mask = binary_erosion(mask.cpu().numpy(), iterations=3)
            mask = torch.from_numpy(mask).to(DEVICE)[None]
            edges = Sobel()
            edge_loss = torch.mean(torch.stack([
                mask * weight * factor * (self.log(edges(y[:, None, i, :, :])) - self.log(edges(t[:, None, i, :, :]))) ** 4 for i in range(y.shape[1])
            ]))
            return (loss + edge_loss) / 2
        return loss

    def get_loss(self, y_all: Iterable[Tensor], t_all: Iterable[Tensor],
                 weight: Optional[Tensor] = None, mask: Optional[Tensor] = None,
                 inds: Optional[list[int]] = None) -> list[Tensor]:
        if weight is None:
            weight = torch.ones_like(y_all[0][0], dtype=float, device=y_all[0][0].device)
        if mask is not None:
            weight = weight.clone()
            weight[~mask] = 0
        if inds is None:
            inds = range(len(y_all))
        return [self.loss_function(y_all[i], t_all[i], weight, mask) for i in inds]

    def fit(self, temp_model: TemperatureNet, inputs: list[Tensor], outputs: list[Tensor]) -> None:
        self.train()
        X, X_f, X_r, X_rf = inputs  # (D_input, H, W)
        t, t_f, t_r, t_rf = outputs  # (D_output, H, W)

        resize = Resize(t_r.shape[1:], InterpolationMode.BILINEAR)
        latitude = X[1]
        latitude_r = resize(X[None, 1])
        # neg_softplus = lambda x: 1 / (torch.log(1 + torch.exp(-x - 15)) + 1)
        # latitude_weight = neg_softplus(torch.mean(t[:12], dim=0))
        # latitude_weight_f = neg_softplus(torch.mean(t_f[:12], dim=0))
        # latitude_weight_r = neg_softplus(torch.mean(t_r[:12], dim=0))
        # latitude_weight_rf = neg_softplus(torch.mean(t_rf[:12], dim=0))
        latitude_weight = torch.cos(latitude * torch.pi / 180) ** 2
        latitude_weight_r = resize(torch.cos(latitude_r * torch.pi / 180))[0] ** 2

        # Filter out water, badly-sampled pixels
        get_continentality = lambda x: torch.max(x[:12], dim=0, keepdim=True).values - torch.min(x[:12], dim=0, keepdim=True).values
        continentality = [get_continentality(t)[0], get_continentality(t_f)[0],
                          get_continentality(t_r)[0], get_continentality(t_rf)[0]]
        polar = torch.abs(latitude) > 50
        polar_r = torch.abs(latitude_r)[0] > 50
        land = X[2].bool() & ~(polar & (continentality[0] < 1))
        land_f = X_f[2].bool() & ~(polar & (continentality[1] < 1))
        land_r = torch.round(resize(X_r[None, 2])[0]).bool() & ~(polar_r & (continentality[2] < 1))
        land_rf = torch.round(resize(X_rf[None, 2])[0]).bool() & ~(polar_r & (continentality[3] < 1))
        t[:, ~land] = 0
        t_f[:, ~land_f] = 0

        extract_t = lambda x: (x[12:], torch.sum(x[12:], dim=0, keepdim=True), x[:12])
        t = extract_t(t)
        t_f = extract_t(t_f)
        t_r = extract_t(t_r)
        t_rf = extract_t(t_rf)

        with torch.no_grad():
            y_temp = temp_model.forward(X)[0]
            y_temp_f = temp_model.forward(X_f)[0]
            y_temp_r = temp_model.forward(X_r)[0]
            y_temp_rf = temp_model.forward(X_rf)[0]

        optimizer = torch.optim.Adam(self.parameters())
        epochs = 30 if DEVICE == 'cpu' else 2000
        j = [0, 1]
        self.use_sobel = True
        for i in range(epochs):
            p = i / (epochs - 1)
            y = self.forward(X, t[2] * p + y_temp * (1 - p))
            y_f = self.forward(X_f, t_f[2] * p + y_temp_f * (1 - p))
            y_r = self.forward(X_r, y_temp_r, resize)
            y_rf = self.forward(X_rf, y_temp_rf, resize)

            weights = torch.softmax(torch.rand(4, device=DEVICE), dim=0)
            weights /= torch.mean(weights)
            weights = weights * (1 - p ** 0.5) + p ** 0.5
            loss1 = self.get_loss(y, t, latitude_weight, land, j)
            loss2 = self.get_loss(y_f, t_f, latitude_weight, land_f, j)
            loss3 = self.get_loss(y_r, t_r, latitude_weight_r, land_r, j)
            loss4 = self.get_loss(y_rf, t_rf, latitude_weight_r, land_rf, j)

            loss = sum([weights[i] * (loss1[i] + loss2[i] + loss3[i] + loss4[i]) for i in range(len(loss1))])
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.parameters(), 100)
            optimizer.step()
            print(f'{i}: temp = {loss}')

    def save(self, path: str = '') -> None:
        torch.save(self.state_dict(), os.path.join(path, 'precipitation-net.pt'))
        # model = PrecipitationNet()
        # model.load(r'...')
        # x = torch.randn(27, 180, 360)
        # temp = torch.randn(12, 180, 360)
        # torch.onnx.export(model, (x, temp), 'precipitation-net.onnx', input_names=['input'], dynamo=False)

    def load(self, path: str = '') -> None:
        self.load_state_dict(torch.load(os.path.join(path, 'precipitation-net.pt'), map_location=DEVICE))


def flip_targets(matrix: Tensor) -> Tensor:
    """Flip upside-down, but also flip seasons"""
    temp = torch.flip(torch.roll(matrix[:12], 6, dims=0), dims=(1, 2))
    prec = torch.flip(torch.roll(matrix[12:], 6, dims=0), dims=(1, 2))
    return torch.concatenate([temp, prec], dim=0)


def load_npy(path: str) -> Tensor:
    return torch.from_numpy(np.load(os.path.join(DATA_PATH, path))).float().to(DEVICE)


def train() -> PrecipitationNet:
    X, X_f = load_npy('x.npy'), load_npy('x-flipped.npy')
    X_r, X_rf = load_npy('x-retrograde.npy'), load_npy('x-retrograde-flipped.npy')
    t, t_r = load_npy('t.npy'), load_npy('t-retrograde.npy')
    t_f, t_rf = flip_targets(t), flip_targets(t_r)

    temp_model = TemperatureNet()
    temp_model.load(DATA_PATH)
    model = PrecipitationNet()
    print(f'Training on {sum(p.numel() for p in model.parameters())} parameters...')
    model.fit(temp_model, [X, X_f, X_r, X_rf], [t, t_f, t_r, t_rf])

    # X = load_npy('x.npy')
    # land = X[2].numpy()
    # temp_model = TemperatureNet()
    # temp_model.load(DATA_PATH)
    # temp = temp_model.predict(X.numpy())[:12]
    # result = model.predict(X.numpy(), temp)
    #
    # t = load_npy('t.npy').numpy()
    # error = np.mean(result[:12] - t[:12], axis=0)

    return model

