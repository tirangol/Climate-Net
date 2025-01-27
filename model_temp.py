from typing import Iterable, Optional
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import Resize, InterpolationMode
import torch.nn as nn
import os
from scipy.ndimage import binary_erosion
from preprocessing import gaussian_blur


# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
DATA_PATH = r'C:\Users\guoli\Documents\Coding\python stuff\climate net\final'


class LipschitzLinear(torch.nn.Module):
    """lipmlp (https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max()  # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)


class Sobel(nn.Module):
    """https://github.com/chaddy1004/sobel-operator-pytorch/blob/master/model.py"""

    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1,
                                bias=False)
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)
        self.to(DEVICE)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


class TemperatureNet(nn.Module):
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
            LipschitzLinear(hidden_dims, 1),
            nn.Tanhshrink()
        )
        d = 20
        hidden_dims = 8
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
            LipschitzLinear(hidden_dims, 4),
            nn.Tanhshrink()
        )

        d = 8
        hidden_dims = 3
        self.elevation_slope = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            LipschitzLinear(hidden_dims, 1),
            # nn.Linear(hidden_dims, 1),
            nn.Sigmoid()
        )

        d = 16
        hidden_dims = 6
        self.continentality = nn.Sequential(
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
            # nn.Linear(hidden_dims, 1),
            LipschitzLinear(hidden_dims, 1),
            nn.Sigmoid()
        )

        d = 13
        hidden_dims = 6
        self.temp_avg = nn.Sequential(
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
            # nn.Linear(hidden_dims, 1),
            LipschitzLinear(hidden_dims, 1),
            nn.Sigmoid()
        )

        d = 13
        hidden_dims = 6
        self.offset = nn.Sequential(
            nn.BatchNorm1d(d),
            # nn.Linear(d, hidden_dims),
            LipschitzLinear(d, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, hidden_dims),
            LipschitzLinear(hidden_dims, hidden_dims),
            nn.GELU(),
            # nn.Linear(hidden_dims, 12)
            LipschitzLinear(hidden_dims, 12),
        )

        # d = 14  # monthly temp (12), avg temp, continentality
        # d_out = 12
        # self.conv = nn.Sequential(
        #     nn.BatchNorm2d(d),
        #     nn.ConvTranspose2d(d, d_out, kernel_size=(7, 7), padding=3),
        #     nn.Sigmoid(),
        # )
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

    def forward(self, x: Tensor, resize: Optional[Resize] = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        x is [D_in, H, W]
        output is [D_out, H, W]

        [elevation, latitude, land, gwi (5), dwi (3), dli, itcz, wind_onshore_offshore,
         water_current_temperature, closest_coast, continentality, ediff (4), west, east]
        """
        # Elevation differences
        latitude = x[1].cpu().numpy()
        rising_left = x[[21, 23]].cpu().numpy()
        rising_right = x[[22, 24]].cpu().numpy()
        boundary_left_down = gaussian_blur((latitude >= 0) & (latitude < 30), 5)
        boundary_left_up = gaussian_blur((latitude < 0) & (latitude >= -30), 5)
        boundary_right_down = gaussian_blur((latitude < -20) & (latitude >= -60), 5)
        boundary_right_up = gaussian_blur((latitude >= 20) & (latitude < 60), 5)
        left_down = boundary_left_down * gaussian_blur(-np.minimum(0, rising_right), 1, True)
        left_up = boundary_left_up * gaussian_blur(np.maximum(0, rising_left), 1, True)
        right_down = boundary_right_down * gaussian_blur(-np.minimum(0, rising_left), 1, True)
        right_up = boundary_right_up * gaussian_blur(np.maximum(0, rising_right), 1, True)
        left_down = self.to2d(torch.from_numpy(left_down).to(DEVICE).float())
        left_up = self.to2d(torch.from_numpy(left_up).to(DEVICE).float())
        right_down = self.to2d(torch.from_numpy(right_down).to(DEVICE).float())
        right_up = self.to2d(torch.from_numpy(right_up).to(DEVICE).float())
        elevation_diffs = torch.concatenate([self.elevation_diffs(left_down),
                                             self.elevation_diffs(right_down),
                                             self.elevation_diffs(left_up),
                                             self.elevation_diffs(right_up)],
                                            dim=1)  # (180 * 360, 4)
        # Wind + water
        y = torch.concatenate([
            self.to2d(x[[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25, 26]]), elevation_diffs
        ], dim=1)
        wind_water = self.wind_water(y)  # (180 * 360, 4)

        # Elevation slope
        elevation = x[0].flatten()[:, None]
        lat_itcz_dli = self.to2d(x[[1, 11, 12]])
        y = torch.concatenate([lat_itcz_dli, elevation, wind_water], dim=1)
        elevation_slope = self.elevation_slope(y)
        elevation_slope = (5 * elevation_slope - 10) / 1000

        # Continentality
        c = self.to2d(x[[16, 17, 18, 19, 20]])
        abs_latitude = torch.abs(x[1])
        l1 = abs_latitude + 5
        l2 = abs_latitude * 1.25 + 1.5
        l3 = abs_latitude * 1.5 - 5
        max_continentality = torch.minimum(torch.tensor(80., device=x.device),
                                           torch.maximum(torch.maximum(torch.maximum(l1, l2), l3),
                                                         torch.tensor(10., device=x.device)))
        y = torch.concatenate([lat_itcz_dli, wind_water, torch.tanh(elevation_diffs / 1000) * 1000, c], dim=1)
        continentality = self.continentality(y) * max_continentality.flatten()[:, None]

        # Average temperature
        min_temp_avg = -0.75 * abs_latitude + 15
        l1 = -0.5 * abs_latitude + 60
        l2 = 0.5 * abs_latitude + 40
        l3 = -0.75 * abs_latitude + 70
        max_temp_avg = torch.minimum(torch.minimum(l1, l2), l3)
        range_temp_avg = max_temp_avg - min_temp_avg
        min_temp_avg = min_temp_avg.flatten()[:, None]
        range_temp_avg = range_temp_avg.flatten()[:, None]

        f = lambda x: torch.exp(-x ** 2 / 200)
        latitude_special = 2 * f(abs_latitude - 20) - f(abs_latitude) / 2 - f(abs_latitude - 50) - abs_latitude ** 2 / 900
        y = torch.concatenate([self.to2d(latitude_special[None]), lat_itcz_dli,
                               continentality, elevation_diffs, wind_water], dim=1)
        temp_avg = min_temp_avg + self.temp_avg(y) * range_temp_avg + elevation * elevation_slope

        # Offsets
        offset = self.offset(y)
        offset = offset - torch.mean(offset, dim=1, keepdim=True)
        max_magnitude = torch.max(torch.abs(offset), dim=1, keepdim=True).values
        month_variation = continentality * offset / torch.maximum(max_magnitude, torch.tensor(1.))
        temp = temp_avg + month_variation

        # y = self.to3d(torch.concatenate([temp, temp_avg, continentality], dim=1))
        # temp = temp + self.to2d(self.conv(y[None])[0]) * 5 - 2.5

        f = self.to3d if resize is None else lambda x: resize(self.to3d(x))
        return f(temp), f(temp_avg), f(continentality), f(elevation_slope)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        print("TEMP START")
        print(x)
        print(x.shape)
        print(torch.from_numpy(x))
        x = torch.from_numpy(x).float()
        print("X")
        x = x.to(DEVICE)
        print("X DEVICE")
        print(self.device)
        return torch.concatenate(self.forward(x), dim=0).cpu().numpy()

    def loss_function(self, y: Tensor, t: Tensor, weight: Tensor, mask: Tensor) -> Tensor:
        if len(y.shape) == 3:
            y = y[None]
            t = t[None]
        weight = weight[None]
        mask = mask[None]
        loss = torch.mean(mask * weight * (y - t) ** 2)

        if self.use_sobel:
            mask = binary_erosion(mask.cpu().numpy(), iterations=3)
            mask = torch.from_numpy(mask).to(DEVICE)[None]
            edges = Sobel()
            edge_loss = torch.mean(torch.stack([
                mask * weight * (edges(y[:, None, i, :, :]) - edges(t[:, None, i, :, :])) ** 2 for i in range(y.shape[1])
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

    def fit(self, inputs: list[Tensor], outputs: list[Tensor]) -> None:
        self.train()
        X, X_f, X_r, X_rf = inputs  # (D_input, H, W)
        t, t_f, t_r, t_rf = outputs  # (D_output, H, W)

        resize = Resize(t_r.shape[1:], InterpolationMode.BILINEAR)
        latitude = X[1]
        latitude_r = resize(X[None, 1])
        neg_softplus = lambda x: 1 / (torch.log(1 + torch.exp(-x - 15)) + 1)
        latitude_weight = neg_softplus(torch.mean(t[:12], dim=0))
        latitude_weight_f = neg_softplus(torch.mean(t_f[:12], dim=0))
        latitude_weight_r = neg_softplus(torch.mean(t_r[:12], dim=0))
        latitude_weight_rf = neg_softplus(torch.mean(t_rf[:12], dim=0))
        # latitude_weight = torch.cos(latitude * torch.pi / 180)
        # latitude_weight_r = resize(torch.cos(latitude_r * torch.pi / 180))[0]

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

        extract_t = lambda x: (x[:12], torch.mean(x[:12], dim=0, keepdim=True), get_continentality(x))
        t = extract_t(t)
        t_f = extract_t(t_f)
        t_r = extract_t(t_r)
        t_rf = extract_t(t_rf)

        optimizer = torch.optim.Adam(self.parameters())
        epochs = 30 if DEVICE == 'cpu' else 500
        j = [0, 2]
        self.use_sobel = True
        for i in range(epochs):
            y = self.forward(X)
            y_f = self.forward(X_f)
            y_r = self.forward(X_r, resize)
            y_rf = self.forward(X_rf, resize)

            weights = torch.softmax(torch.rand(4, device=DEVICE), dim=0)
            weights /= torch.mean(weights)
            weights = weights * (1 - (i / (epochs - 1)) ** 0.5) + (i / (epochs - 1)) ** 0.5
            loss1 = self.get_loss(y, t, latitude_weight, land, j)
            loss2 = self.get_loss(y_f, t_f, latitude_weight_f, land_f, j)
            loss3 = self.get_loss(y_r, t_r, latitude_weight_r, land_r, j)
            loss4 = self.get_loss(y_rf, t_rf, latitude_weight_rf, land_rf, j)

            loss = sum([weights[i] * (loss1[i] + loss2[i] + loss3[i] + loss4[i]) for i in range(len(loss1))])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'{i}: temp = {loss}')

    def save(self, path: str = '') -> None:
        torch.save(self.state_dict(), os.path.join(path, 'temperature-net.pt'))

    def load(self, path: str = '') -> None:
        self.load_state_dict(torch.load(os.path.join(path, 'temperature-net.pt'), map_location=DEVICE))


def flip_targets(matrix: Tensor) -> Tensor:
    """Flip upside-down, but also flip seasons"""
    temp = torch.flip(torch.roll(matrix[:12], 6, dims=0), dims=(1, 2))
    prec = torch.flip(torch.roll(matrix[12:], 6, dims=0), dims=(1, 2))
    return torch.concatenate([temp, prec], dim=0)


def load_npy(path: str) -> Tensor:
    return torch.from_numpy(np.load(os.path.join(DATA_PATH, path))).float().to(DEVICE)


def train() -> TemperatureNet:
    X, X_f = load_npy('x.npy'), load_npy('x-flipped.npy')
    X_r, X_rf = load_npy('x-retrograde.npy'), load_npy('x-retrograde-flipped.npy')
    t, t_r = load_npy('t.npy'), load_npy('t-retrograde.npy')
    t_f, t_rf = flip_targets(t), flip_targets(t_r)

    model = TemperatureNet()
    print(f'Training on {sum(p.numel() for p in model.parameters())} parameters...')
    model.fit([X, X_f, X_r, X_rf], [t, t_f, t_r, t_rf])

    X = load_npy('x.npy')
    land = X[2].numpy()
    result = model.predict(X.numpy())

    t = load_npy('t.npy').numpy()
    error = np.mean(result[:12] - t[:12], axis=0)

    return model

