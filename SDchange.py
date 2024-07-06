import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
import os

# ハイパーパラメータの設定
batch_size = 128
num_timesteps = 1000
epochs = 1
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_type = "mse"

# フォルダの作成
experiment_name = f"batch_size_{batch_size}_num_timesteps_{num_timesteps}_epochs_{epochs}_lr_{lr}_loss_{loss_type}"
experiment_dir = os.path.join("experiments", experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

# CSVデータの読み込み
data = pd.read_csv('change.csv')
labels = data['x'].values
data = data.drop(columns=['x']).values

# データを1x9から3x3に変換する関数
def transform_data(data):
    transformed_data = data.reshape(-1, 1, 3, 3)
    return transformed_data

# カスタムデータセットクラス
class CoinDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(transform_data(data), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# データローダーの設定
dataset = CoinDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 位置エンベディング関数の定義
def _pos_encoding(time_idx, output_dim, device = "gpu"):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)
    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))
    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(timesteps, output_dim, device='gpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v

import torch
import torch.nn as nn
import math

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y

class UNetCond(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 8, time_embed_dim)
        self.bot1 = ConvBlock(8, 16, time_embed_dim)
        self.up1 = ConvBlock(16 + 8, 8, time_embed_dim)
        self.out = nn.Conv2d(8, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        t = pos_encoding(timesteps, self.time_embed_dim)

        if labels is not None:
            t += self.label_emb(labels)

        x1 = self.down1(x, t)
        x = self.maxpool(x1)

        x = self.bot1(x, t)

        x = self.upsample(x)
        
        # Adjust shape of x1 and x to be the same
        if x.size(2) != x1.size(2) or x.size(3) != x1.size(3):
            diffY = x1.size(2) - x.size(2)
            diffX = x1.size(3) - x.size(3)
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, t)
        x = self.out(x)
        return x

def pos_encoding(timesteps, time_embed_dim):
    half_dim = time_embed_dim // 2
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / (half_dim - 1)))
    emb = emb.to(timesteps.device)  # Ensure emb is on the same device as timestep
    emb = timesteps[:, None] * emb[None, :]
    pos_enc = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return pos_enc

# 画像の保存関数
def save_plot(fig, filename):
    fig.savefig(os.path.join(experiment_dir, filename))

# データフレームの保存関数
def save_dataframe(df, filename):
    df.to_csv(os.path.join(experiment_dir, filename), index=False)

# Diffuserクラスの定義
class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='gpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        alpha_bar = alpha_bar.view(alpha_bar.size(0), 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, labels):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t, labels)  # add label embedding
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std

    def reverse_to_img(self, x):
        x = x * 1
#        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 3, 3), labels=None):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        if labels is None:
            labels = torch.randint(0, 1000001, (len(x),), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images, labels

# モデルの初期化
diffuser = Diffuser(num_timesteps, device=device)
model = UNetCond(num_labels=1000001).to(device)
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

# 訓練ループ
losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        labels = labels.to(device)
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t, labels)
        if loss_type == "mse":
            loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')

# 損失のプロットを保存
fig, ax = plt.subplots()
ax.plot(losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss over Epochs')
save_plot(fig, "losses.png")

# サンプル生成
images, labels = diffuser.sample(model)

# サンプル画像の表示と保存
def show_samples(images, labels, columns=3, image_size=(5, 5)):
    num_samples = len(images)
    rows = (num_samples + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(image_size[0] * columns, image_size[1] * rows))
    
    for i in range(num_samples):
        ax = axes[i // columns, i % columns]
        img_array = np.array(images[i])
        ax.imshow(img_array, cmap='gray')
        ax.set_title(f"Label: {labels[i]}", fontsize=8, pad=20)  # Adjust fontsize and pad for spacing
        ax.axis('off')
    
    plt.subplots_adjust(wspace=0.3, hspace=0.6)  # Adjust the spacing between subplots
    save_plot(fig, "sample_images.png")
        
show_samples(images, labels)

def save_images_to_dataframe(images, labels):
    # Flatten the image arrays and convert them to lists
    flattened_images = [np.array(image).flatten().tolist() for image in images]
    
    # Move labels to CPU if necessary
    if labels.is_cuda:
        labels = labels.cpu()
    
    # Convert labels to numpy
    labels_np = labels.numpy()
    
    # Create a DataFrame with the image data and labels
    data = {
        "label": labels_np,
        "image_data": flattened_images
    }
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    save_dataframe(df, "images_labels.csv")
    print(f"DataFrame saved")
    
# Save images and labels to DataFrame
save_images_to_dataframe(images, labels)

# CSVファイルからデータを読み取る関数
def load_images_labels_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    image_data = df['image_data'].apply(eval).values  # Convert string representation of list to actual list
    return labels, image_data

# 金額を計算する関数
def calculate_amount(image_data):
    # 10000円札、5000円札、1000円札、500円玉、100円玉、50円玉、10円玉、5円玉、1円玉
    denominations = [10000, 5000, 1000, 500, 100, 50, 10, 5, 1]
    return sum([d * n for d, n in zip(denominations, image_data)])

# 元のデータと生成されたデータの比較
def compare_data(labels, image_data):
    calculated_amounts = [calculate_amount(data) for data in image_data]
    differences = np.abs(labels - calculated_amounts)
    avg_difference = np.mean(differences)
    return avg_difference, calculated_amounts, differences

# 枚数制約の定義
max_counts = {
    10000: np.inf,  # 無制限
    5000: 1,
    1000: 4,
    500: 1,
    100: 4,
    50: 1,
    10: 4,
    5: 1,
    1: 4
}

# 制約違反率を計算する関数
def calculate_violation_rate(image_data):
    total_violations = 0
    total_checks = 0
    for data in image_data:
        for i, count in enumerate(data):
            denomination = list(max_counts.keys())[i]
            if count > max_counts[denomination]:
                total_violations += 1
            total_checks += 1
    violation_rate = total_violations / total_checks
    return violation_rate

# CSVファイルからデータを読み取る
labels, image_data = load_images_labels_from_csv(os.path.join(experiment_dir, "images_labels.csv"))

# 比較を行い結果を表示
avg_difference, calculated_amounts, differences = compare_data(labels, image_data)
print(f'Average difference between original labels and calculated amounts: {avg_difference}')

# R2, MSE, RMSEの計算
mse = mean_squared_error(labels, calculated_amounts)
rmse = np.sqrt(mse)
r2 = r2_score(labels, calculated_amounts)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')

# y-yプロットの作成と保存
fig, ax = plt.subplots()
ax.scatter(labels, calculated_amounts, alpha=0.5)
ax.plot([min(labels), max(labels)], [min(labels), max(labels)], 'r--')
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')
ax.set_title('y-y Plot')
save_plot(fig, "yy_plot.png")

# 詳細な結果を表示
for i, (label, calc_amount, diff) in enumerate(zip(labels, calculated_amounts, differences)):
    print(f'Row {i}: Label = {label}, Calculated Amount = {calc_amount}, Difference = {diff}')

# 制約違反率を計算
violation_rate = calculate_violation_rate(image_data)
print(f'Violation Rate: {violation_rate}')

# 評価結果の保存
evaluation_results = {
    "MSE": mse,
    "RMSE": rmse,
    "R2 Score": r2,
    "Violation Rate": violation_rate
}
with open(os.path.join(experiment_dir, "evaluation_results.txt"), "w") as f:
    for key, value in evaluation_results.items():
        f.write(f"{key}: {value}\n")