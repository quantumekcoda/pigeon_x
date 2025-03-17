import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from typing import List, Optional, Tuple
import torch.nn.utils as utils

data = pd.read_csv('C:/FowlPlay/pigeon_data.csv')

pigeon_features = [
    'wing_flap',
    'coo_intensity',
    'seed_savor',
    'urban_flock',
    'sky_exposure',
    'peck_rate',
    'feather_entropy',
    'flock_size',
    'pigeon_speed',
    'perch_preference'
]

data.dropna(subset=pigeon_features + ['class'], inplace=True)
train_size = int(0.7 * len(data))
train_data = data.iloc[:train_size].copy()
test_data = data.iloc[train_size:].copy()

train_data['squawk_probability'] = train_data.groupby('flock_size')['class'].transform('mean')
bin_means = train_data.groupby('flock_size')['squawk_probability'].mean()
test_data['squawk_probability'] = test_data['flock_size'].map(bin_means)
test_data['squawk_probability'] = test_data['squawk_probability'].fillna(bin_means.mean())
final_data = pd.concat([train_data, test_data], ignore_index=True)

X_train = train_data[pigeon_features].values
y_train = train_data['squawk_probability'].values
X_test = test_data[pigeon_features].values
y_test = test_data['squawk_probability'].values

print("Training XGBoost Pigeon Model... squawk, squawk!")
xgb_pigeon = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_pigeon.fit(X_train, y_train)
preds_xgb = xgb_pigeon.predict(X_test)
mae_xgb = mean_absolute_error(y_test, preds_xgb)
mse_xgb = mean_squared_error(y_test, preds_xgb)
print(f"XGBoost Pigeon Model Evaluation -> MAE: {mae_xgb:.6f}, MSE: {mse_xgb:.6f}")
xgb_model_path = 'C:/FowlPlay/xgb_pigeon_model.pkl'
with open(xgb_model_path, 'wb') as f:
    pickle.dump(xgb_pigeon, f)
print(f"XGBoost Pigeon Model saved at: {xgb_model_path}")

class PigeonDeepNet(nn.Module):
    def __init__(self, input_dim: int):
        super(PigeonDeepNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def pigeon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    coo_penalty = torch.mean(torch.clamp(target - pred, min=0)) * 0.1
    return mse + coo_penalty

def genetic_algorithm_train(model: PigeonDeepNet,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            population_size: int = 20,
                            generations: int = 30,
                            mutation_rate: float = 0.05,
                            elite_frac: float = 0.3) -> float:
    original_params = utils.parameters_to_vector(model.parameters())
    param_size = original_params.numel()
    population = [original_params + torch.randn_like(original_params) * 0.1 for _ in range(population_size)]
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    best_loss = float('inf')
    best_individual = None
    for gen in range(generations):
        losses = []
        for individual in population:
            utils.vector_to_parameters(individual, model.parameters())
            outputs = model(X_tensor)
            loss = pigeon_loss(outputs, y_tensor)
            losses.append(loss.item())
        sorted_indices = np.argsort(losses)
        elite_count = int(population_size * elite_frac)
        elites = [population[i] for i in sorted_indices[:elite_count]]
        gen_best_loss = min(losses)
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_individual = population[sorted_indices[0]].clone()
        print(f"Generation {gen + 1}/{generations}, Best Pigeon Loss: {gen_best_loss:.6f}")
        new_population = elites.copy()
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(elites, 2, replace=False)
            crossover_point = np.random.randint(0, param_size)
            child = torch.cat([parent1[:crossover_point], parent2[crossover_point:]])
            child += torch.randn_like(child) * mutation_rate
            new_population.append(child)
        population = new_population
    utils.vector_to_parameters(best_individual, model.parameters())
    return best_loss

X_train_deep = train_data[pigeon_features].values
y_train_deep = train_data['squawk_probability'].values
input_dim = len(pigeon_features)
pigeon_deep_net = PigeonDeepNet(input_dim)
print("\n--- Training Pigeon Deep Net with the Genetic Algorithm ---")
best_loss = genetic_algorithm_train(pigeon_deep_net, X_train_deep, y_train_deep,
                                    population_size=20, generations=30,
                                    mutation_rate=0.05, elite_frac=0.3)
print(f"Best Pigeon Loss after GA training: {best_loss:.6f}")
pigeon_net_path = 'C:/FowlPlay/pigeon_deep_net.pkl'
with open(pigeon_net_path, 'wb') as f:
    pickle.dump(pigeon_deep_net.state_dict(), f)
print(f"Pigeon Deep Net saved at: {pigeon_net_path}")
