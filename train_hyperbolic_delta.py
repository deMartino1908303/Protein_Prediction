import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import geoopt
import numpy as np
import pickle
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import f1_score, classification_report
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

# Set non-interactive backend to avoid crashes on some OS/IDE
matplotlib.use('Agg')


# --- CONFIGURATION ---
class Config:
    data_path = 'finalized_pooled_embeddings.pkl'

    epochs_trans = 100
    batch_size = 512
    hyp_lr = 1e-4  # Slightly higher for RiemannianAdam

    hyp_dim = 256
    hidden_dims = [512, 512]
    c = 1.0  # Curvature

    # Class Definitions
    interval = [-np.inf, -2, -0.5, 0.5, 2, np.inf]
    inter_names = ["Destroyed", "Deleterious", "Neutral", "Beneficial", "Improved"]
    # Colors for visualization (Cones)
    class_colors = ['#d62728', '#ff7f0e', '#7f7f7f', '#1f77b4', '#2ca02c']  # Red, Orange, Grey, Blue, Green
    output_dim = len(inter_names)

    device =  "cpu" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42


# --- UTILS ---
def discretize_fitness(fitness_array):
    bins = Config.interval
    labels = np.digitize(fitness_array, bins) - 1
    return labels


def compute_class_weights(dataset, device):
    labels = [sample[2].item() for sample in dataset]
    counts = np.bincount(labels, minlength=Config.output_dim)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * Config.output_dim
    return torch.tensor(weights, dtype=torch.float64, device=device)


# --- 1. DATASET (Robust WT Matching) ---
class PairedFeatureDataset(Dataset):
    def __init__(self, pkl_path):
        print(f"Loading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Handle multi-dim arrays in dict
        if isinstance(data, dict):
            if 'embedding' in data and isinstance(data['embedding'], np.ndarray):
                if data['embedding'].ndim > 1:
                    data['embedding'] = list(data['embedding'])

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # --- WT MATCHING LOGIC ---
        # 1. Clean names
        data['clean_name'] = data['file_name'].astype(str).str.strip()
        # 2. Extract family ID (first part of name)
        data['family_id'] = data['clean_name'].apply(lambda x: x.split('_')[0])
        # 3. Identify WT rows
        data['is_wt'] = data['clean_name'].str.contains('WT', case=False, regex=False)

        # Create WT Dictionary
        wt_groups = data[data['is_wt'] == True]
        wt_map = {}
        for _, row in wt_groups.iterrows():
            emb = torch.tensor(row['embedding'], dtype=torch.float64)
            wt_map[row['family_id']] = emb

        print(f"Found {len(wt_map)} unique Wild Type families.")

        if len(wt_map) == 0:
            raise ValueError("No Wild Type sequences found! Check filenames.")

        self.samples = []
        valid_count = 0

        for idx, row in data.iterrows():
            fam_id = row['family_id']
            if fam_id in wt_map:
                mut_emb = torch.tensor(row['embedding'], dtype=torch.float64)
                wt_emb = wt_map[fam_id]
                fitness = float(row['fitness'])
                label = discretize_fitness(np.array([fitness]))[0]

                self.samples.append((mut_emb, wt_emb, label, fitness))
                valid_count += 1

        print(f"Dataset created: {valid_count} paired samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# --- 2. HYPERBOLIC MODEL (With Embedding Extraction) ---
class HyperbolicDeltaMLR(nn.Module):
    def __init__(self, input_dim, hidden_dims, hyp_dim, num_classes, c=1.0):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.c = c

        # Encoder
        layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.1))
            in_d = h_dim
        self.encoder = nn.Sequential(*layers)

        # Tangent Projection
        self.to_tangent = nn.Linear(in_d, hyp_dim)

        # Class Prototypes (Directions on Manifold)
        self.weight = geoopt.ManifoldParameter(
            torch.randn(num_classes, hyp_dim) * 1e-2,
            manifold=self.manifold
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x_mut, x_wt, return_emb=False):
        x_mut = x_mut.double()
        x_wt = x_wt.double()

        # Encode & Project
        hyp_mut = self.manifold.expmap0(self.to_tangent(self.encoder(x_mut)))
        hyp_wt = self.manifold.expmap0(self.to_tangent(self.encoder(x_wt)))

        # Delta: Relative position of mutant w.r.t WT (WT becomes origin)
        # hyp_relative = -WT (+) Mut
        hyp_relative = self.manifold.mobius_add(-hyp_wt, hyp_mut)

        if return_emb:
            return hyp_relative

        # Classification (Hyperbolic MLR / "Hyper Loss")
        # Distance to class hyperplanes
        output = self.manifold.mobius_matvec(self.weight, hyp_relative)
        logits = self.manifold.logmap0(output) + self.bias

        return logits


# --- 3. VISUALIZATION FUNCTION ---
def visualize_poincare_cones(model, loader, device):
    """
    Projects high-dim hyperbolic embeddings to 2D Poincaré disk using PCA on tangent space.
    Visualizes the 'Cones' by plotting class centroids.
    """
    print("Generating Hyperbolic Visualization...")
    model.eval()

    # FIX: Ensure model is on the correct device
    model.to(device)

    embeddings = []
    labels_list = []

    # FIX: Removed hardcoded 'device = "cpu"' to respect the passed argument

    # 1. Extract Embeddings (Relative Positions)
    with torch.no_grad():
        for mut, wt, label, _ in loader:
            # Move inputs to the specified device (e.g., GPU)
            mut, wt = mut.to(device), wt.to(device)

            # Get hyp_relative (Computation happens on device)
            emb = model(mut, wt, return_emb=True)

            # Move result to CPU for storage/plotting to save GPU memory
            embeddings.append(emb.cpu())
            labels_list.append(label)

    X = torch.cat(embeddings, dim=0).double()
    y = torch.cat(labels_list, dim=0).numpy()

    # --- DEBUG START ---
    print("\n--- DEBUGGING PLOT ---")
    print(f"Total points to plot: {X.shape[0]}")
    print(f"Unique labels in validation: {np.unique(y)}")

    # Check if all embeddings are identical (Variance check)
    std_dev = torch.std(X, dim=0).mean().item()
    print(f"Average Standard Deviation of embeddings: {std_dev:.6f}")

    if std_dev < 1e-5:
        print("WARNING: MODEL COLLAPSE DETECTED. All embeddings are identical.")
    else:
        print("Embeddings are distinct (Good).")
    # --- DEBUG END ---
    # 2. Extract Class Weights (The "Cones")
    # Move to CPU for PCA/Plotting compatibility
    class_weights = model.weight.detach().cpu().double()

    # 3. Project to 2D for Visualization
    # Strategy: LogMap to Tangent -> PCA -> ExpMap to 2D Disk

    # Move to tangent space (Euclidean)
    manifold = model.manifold
    X_tan = manifold.logmap0(X).numpy()
    W_tan = manifold.logmap0(class_weights).detach().cpu().numpy()

    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(X_tan)

    X_tan_2d = pca.transform(X_tan)
    W_tan_2d = pca.transform(W_tan)

    # Map back to Poincare Disk (2D)
    # Use a temp 2D manifold for safe projection
    manifold_2d = geoopt.PoincareBall(c=model.c)
    X_hyp_2d = manifold_2d.expmap0(torch.tensor(X_tan_2d)).numpy()
    W_hyp_2d = manifold_2d.expmap0(torch.tensor(W_tan_2d)).numpy()

    # 4. PLOT
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Draw Poincaré Boundary
    circle = plt.Circle((0, 0), 1 / np.sqrt(model.c), color='black', fill=False, linewidth=2)
    ax.add_artist(circle)

    # Plot Points
    for i, class_name in enumerate(Config.inter_names):
        indices = np.where(y == i)
        plt.scatter(
            X_hyp_2d[indices, 0], X_hyp_2d[indices, 1],
            c=Config.class_colors[i], label=class_name, alpha=0.6, s=20
        )

    # Plot Class Centroids (The "Cones" directions)
    for i in range(len(Config.inter_names)):
        wx, wy = W_hyp_2d[i, 0], W_hyp_2d[i, 1]
        # Scale up arrow for visibility
        plt.arrow(0, 0, wx * 0.9, wy * 0.9, head_width=0.03, head_length=0.05,
                  fc='black', ec='black', linewidth=2, zorder=5)

    # Origin (Wild Type)
    plt.scatter(0, 0, c='black', marker='x', s=100, label='WT Origin', zorder=10)

    # plt.xlim(-1.1, 1.1)
    # plt.ylim(-1.1, 1.1)
    plt.legend(loc='upper right')
    plt.title("Hyperbolic Delta Representation (2D Projection)")
    plt.axis('off')

    # Save image ONLY (Do not call show())
    output_file = 'hyperbolic_cones.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory

    print(f"Visualization successfully saved to '{output_file}'")


# --- 4. PIPELINE ---
def train_delta_pipeline():
    full_dataset = PairedFeatureDataset(Config.data_path)

    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

    class_weights = compute_class_weights(train_dataset, Config.device)
    feat_dim = train_dataset[0][0].shape[0]

    print("\n--- Training Hyperbolic Delta Classifier (Hyper Loss) ---")
    torch.set_default_dtype(torch.float64)

    model = HyperbolicDeltaMLR(
        input_dim=feat_dim,
        hidden_dims=Config.hidden_dims,
        hyp_dim=Config.hyp_dim,
        num_classes=Config.output_dim,
        c=Config.c
    ).to(Config.device)

    # RiemannianAdam automatically handles the hyperbolic gradients
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=Config.hyp_lr)

    # Hyperbolic Multi-class Logistic Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1_m = -1.0

    for epoch in range(Config.epochs_trans):
        model.train()
        total_loss = 0.0

        for mut, wt, labels, _ in train_loader:
            mut, wt = mut.to(Config.device), wt.to(Config.device)
            labels = labels.to(Config.device)

            optimizer.zero_grad()
            logits = model(mut, wt)  # Forward calculates Delta & MLR
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for mut, wt, labels, _ in val_loader:
                    mut, wt = mut.to(Config.device), wt.to(Config.device)
                    logits = model(mut, wt)
                    preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_true.append(labels.numpy())

            y_p = np.concatenate(all_preds)
            y_t = np.concatenate(all_true)
            f1_m = f1_score(y_t, y_p, average='macro', zero_division=0)

            print(f"Epoch {epoch + 1:03d} | Loss: {total_loss / len(train_loader):.4f} | Val F1(Macro): {f1_m:.4f}")

            if f1_m > best_f1_m:
                best_f1_m = f1_m
                torch.save(model.state_dict(), 'best_hyp_model.pth')

    print(f"\nBest Validation F1 Macro: {best_f1_m:.4f}")

    return model


if __name__ == "__main__":
    # 1. Train
    trained_model = train_delta_pipeline()

    # 2. Carica tutto il dataset per la visualizzazione dettagliata
    full_dataset = PairedFeatureDataset(Config.data_path)
    visualization_loader = DataLoader(full_dataset, batch_size=Config.batch_size, shuffle=False)
    # 3. Genera plot per famiglie
    visualize_poincare_cones(trained_model, visualization_loader, Config.device)
