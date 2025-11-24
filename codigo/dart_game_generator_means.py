
# -*- coding: utf-8 -*-
"""
dart_game_generator.py — example using the extended means-enabled API.
"""

import dart_shots_means as ds
import dartslab as dl
# Optional visualization utilities if you already have them:
# import dartslab as dl

# Configuration with custom means (bias) per axis
cfg = ds.GenConfig(
    n = 5,
    seed = 1234,
    # Player A: independent Gaussian with biased mean (aims slightly up-right)
    A_sigma_x = 18, A_sigma_y = 16,
    A_mu_x = 5,  A_mu_y = 3,
    # Player B: correlated Gaussian, centered left/down
    B_sigma = 15, B_rho = 0.8,
    B_mu_x = -8, B_mu_y = -12,
    # Player C: Laplace heavy tails with vertical bias
    C_bx = 10, C_by = 10,
    C_mu_x = 0, C_mu_y = 15
)

# Generate datasets
paths = ds.generate_datasets(
    outdir = "./darts_data_5_shots_novice",
    prefix = "darts",
    cfg = cfg,
    write_meta = True
)

print("Generated files:")
for k, p in paths.items():
    print(f"{k}: {p}")

# If you have plotting helpers, you can load & plot here.
datasets = dl.load_many(paths)
fig, ax = dl.scatter_all(datasets)
