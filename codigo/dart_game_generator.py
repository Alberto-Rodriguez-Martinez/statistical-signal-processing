# -*- coding: utf-8 -*-
"""

Created on Thu Oct 16 08:19:33 2025

dart_game_generator.py

example of use for dart_shots.py 
used to generate shots and games

@author: alrom
"""

import dart_shots as ds
import dartslab as dl

# Create a configuration object (all arguments are optional)
"""
If you omit it, dartslab will use the default values 
(500 samples per player, ρ = 0.75, etc.).
"""
cfg = ds.GenConfig(
    n = 600,        # number of throws per player
    seed = 1234,    # random seed for reproducibility
    A_sigma_x = 20, # std deviation of X for player A
    A_sigma_y = 20, # std deviation of Y for player A
    B_sigma   = 15, # std deviation (both axes) for player B
    B_rho     = 0.8,# correlation for player B
    C_bx      = 10, # Laplace scale parameter X for player C
    C_by      = 10  # Laplace scale parameter Y for player C
)

# Generate datasets and save them into ./data folder
paths = ds.generate_datasets(
    outdir = "./darts_data",    # output directory (created if missing)
    prefix = "darts",     # prefix for file names
    cfg = cfg,            # configuration defined above
    write_meta = True     # also write a JSON file with parameters
)

"""
After running this cell, you’ll find the following files inside ./darts_data/:
darts_A.csv       # Player A: independent Gaussian errors
darts_B.csv       # Player B: correlated Gaussian errors
darts_C.csv       # Player C: heavy-tailed Laplace errors
darts_meta.json   # (optional) parameters used for generation
"""

datasets = dl.load_many(paths)
for name, df in datasets.items():
    print(name, df.shape)
    print(df.head())
 
#%%
# Show all players together on one dartboard
fig, ax = dl.scatter_all(datasets)

# or Plot each player separately
# limits = dl.compute_limits(datasets)
# for name, df in datasets.items():
#     dl.scatter_player(df, title=f"Player {name}", limits=limits)