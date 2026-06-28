import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import Config
from levers.multi_asset import basket_positions
cfg = Config.load("config/default.yaml")
seeds = tuple(range(cfg.jump_model.n_seeds))
t0 = time.time()
bp = basket_positions(cfg, cfg.levers.multi_asset, seeds)
Path("results/levers").mkdir(parents=True, exist_ok=True)
pd.DataFrame(bp).to_csv("results/levers/positions_basket.csv")
print("done", len(bp), "assets in %.0fs" % (time.time()-t0))
