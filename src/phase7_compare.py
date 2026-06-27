"""
Phase 7 — does the constituent Absorption Ratio change the result?

Horses (all through the identical no-look-ahead engine, common 2018-start window,
CV-tuned lambda, t+1 execution, 2 bps cost), OOS 2020-2025:
    Buy & hold SPY
    JM-only
    AR-ETF only / AR-PIT only       (standalone absorption-ratio regime models)
    Fused-ETF  / Fused-PIT          (jump model + the two AR feeds)
Full metric set incl. Calmar, exposure, turnover, regime duration.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import core, engine, features

OUT = Path(__file__).resolve().parents[1] / "outputs"
_, spy, sec = core.load(); spy_ret = spy.pct_change()

f_jm, r, _ = features.build("jm")
f_fe, _, _ = features.build("fused_etf")
f_fp, _, _ = features.build("fused_pit")
ar_pit = features.build("ar_pit")[0]
ar_etf = core.ar_features(sec)
common = f_jm.index
for ix in (f_fe.index, f_fp.index, ar_pit.index, ar_etf.index):
    common = common.intersection(ix)

sets = {"JM-only": f_jm, "AR-ETF only": ar_etf, "AR-PIT only": ar_pit,
        "Fused-ETF": f_fe, "Fused-PIT": f_fp}


def full_metrics(ret, pos):
    ret = ret.dropna(); eq = (1 + ret).cumprod(); yrs = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dn = ret[ret < 0].std() * np.sqrt(252); mdd = (eq / eq.cummax() - 1).min()
    runs = (pos.diff().fillna(0) != 0).cumsum()
    m = dict(CAGR=cagr, AnnRet=ret.mean() * 252, Vol=ret.std() * np.sqrt(252),
             Sharpe=ret.mean() / ret.std() * np.sqrt(252) if ret.std() else np.nan,
             Sortino=ret.mean() * 252 / dn if dn else np.nan, MaxDD=mdd,
             Calmar=cagr / abs(mdd) if mdd else np.nan,
             Expo=pos.mean(), Switch=int((pos.diff() != 0).sum()),
             AvgDur=pos.groupby(runs).size().mean())
    return m

rows, curves = {}, {}
bh = spy_ret.loc["2020-01-01":]
rows["Buy & hold SPY"] = full_metrics(bh, pd.Series(1.0, index=bh.index))
curves["Buy & hold SPY"] = bh
for name, feat in sets.items():
    pos = engine.walk_forward(feat.loc[common], r, spy_ret, lam_mode="cv", seeds=range(3))
    ret = engine.pnl(pos, spy_ret)
    rows[name] = full_metrics(ret, pos); curves[name] = ret

tbl = pd.DataFrame(rows).T[["CAGR", "AnnRet", "Vol", "Sharpe", "Sortino", "MaxDD",
                            "Calmar", "Expo", "Switch", "AvgDur"]]
fmt = tbl.copy()
for c in ["CAGR", "AnnRet", "Vol", "MaxDD", "Expo"]:
    fmt[c] = (fmt[c] * 100).round(1).astype(str) + "%"
for c in ["Sharpe", "Sortino", "Calmar", "AvgDur"]:
    fmt[c] = fmt[c].round(2)
print(fmt.to_string())
tbl.to_csv(OUT / "phase7_pit_metrics.csv")

fig, ax = plt.subplots(figsize=(13, 6.5))
col = {"Buy & hold SPY": "black", "JM-only": "tab:blue", "AR-ETF only": "tab:gray",
       "AR-PIT only": "tab:green", "Fused-ETF": "tab:orange", "Fused-PIT": "tab:purple"}
for k, rr in curves.items():
    eq = (1 + rr).cumprod()
    ax.plot(eq.index, eq.values, label=k, color=col[k],
            lw=1.8 if k in ("Fused-PIT", "Buy & hold SPY") else 1.1)
ax.set_yscale("log"); ax.legend()
ax.set_title("Phase 7 — constituent AR vs ETF AR vs baselines (OOS growth of $1, 2020-2025)")
plt.tight_layout(); plt.savefig(OUT / "phase7_pit_compare.png", dpi=110)
print("saved -> outputs/phase7_pit_compare.png")
