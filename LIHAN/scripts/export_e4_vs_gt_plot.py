"""E4 冻结预测与方向光 GT 对比可视化(只读已有评测 per_frame.csv,不重新拟合)。"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Droid Sans Fallback"]:
    if any(f in x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]; break
plt.rcParams["axes.unicode_minus"] = False

csv, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv)
def get(m):
    d = df[df.method == m].sort_values("frame")
    p = d[["pred_x", "pred_y", "pred_z"]].values; g = d[["gt_x", "gt_y", "gt_z"]].values
    return d.frame.values, p, g, d.angular_error_deg.values
def az(v): return np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 0])))
def el(v): return np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
f, p4, g, e4 = get("E4"); _, p3, _, e3 = get("E3")

fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
a = ax[0, 0]; a.plot(f, az(g), "k-", lw=2, label="GT"); a.plot(f, az(p3), "-", c="tab:orange", lw=1, alpha=.7, label="E3")
a.plot(f, az(p4), "-", c="tab:red", lw=1.6, label="E4"); a.set_title("Azimuth (unwrapped)"); a.set_xlabel("frame"); a.set_ylabel("deg"); a.legend(); a.grid(alpha=.3)
a = ax[0, 1]; a.plot(f, el(g), "k-", lw=2, label="GT"); a.plot(f, el(p3), "-", c="tab:orange", lw=1, alpha=.7, label="E3")
a.plot(f, el(p4), "-", c="tab:red", lw=1.6, label="E4"); a.axhspan(40, 50, color="gray", alpha=.12, label="prior band 40-50 deg")
a.set_title(f"Elevation (E4-GT mean {np.mean(el(p4)-el(g)):+.2f} deg)"); a.set_xlabel("frame"); a.set_ylabel("deg"); a.legend(); a.grid(alpha=.3)
a = ax[1, 0]; a.plot(f, e3, c="tab:orange", lw=1, alpha=.7, label=f"E3 mean {e3.mean():.2f} deg"); a.plot(f, e4, c="tab:red", lw=1.6, label=f"E4 mean {e4.mean():.2f} deg")
a.axhline(5, c="g", ls="--", label="5 deg pass line"); a.set_title(f"Per-frame angular error (E4 <=5 deg: {(e4<=5).sum()}/{len(e4)}, max {e4.max():.2f} deg)")
a.set_xlabel("frame"); a.set_ylabel("deg"); a.legend(); a.grid(alpha=.3)
a = ax[1, 1]; a.remove(); a = fig.add_subplot(2, 2, 4, projection="polar")
a.plot(np.radians(az(g)), 90 - el(g), "k-", lw=2, label="GT"); a.plot(np.radians(az(p4)), 90 - el(p4), "r-", lw=1.4, label="E4")
a.scatter(np.radians(az(g))[0], 90 - el(g)[0], c="k", s=40, zorder=5); a.scatter(np.radians(az(p4))[0], 90 - el(p4)[0], c="r", s=40, zorder=5)
a.set_title("Top view (radius = 90 - elevation, dot = frame 1)"); a.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
fig.suptitle("E4 frozen prediction vs directional-light GT", fontsize=15); fig.tight_layout(); fig.savefig(out, dpi=160)
print("saved", out, "E4 mean", e4.mean())
