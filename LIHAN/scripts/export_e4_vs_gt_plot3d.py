"""E4 冻结预测与方向光 GT 的三维单位球对比(只读 per_frame.csv,输出单张 PNG)。"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv); d = d[d.method == "E4"].sort_values("frame")
P = d[["pred_x", "pred_y", "pred_z"]].values; G = d[["gt_x", "gt_y", "gt_z"]].values; err = d.angular_error_deg.values
norm = colors.Normalize(0, 10); cmap = cm.viridis_r

u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi/2:30j]
SX, SY, SZ = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)

fig = plt.figure(figsize=(20, 7.5))
views = [("Perspective", 22, -55), ("Top (looking down +Z)", 90, -90), ("Side (looking along Y)", 8, -90)]
for i, (title, elev, azim) in enumerate(views):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.plot_surface(SX, SY, SZ, color="lightgray", alpha=.10, linewidth=0, shade=False)
    ax.plot_wireframe(SX, SY, SZ, color="gray", alpha=.15, linewidth=.3, rstride=5, cstride=5)
    t = np.linspace(0, 2*np.pi, 200); ax.plot(np.cos(t), np.sin(t), 0*t, c="gray", lw=.8)
    for e in (40, 50):  # 先验仰角带边界
        r, z = np.cos(np.radians(e)), np.sin(np.radians(e)); ax.plot(r*np.cos(t), r*np.sin(t), z + 0*t, c="tab:blue", lw=.6, ls="--", alpha=.6)
    segs = np.stack([G, P], axis=1)  # 每帧 GT→E4 的连线,颜色=角误差
    ax.add_collection3d(Line3DCollection(segs, colors=cmap(norm(err)), linewidths=0.9, alpha=.7))
    ax.plot(*G.T, c="k", lw=2.2, label="GT"); ax.plot(*P.T, c="tab:red", lw=2.2, label="E4")
    ax.scatter(*G[0], c="k", s=60, marker="o"); ax.scatter(*P[0], c="tab:red", s=60, marker="o")
    ax.scatter(*G[-1], c="k", s=60, marker="s"); ax.scatter(*P[-1], c="tab:red", s=60, marker="s")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0.5, 0.9); ax.set_box_aspect((1, 1, .45)); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([0.5, 0.7, 0.9])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.view_init(elev, azim); ax.set_title(title)
    if i == 0: ax.legend(loc="upper left")
sm = cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, ax=fig.axes, shrink=.7, pad=.02); cb.set_label("per-frame angular error (deg)")
fig.suptitle(f"E4 vs GT on unit sphere  |  mean {err.mean():.2f} deg, max {err.max():.2f} deg, <=5 deg: {(err<=5).sum()}/{len(err)}   (circle = frame 1, square = frame 150, dashed = 40/50 deg prior band)", fontsize=12, y=0.97)
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out, err.mean())
