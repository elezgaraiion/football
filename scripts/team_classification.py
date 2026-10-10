"""
Team classification from shirt colour, and repair of ID swaps between rival players.

Runs after tracking and before tracklet stitching:
  1. torso_lab(): for every detection, the median shirt colour (CIELAB), sampled from the
     torso between the shoulder and hip keypoints with grass pixels removed. A detection
     whose torso is covered by a player closer to the camera is skipped, so the shirt of
     the player in front is never measured as the colour of the one behind.
  2. TeamModel.fit(): the two team colours are learnt from the clip itself (2-means with
     outlier pruning). Colours that fit neither team (referees, goalkeepers, staff) are
     classified as "other". Team "A" is the lighter kit.
  3. TeamModel.classify(): a label per detection and frame: "A", "B", "other", or None when
     the colour can't be measured or is too close to both teams.
  4. repair_team_swaps(): a sustained A <-> B change along one track means the tracker
     swapped identities. When two nearby tracks change in opposite directions at the same
     time (attacker <-> defender in a duel), their IDs are exchanged back from the swap
     frame on. A change without a partner is split into a new track instead.
  5. vote_teams(): each track's team is the majority of its per-frame labels.

Two safety checks decide whether the team labels are trusted at all: the colours must form
two groups with a clear gap (TeamModel.separated), and tracks must keep a consistent team
over time (track_purity). If either fails, the caller should skip the team steps.

Known limits: swaps between teammates can't be seen by colour (they don't matter for
offside); swaps involving a referee or goalkeeper are not repaired; a kit whose colour is
very close to the grass is removed together with the grass.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

TEAMS = ("A", "B")


@dataclass
class TeamConfig:
    min_pixels: int = 20           # shirt pixels needed to measure a colour
    occlusion_max: float = 0.2     # skip if a player in front covers more than this share of the torso
    lightness_weight: float = 0.5  # L* counts half as much as colour: shadows change lightness the most
    min_samples: int = 50          # measured detections needed to learn the two kits
    min_radius: float = 8.0        # minimum colour radius of a team (weighted CIELAB units)
    ambiguity: float = 0.8         # no label when distance to nearest team > this x distance to the other
    min_run: int = 5               # labelled frames in a row for a team change to count as sustained
    pair_max_gap_s: float = 1.0    # max time between two opposite changes to treat them as one swap
    swap_max_dist: float = 1.5     # max distance between the two players at the swap, in body heights
    min_votes: int = 3             # labelled frames needed to give a track a team
    max_valley_ratio: float = 0.5  # kits must be split by a clear gap: valley/peak density below this
    min_track_purity: float = 0.8  # share of each track's labels that must agree with its own majority team


# ---------------------------------------------------------------------------
# 1. Shirt colour
# ---------------------------------------------------------------------------

def grass_hue(frame: np.ndarray) -> float | None:
    """Dominant hue of the grass in this frame (OpenCV hue, 0-180), or None if not found."""
    small = np.ascontiguousarray(frame[::8, ::8])
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    green = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 60) & (hsv[:, 2] > 40)
    return float(np.median(hsv[green, 0])) if green.sum() > 100 else None


def _overlap_area(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def torso_region(box, kpts) -> tuple[float, float, float, float] | None:
    """Rectangle on the shirt between shoulders (5, 6) and hips (11, 12); None without both shoulders."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    def kp(i):
        return kpts[i] if kpts is not None and i < len(kpts) and kpts[i] else None

    ls, rs = kp(5), kp(6)
    if not (ls and rs):
        return None
    hips = [p for p in (kp(11), kp(12)) if p]
    sy = (ls[1] + rs[1]) / 2
    hy = sum(p[1] for p in hips) / len(hips) if hips else sy + 0.3 * h
    if hy <= sy:
        hy = sy + 0.3 * h
    xs = [ls[0], rs[0]] + [p[0] for p in hips]
    cx = (min(xs) + max(xs)) / 2
    half = 0.7 * max((max(xs) - min(xs)) / 2, 0.15 * w)   # trimmed so arms stay out
    trim = 0.15 * (hy - sy)                                 # away from the neck and the shorts
    return cx - half, sy + trim, cx + half, hy - trim


def torso_lab(frame: np.ndarray, box, kpts, others, grass_h: float | None,
              cfg: TeamConfig | None = None) -> list[float] | None:
    """Median shirt colour [L, a, b] of one detection, or None if it can't be measured reliably.

    `others` are the boxes of the other detections in the same frame.
    """
    cfg = cfg or TeamConfig()
    region = torso_region(box, kpts)
    if region is None:
        # No shoulders: use the upper-middle of the box, but only if no other player touches it
        if any(_overlap_area(box, o) > 0 for o in others):
            return None
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        region = (x1 + 0.3 * w, y1 + 0.2 * h, x2 - 0.3 * w, y1 + 0.45 * h)

    area = max((region[2] - region[0]) * (region[3] - region[1]), 1e-6)
    for o in others:
        if o[3] > box[3] and _overlap_area(region, o) > cfg.occlusion_max * area:
            return None   # a player closer to the camera (feet lower in the image) covers the shirt

    img_h, img_w = frame.shape[:2]
    rx1, ry1 = max(int(region[0]), 0), max(int(region[1]), 0)
    rx2, ry2 = min(int(math.ceil(region[2])), img_w), min(int(math.ceil(region[3])), img_h)
    if rx2 <= rx1 or ry2 <= ry1 or (rx2 - rx1) * (ry2 - ry1) < cfg.min_pixels:
        return None

    crop = np.ascontiguousarray(frame[ry1:ry2, rx1:rx2])
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    if grass_h is None:
        grass = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85)
    else:
        grass = np.abs(hsv[:, 0] - grass_h) <= 10
    grass &= (hsv[:, 1] > 50) & (hsv[:, 2] > 40)
    keep = ~grass
    if keep.sum() < cfg.min_pixels:
        return None

    lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).reshape(-1, 3)[keep]
    return [round(float(v), 1) for v in np.median(lab, axis=0)]


# ---------------------------------------------------------------------------
# 2-3. Team colours and per-frame labels
# ---------------------------------------------------------------------------

def _kmeans2(X: np.ndarray, rng: np.random.Generator, init: np.ndarray | None = None,
             n_init: int = 8, iters: int = 50):
    best = None
    for start in ([init] if init is not None else [None] * n_init):
        if start is None:   # k-means++ initialisation
            c0 = X[rng.integers(len(X))]
            d2 = ((X - c0) ** 2).sum(1)
            c1 = X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))]
            start = np.stack([c0, c1])
        c = np.array(start, dtype=float)
        for _ in range(iters):
            labels = ((X[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
            new = np.stack([X[labels == k].mean(0) if (labels == k).any() else c[k] for k in range(2)])
            if np.allclose(new, c):
                break
            c = new
        labels = ((X[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
        sse = float(((X - c[labels]) ** 2).sum())
        if best is None or sse < best[2]:
            best = (c, labels, sse)
    return best


def _robust_threshold(d: np.ndarray, floor: float) -> float:
    if len(d) == 0:
        return floor
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    return max(floor, med + 3 * 1.4826 * mad)


def _valley_ratio(X: np.ndarray, centers: np.ndarray) -> float:
    """Bimodality along the axis joining the two centres: lowest density between them divided
    by the smaller of the two peak densities. ~0 for two distinct kits, ~1 for one colour
    spread out (e.g. by lighting) and cut in half."""
    axis = centers[1] - centers[0]
    norm2 = float(axis @ axis)
    if norm2 < 1e-9:
        return 1.0
    t = ((X - centers[0]) @ axis) / norm2           # centre 0 -> 0, centre 1 -> 1
    hist, edges = np.histogram(t, bins=32, range=(-0.3, 1.3))
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    mids = (edges[:-1] + edges[1:]) / 2
    peak = min(smooth[mids <= 0.3].max(), smooth[mids >= 0.7].max())
    valley = smooth[(mids > 0.3) & (mids < 0.7)].min()
    return float(valley / peak) if peak > 0 else 1.0


@dataclass
class TeamModel:
    centers: np.ndarray   # (2, 3) CIELAB colour of team A and team B
    radius: np.ndarray    # (2,) colour radius of each team, weighted units
    weights: np.ndarray   # (3,) channel weights used for distances
    sizes: list[int]      # measured detections assigned to each team when fitting
    valley_ratio: float   # density between the two kits relative to their peaks (low = well separated)
    cfg: TeamConfig

    @classmethod
    def fit(cls, samples: list[list[float]], cfg: TeamConfig | None = None) -> "TeamModel | None":
        cfg = cfg or TeamConfig()
        if len(samples) < cfg.min_samples:
            return None
        w = np.array([cfg.lightness_weight, 1.0, 1.0])
        X = np.asarray(samples, dtype=float) * w
        rng = np.random.default_rng(0)
        centers, labels, _ = _kmeans2(X, rng)
        inlier = np.ones(len(X), dtype=bool)
        for _ in range(4):   # drop colours far from both teams (referees, goalkeepers) and refit
            d = np.linalg.norm(X - centers[labels], axis=1)
            thr = np.array([_robust_threshold(d[labels == k], cfg.min_radius) for k in range(2)])
            new_inlier = d <= thr[labels]
            if new_inlier.sum() < cfg.min_samples or (new_inlier == inlier).all():
                break
            inlier = new_inlier
            centers, _, _ = _kmeans2(X[inlier], rng, init=centers)
            labels = np.linalg.norm(X[:, None, :] - centers[None], axis=2).argmin(1)

        d = np.linalg.norm(X - centers[labels], axis=1)
        radius = np.array([_robust_threshold(d[inlier & (labels == k)], cfg.min_radius) for k in range(2)])
        sizes = [int((inlier & (labels == k)).sum()) for k in range(2)]
        order = np.argsort(-centers[:, 0])   # team A = lighter kit
        return cls(centers=centers[order] / w, radius=radius[order], weights=w,
                   sizes=[sizes[i] for i in order], valley_ratio=_valley_ratio(X[inlier], centers), cfg=cfg)

    @property
    def separated(self) -> bool:
        """True when the colours form two groups with a clear gap between them.

        Without a gap, the kits are too similar, or the two "teams" are just one colour under
        changing light, and team labels can't be trusted.
        """
        return self.valley_ratio < self.cfg.max_valley_ratio

    def classify(self, lab) -> str | None:
        if lab is None:
            return None
        x = np.asarray(lab, dtype=float) * self.weights
        d = np.linalg.norm(self.centers * self.weights - x, axis=1)
        k = int(d.argmin())
        if d[k] > self.radius[k]:
            return "other"
        if d[k] > self.cfg.ambiguity * d[1 - k]:
            return None
        return TEAMS[k]

    def bgr(self, team: str) -> tuple[int, int, int]:
        """The learnt kit colour, for drawing."""
        lab = self.centers[TEAMS.index(team)].astype(np.float32).reshape(1, 1, 3)
        b, g, r = (cv2.cvtColor(lab, cv2.COLOR_Lab2BGR).reshape(3) * 255).clip(0, 255)
        return int(b), int(g), int(r)

    def to_dict(self) -> dict:
        out: dict = {t: {"lab": [round(float(v), 1) for v in self.centers[i]],
                         "radius": round(float(self.radius[i]), 2), "detections": self.sizes[i]}
                     for i, t in enumerate(TEAMS)}
        out["valley_ratio"] = round(self.valley_ratio, 3)
        return out


# ---------------------------------------------------------------------------
# 4. Swap repair
# ---------------------------------------------------------------------------

@dataclass
class _Flip:
    tid: int
    from_team: str
    to_team: str
    start: int   # first frame after the last stable frame of the old team
    end: int     # first frame of the stable run of the new team


def _team_flips(frames: list[dict], min_run: int) -> list[_Flip]:
    seq: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for fr in frames:
        for p in fr["players"]:
            if p.get("team_frame") in TEAMS:
                seq[p["track_id"]].append((fr["frame"], p["team_frame"]))
    flips = []
    for tid, items in seq.items():
        runs: list[list] = []   # [team, first frame, last frame, count]
        for f, team in sorted(items):
            if runs and runs[-1][0] == team:
                runs[-1][2] = f
                runs[-1][3] += 1
            else:
                runs.append([team, f, f, 1])
        stable = [r for r in runs if r[3] >= min_run]
        for prev, nxt in zip(stable, stable[1:]):
            if prev[0] != nxt[0]:
                flips.append(_Flip(tid, prev[0], nxt[0], prev[2] + 1, nxt[1]))
    return flips


def _foot_dist(a, b) -> float:
    fa, fb = ((a[0] + a[2]) / 2, a[3]), ((b[0] + b[2]) / 2, b[3])
    h = max(((a[3] - a[1]) + (b[3] - b[1])) / 2, 1.0)
    return math.hypot(fa[0] - fb[0], fa[1] - fb[1]) / h


def _best_swap(flips: list[_Flip], boxes: dict, fps: float, cfg: TeamConfig):
    best = None
    max_gap = cfg.pair_max_gap_s * fps
    for i, fx in enumerate(flips):
        for fy in flips[i + 1:]:
            if fx.tid == fy.tid or fx.from_team != fy.to_team or fx.to_team != fy.from_team:
                continue
            lo, hi = max(fx.start, fy.start), min(fx.end, fy.end)
            if lo - hi > max_gap:
                continue
            span = range(lo, hi + 1) if lo <= hi else range(min(fx.start, fy.start), max(fx.end, fy.end) + 1)
            options = [(_foot_dist(boxes[fx.tid, f], boxes[fy.tid, f]), f)
                       for f in span if (fx.tid, f) in boxes and (fy.tid, f) in boxes]
            if not options:
                continue
            dist, frame = min(options)
            if dist > cfg.swap_max_dist:
                continue
            key = (max(0, lo - hi), dist)
            if best is None or key < best[0]:
                best = (key, fx, fy, frame, dist)
    return best


def repair_team_swaps(frames: list[dict], fps: float, cfg: TeamConfig | None = None):
    """Return (frames with corrected track_id, swaps repaired, splits made)."""
    cfg = cfg or TeamConfig()
    frames = [{**fr, "players": [dict(p) for p in fr["players"]]} for fr in frames]

    swaps = []
    for _ in range(1000):
        flips = _team_flips(frames, cfg.min_run)
        boxes = {(p["track_id"], fr["frame"]): p["bbox_xyxy"] for fr in frames for p in fr["players"]}
        best = _best_swap(flips, boxes, fps, cfg)
        if best is None:
            break
        _, fx, fy, frame, dist = best
        for fr in frames:
            if fr["frame"] >= frame:
                for p in fr["players"]:
                    if p["track_id"] == fx.tid:
                        p["track_id"] = fy.tid
                    elif p["track_id"] == fy.tid:
                        p["track_id"] = fx.tid
        swaps.append({"frame": frame, "track_ids": [fx.tid, fy.tid],
                      "teams": [f"{fx.from_team}->{fx.to_team}", f"{fy.from_team}->{fy.to_team}"],
                      "distance": round(dist, 3)})

    # Changes without a partner: split the track in the middle of the uncertain window
    all_ids = [p["track_id"] for fr in frames for p in fr["players"]]
    next_id = max(all_ids, default=0) + 1
    cuts: dict[int, list[tuple[int, int]]] = defaultdict(list)
    splits = []
    for fl in sorted(_team_flips(frames, cfg.min_run), key=lambda f: (f.tid, f.start)):
        cut = (fl.start + fl.end + 1) // 2
        cuts[fl.tid].append((cut, next_id))
        splits.append({"frame": cut, "track_id": fl.tid, "teams": f"{fl.from_team}->{fl.to_team}",
                       "new_track_id": next_id})
        next_id += 1
    for fr in frames:
        for p in fr["players"]:
            new_id = p["track_id"]
            for cut, nid in cuts.get(p["track_id"], []):
                if fr["frame"] >= cut:
                    new_id = nid
            p["track_id"] = new_id
    return frames, swaps, splits


# ---------------------------------------------------------------------------
# 5. Track teams
# ---------------------------------------------------------------------------

def track_purity(frames: list[dict], min_votes: int = 3) -> float:
    """How consistently tracks keep one team: labels agreeing with each track's majority team,
    over all team labels (A/B only; tracks with fewer than `min_votes` labels ignored).

    Close to 1 when the colours follow the players. Around 0.5 when the "teams" are really
    lighting changes that hit every player at once, which is a reason to distrust the model.
    """
    counts: dict[int, Counter] = defaultdict(Counter)
    for fr in frames:
        for p in fr["players"]:
            if p.get("team_frame") in TEAMS:
                counts[p["track_id"]][p["team_frame"]] += 1
    agree = total = 0
    for c in counts.values():
        n = sum(c.values())
        if n >= min_votes:
            agree += max(c.values())
            total += n
    return agree / total if total else 0.0


def vote_teams(frames: list[dict], min_votes: int = 3) -> dict[int, str | None]:
    """Majority of per-frame labels for every track ID (None with too few labelled frames)."""
    counts: dict[int, Counter] = defaultdict(Counter)
    ids = set()
    for fr in frames:
        for p in fr["players"]:
            ids.add(p["track_id"])
            if p.get("team_frame"):
                counts[p["track_id"]][p["team_frame"]] += 1
    teams = {}
    for tid in ids:
        c = counts.get(tid)
        if not c or sum(c.values()) < min_votes:
            teams[tid] = None
        else:
            teams[tid] = max(c, key=lambda t: (c[t], t in TEAMS))   # ties go to a team, not "other"
    return teams
