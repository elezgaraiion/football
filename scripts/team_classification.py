"""
Team classification from kit colour, made robust with trajectories.

Runs after tracking, before identity assignment (identity_tracking.py):

  1. torso_color() / shorts_color(): for every detection, the shirt descriptor
        [L, a, b, light %, dark %]
     (median CIELAB colour plus the soft share of light and dark pixels, sampled from the side
     strips of the torso between shoulder and hip keypoints) and the shorts colour [L, a, b]
     (between hips and knees). Grass pixels are removed; a body part covered by a player
     closer to the camera is not measured.

  2. TeamModel.fit(): the two kits are learnt from the clip itself (2-means on shirts with
     outlier pruning). Team "A" is the lighter kit. A shirt whose colour (a*, b*: hue and
     saturation, not lightness) fits neither kit is "other": referees, goalkeepers, staff.
     Safety check: the shirts must form two groups with a clear gap (TeamModel.separated).

  3. label_teams(): the per-frame team labels actually used. A single shirt colour can lie:
     a dark shirt with a big number in the other team's colour looks like the other team
     from behind. So instead of "nearest kit":
       a. detections are grouped into several appearance groups (shirt + shorts), e.g. dark
          from the front, dark from behind with a light number, light, light in the shade,
          referee...;
       b. groups are joined into teams using trajectories. While a player is isolated (no
          one else close, so the tracker cannot swap him with anybody), all his detections
          are the same person, hence the same team. If isolated players keep passing from
          group X to group Y (turning around), X and Y belong to the same team, whatever the
          colour of the number. A group seen in isolated segments of both teams (e.g. backs
          of both teams look alike) gets a low confidence;
       c. each isolated segment gets one team for all its detections, by a confidence-weighted
          majority. Elsewhere (in groups of players), a detection gets its appearance group's
          team only if that group is confident.
     This is universal: it does not assume any colour for numbers, sponsors or shorts.
     Safety check: if no split of the groups is clearly respected by the isolated players
     (normalised cut above max_ncut), the colours don't follow the players (e.g. lighting
     changes hitting everyone at once) and team labels are not used.

  4. vote_teams(): a player's team is the confidence-weighted majority of their labels.
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
    min_pixels: int = 20             # pixels needed to measure a colour
    side_share: float = 0.3          # width of each side strip of the torso that is sampled
    occlusion_max: float = 0.2       # skip if a player in front covers more than this share of the region
    light_l: float = 65.0            # L* around which a pixel counts as light...
    dark_l: float = 35.0             # ...and as dark
    share_softness: float = 10.0     # width (L* units) of the soft light/dark transition
    chroma_floor: float = 6.0        # minimum colour (a*, b*) radius of a kit
    min_samples: int = 50            # measured detections needed to learn the two kits
    min_radius: float = 8.0          # minimum radius of a kit in descriptor space
    ambiguity: float = 0.8           # no kit label when nearest distance > this x distance to the other kit
    max_valley_ratio: float = 0.5    # kits must be split by a clear gap: valley/peak density below this
    n_appearance: int = 8            # appearance groups (views, shade, referee...) before joining them into teams
    min_group_size: int = 30         # detections per appearance group, at least on average
    isolation_dist: float = 1.5      # isolated = nobody else's feet within this many body heights
    segment_min_len: int = 5         # isolated detections in a row needed to use a segment as evidence
    label_min_conf: float = 0.5      # appearance-group confidence needed to label a detection on its own
    segment_min_weight: float = 2.0  # confidence-weighted votes needed to label a whole isolated segment...
    segment_min_share: float = 0.7   # ...and the share of them that must agree
    max_ncut: float = 0.6            # trajectories must split the groups into two teams with a cut below this
    min_team_share: float = 0.15     # each team needs this share of the trajectory evidence
    min_votes: int = 3               # labelled frames needed to give a player a team


# Distance weights. Shirt [L, a, b, light %, dark %] + shorts [L, a, b]: lightness and the
# light/dark shares count half as much as colour, because shadows change them the most.
WEIGHTS = np.array([0.5, 1.0, 1.0, 0.5, 0.5])
FULL_WEIGHTS = np.array([0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 1. Colour measurements
# ---------------------------------------------------------------------------

def grass_hsv(frame: np.ndarray) -> tuple[float, float, float] | None:
    """Median hue, saturation and value of the grass in this frame (OpenCV HSV), or None."""
    small = np.ascontiguousarray(frame[::8, ::8])
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    green = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 60) & (hsv[:, 2] > 40)
    if green.sum() <= 100:
        return None
    g = hsv[green]
    return float(np.median(g[:, 0])), float(np.median(g[:, 1])), float(np.median(g[:, 2]))


def _overlap_area(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def _kp(kpts, i):
    return kpts[i] if kpts is not None and i < len(kpts) and kpts[i] else None


def torso_region(box, kpts) -> tuple[float, float, float, float] | None:
    """Rectangle on the shirt between shoulders (5, 6) and hips (11, 12); None without both shoulders."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ls, rs = _kp(kpts, 5), _kp(kpts, 6)
    if not (ls and rs):
        return None
    hips = [p for p in (_kp(kpts, 11), _kp(kpts, 12)) if p]
    sy = (ls[1] + rs[1]) / 2
    hy = sum(p[1] for p in hips) / len(hips) if hips else sy + 0.3 * h
    if hy <= sy:
        hy = sy + 0.3 * h
    xs = [ls[0], rs[0]] + [p[0] for p in hips]
    cx = (min(xs) + max(xs)) / 2
    half = 0.7 * max((max(xs) - min(xs)) / 2, 0.15 * w)   # trimmed so arms stay out
    trim = 0.15 * (hy - sy)                                 # away from the neck and the shorts
    return cx - half, sy + trim, cx + half, hy - trim


def shorts_region(box, kpts) -> tuple[float, float, float, float] | None:
    """Rectangle on the shorts: just below the hips (11, 12), towards the knees (13, 14)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    hips = [p for p in (_kp(kpts, 11), _kp(kpts, 12)) if p]
    if not hips:
        return None
    knees = [p for p in (_kp(kpts, 13), _kp(kpts, 14)) if p]
    hy = sum(p[1] for p in hips) / len(hips)
    ky = sum(p[1] for p in knees) / len(knees) if knees else hy + 0.25 * h
    if ky <= hy:
        ky = hy + 0.25 * h
    xs = [p[0] for p in hips]
    cx = (min(xs) + max(xs)) / 2
    half = max((max(xs) - min(xs)) / 2 + 0.05 * w, 0.12 * w)
    return cx - half, hy + 0.08 * (ky - hy), cx + half, hy + 0.45 * (ky - hy)


def _measure(frame, box, region, others, grass, cfg, sides: bool):
    """Pixels (CIELAB) of `region`, without grass and without parts hidden by a closer player."""
    area = max((region[2] - region[0]) * (region[3] - region[1]), 1e-6)
    for o in others:
        if o[3] > box[3] and _overlap_area(region, o) > cfg.occlusion_max * area:
            return None   # a player closer to the camera (feet lower in the image) covers it
    img_h, img_w = frame.shape[:2]
    rx1, ry1 = max(int(region[0]), 0), max(int(region[1]), 0)
    rx2, ry2 = min(int(math.ceil(region[2])), img_w), min(int(math.ceil(region[3])), img_h)
    if rx2 <= rx1 or ry2 <= ry1 or (rx2 - rx1) * (ry2 - ry1) < cfg.min_pixels:
        return None
    crop = np.ascontiguousarray(frame[ry1:ry2, rx1:rx2])
    if sides:   # numbers (back) and sponsors (front) sit in the middle: keep the side strips
        cw = crop.shape[1]
        side = int(round(cfg.side_share * cw))
        if side >= 1 and 2 * side < cw and 2 * side * crop.shape[0] >= cfg.min_pixels:
            crop = np.ascontiguousarray(np.concatenate([crop[:, :side], crop[:, cw - side:]], axis=1))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    if grass is None:
        is_grass = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] > 50) & (hsv[:, 2] > 40)
    else:
        gh, _, gv = grass
        # close to the grass in hue AND brightness: a bright yellow shirt is near in hue but much brighter
        is_grass = (np.abs(hsv[:, 0] - gh) <= 8) & (hsv[:, 1] > 40) & (np.abs(hsv[:, 2] - gv) <= 70)
    keep = ~is_grass
    if keep.sum() < cfg.min_pixels:
        return None
    return cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).reshape(-1, 3)[keep]


def torso_color(frame: np.ndarray, box, kpts, others, grass, cfg: TeamConfig | None = None) -> list[float] | None:
    """Shirt descriptor [L, a, b, light %, dark %], or None if not measurable.

    `others` are the boxes of the other detections in the same frame; `grass` is grass_hsv(frame).
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
    lab = _measure(frame, box, region, others, grass, cfg, sides=True)
    if lab is None:
        return None
    # Soft shares (sigmoid instead of a hard cut) so the descriptor changes smoothly with lighting
    light = float((1 / (1 + np.exp(-(lab[:, 0] - cfg.light_l) / cfg.share_softness))).mean() * 100)
    dark = float((1 / (1 + np.exp(-(cfg.dark_l - lab[:, 0]) / cfg.share_softness))).mean() * 100)
    return [round(float(v), 1) for v in np.median(lab, axis=0)] + [round(light, 1), round(dark, 1)]


def shorts_color(frame: np.ndarray, box, kpts, others, grass, cfg: TeamConfig | None = None) -> list[float] | None:
    """Median shorts colour [L, a, b], or None if not measurable (no hip keypoints, covered...)."""
    cfg = cfg or TeamConfig()
    region = shorts_region(box, kpts)
    if region is None:
        return None
    lab = _measure(frame, box, region, others, grass, cfg, sides=False)
    return None if lab is None else [round(float(v), 1) for v in np.median(lab, axis=0)]


# ---------------------------------------------------------------------------
# 2. The two kits (seeds, "other" and safety checks)
# ---------------------------------------------------------------------------

def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator, mask: np.ndarray | None = None,
            init: np.ndarray | None = None, n_init: int = 6, iters: int = 60):
    """k-means with optional missing values (mask = 0): distances use the available dims only."""
    if mask is None:
        mask = np.ones_like(X)
    fill = np.array([X[mask[:, j] > 0, j].mean() if (mask[:, j] > 0).any() else 0.0 for j in range(X.shape[1])])
    Xf = np.where(mask > 0, X, fill)
    scale = X.shape[1] / np.maximum(mask.sum(1, keepdims=True), 1)

    def d2(c):
        return (((Xf[:, None, :] - c[None]) ** 2) * mask[:, None, :]).sum(2) * scale

    best = None
    for start in ([init] if init is not None else [None] * n_init):
        if start is None:   # k-means++ initialisation
            idx = [int(rng.integers(len(X)))]
            for _ in range(1, k):
                dmin = d2(Xf[idx]).min(1)
                idx.append(int(rng.choice(len(X), p=dmin / dmin.sum())) if dmin.sum() > 0 else int(rng.integers(len(X))))
            start = Xf[idx]
        c = np.array(start, dtype=float)
        for _ in range(iters):
            labels = d2(c).argmin(1)
            new = c.copy()
            for j in range(k):
                m = labels == j
                if m.any():
                    w = mask[m]
                    new[j] = np.where(w.sum(0) > 0, (Xf[m] * w).sum(0) / np.maximum(w.sum(0), 1e-9), c[j])
            if np.allclose(new, c):
                break
            c = new
        labels = d2(c).argmin(1)
        sse = float(d2(c)[np.arange(len(X)), labels].sum())
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
    centers: np.ndarray        # (2, 5) shirt descriptor of kit A and kit B
    radius: np.ndarray         # (2,) radius of each kit, weighted units (kept for reference)
    chroma_radius: np.ndarray  # (2,) colour (a*, b*) radius of each kit, used to spot "other"
    sizes: list[int]           # measured detections assigned to each kit when fitting
    valley_ratio: float        # density between the two kits relative to their peaks (low = well separated)
    cfg: TeamConfig

    @classmethod
    def fit(cls, samples: list[list[float]], cfg: TeamConfig | None = None) -> "TeamModel | None":
        cfg = cfg or TeamConfig()
        if len(samples) < cfg.min_samples:
            return None
        X = np.asarray(samples, dtype=float) * WEIGHTS
        rng = np.random.default_rng(0)
        centers, labels, _ = _kmeans(X, 2, rng)
        inlier = np.ones(len(X), dtype=bool)
        for _ in range(4):   # drop shirts far from both kits (referees, goalkeepers) and refit
            d = np.linalg.norm(X - centers[labels], axis=1)
            thr = np.array([_robust_threshold(d[labels == k], cfg.min_radius) for k in range(2)])
            new_inlier = d <= thr[labels]
            if new_inlier.sum() < cfg.min_samples or (new_inlier == inlier).all():
                break
            inlier = new_inlier
            centers, _, _ = _kmeans(X[inlier], 2, rng, init=centers)
            labels = np.linalg.norm(X[:, None, :] - centers[None], axis=2).argmin(1)

        d = np.linalg.norm(X - centers[labels], axis=1)
        radius = np.array([_robust_threshold(d[inlier & (labels == k)], cfg.min_radius) for k in range(2)])
        dc = np.linalg.norm(X[:, 1:3] - centers[labels, 1:3], axis=1)
        chroma_radius = np.array([_robust_threshold(dc[inlier & (labels == k)], cfg.chroma_floor) for k in range(2)])
        sizes = [int((inlier & (labels == k)).sum()) for k in range(2)]
        order = np.argsort(-centers[:, 0])   # kit A = lighter
        return cls(centers=centers[order] / WEIGHTS, radius=radius[order], chroma_radius=chroma_radius[order],
                   sizes=[sizes[i] for i in order], valley_ratio=_valley_ratio(X[inlier], centers), cfg=cfg)

    @property
    def separated(self) -> bool:
        """True when the shirts form two groups with a clear gap between them."""
        return self.valley_ratio < self.cfg.max_valley_ratio

    def classify(self, desc) -> str | None:
        """Nearest-kit label of one shirt: "A", "B", "other" or None (ambiguous / not measured)."""
        if desc is None:
            return None
        x = np.asarray(desc, dtype=float)[:5]
        dc = np.linalg.norm(self.centers[:, 1:3] - x[1:3], axis=1)
        if (dc > self.chroma_radius).all():
            return "other"   # a colour neither kit has (referee, goalkeeper, staff)
        d = np.linalg.norm((self.centers - x) * WEIGHTS, axis=1)
        k = int(d.argmin())
        if d[k] > self.cfg.ambiguity * d[1 - k]:
            return None
        return TEAMS[k]

    def bgr(self, team: str) -> tuple[int, int, int]:
        """The learnt kit colour, for drawing."""
        lab = self.centers[TEAMS.index(team), :3].astype(np.float32).reshape(1, 1, 3)
        b, g, r = (cv2.cvtColor(lab, cv2.COLOR_Lab2BGR).reshape(3) * 255).clip(0, 255)
        return int(b), int(g), int(r)

    def to_dict(self) -> dict:
        out: dict = {t: {"lab": [round(float(v), 1) for v in self.centers[i, :3]],
                         "light_pct": round(float(self.centers[i, 3]), 1),
                         "dark_pct": round(float(self.centers[i, 4]), 1),
                         "radius": round(float(self.radius[i]), 2),
                         "chroma_radius": round(float(self.chroma_radius[i]), 2), "detections": self.sizes[i]}
                     for i, t in enumerate(TEAMS)}
        out["valley_ratio"] = round(self.valley_ratio, 3)
        return out


# ---------------------------------------------------------------------------
# 3. Team labels from appearance groups + trajectories
# ---------------------------------------------------------------------------

@dataclass
class AppearanceGroup:
    id: int
    size: int
    shirt: list[float]          # [L, a, b, light %, dark %]
    shorts: list[float] | None  # [L, a, b] (None if never measured)
    team: str                   # "A", "B" or "other"
    confidence: float           # 1 = trajectory evidence only ever links it to its team; 0 = 50/50
    evidence: float             # isolated co-occurrences behind that decision (0 = appearance only)

    def to_dict(self) -> dict:
        return {"id": self.id, "size": self.size, "shirt": [round(float(v), 1) for v in self.shirt],
                "shorts": None if self.shorts is None else [round(float(v), 1) for v in self.shorts],
                "team": self.team, "confidence": round(self.confidence, 3), "evidence": round(self.evidence, 1)}


def isolated_segments(frames: list[dict], cfg: TeamConfig | None = None) -> list[list[dict]]:
    """Runs of consecutive frames (gaps of 1-2 frames allowed) in which one tracker ID follows a
    player with nobody else's feet within `isolation_dist` body heights. The tracker can't swap
    players there, so every detection in a segment is the same person."""
    cfg = cfg or TeamConfig()
    by_raw: dict[int, list[tuple[int, dict, bool]]] = defaultdict(list)
    for fr in frames:
        players = fr["players"]
        feet = [((p["bbox_xyxy"][0] + p["bbox_xyxy"][2]) / 2, p["bbox_xyxy"][3]) for p in players]
        hs = [max(p["bbox_xyxy"][3] - p["bbox_xyxy"][1], 1.0) for p in players]
        for i, p in enumerate(players):
            iso = all(math.hypot(feet[i][0] - feet[j][0], feet[i][1] - feet[j][1]) / ((hs[i] + hs[j]) / 2)
                      > cfg.isolation_dist for j in range(len(players)) if j != i)
            by_raw[p.get("raw_track_id", p.get("track_id"))].append((fr["frame"], p, iso))
    segments = []
    for items in by_raw.values():
        items.sort(key=lambda x: x[0])
        cur, last = [], None
        for f, p, iso in items:
            if iso and cur and f - last <= 2:
                cur.append(p)
            else:
                if cur:
                    segments.append(cur)
                cur = [p] if iso else []
            last = f
        if cur:
            segments.append(cur)
    return segments


@dataclass
class TeamLabels:
    groups: list[AppearanceGroup]
    supported: bool                  # trajectories support a split into two teams
    ncut: float | None               # normalised cut of the chosen split (0 = no isolated player crosses it)
    reason: str | None               # why teams are not supported
    colors: dict                     # display shirt colour (CIELAB) of each team, from its main group


def lab_to_bgr(lab) -> tuple[int, int, int]:
    arr = np.asarray(lab[:3], dtype=np.float32).reshape(1, 1, 3)
    b, g, r = (cv2.cvtColor(arr, cv2.COLOR_Lab2BGR).reshape(3) * 255).clip(0, 255)
    return int(b), int(g), int(r)


def _best_split(M: np.ndarray, nodes: list[int], min_share: float) -> tuple[dict[int, int] | None, float]:
    """Two-way split of `nodes` minimising the normalised cut of the co-occurrence graph M
    (exhaustive: there are at most a few hundred splits). Both sides need at least `min_share`
    of the total co-occurrence volume."""
    sub = M[np.ix_(nodes, nodes)]
    vol = sub.sum(1)
    total = vol.sum()
    best, best_ncut = None, float("inf")
    for bits in range(1, 2 ** (len(nodes) - 1)):
        side = np.array([0] + [(bits >> (i - 1)) & 1 for i in range(1, len(nodes))])
        v0, v1 = vol[side == 0].sum(), vol[side == 1].sum()
        if min(v0, v1) < min_share * total:
            continue
        cut = sub[np.ix_(side == 0, side == 1)].sum()
        ncut = cut / v0 + cut / v1
        if ncut < best_ncut:
            best, best_ncut = {g: int(s) for g, s in zip(nodes, side)}, float(ncut)
    return best, best_ncut


def label_teams(frames: list[dict], model: TeamModel, cfg: TeamConfig | None = None) -> TeamLabels:
    """In place, for every detection: `appearance_group`, `team_frame` ("A"/"B"/"other"/None) and
    `team_weight` (confidence of that label, 0-1). Returns the groups and whether trajectories
    support two teams (if not, labels are left empty)."""
    cfg = cfg or TeamConfig()
    every = [p for fr in frames for p in fr["players"]]
    for p in every:
        p["appearance_group"], p["team_frame"], p["team_weight"] = None, None, 0.0
    dets = [p for p in every if p.get("torso_color")]
    if len(dets) < cfg.min_samples:
        return TeamLabels([], False, None, "not enough measurable shirts", {})

    # a. Appearance groups on shirt + shorts (shorts may be missing)
    X = np.array([p["torso_color"] + (p.get("shorts_color") or [0.0, 0.0, 0.0]) for p in dets]) * FULL_WEIGHTS
    mask = np.ones_like(X)
    mask[:, 5:] = np.array([[1.0 if p.get("shorts_color") else 0.0] for p in dets])
    k = int(min(cfg.n_appearance, max(2, len(dets) // cfg.min_group_size)))
    centers, labels, _ = _kmeans(X, k, np.random.default_rng(0), mask=mask)
    for p, g in zip(dets, labels):
        p["appearance_group"] = int(g)
    sizes = np.bincount(labels, minlength=k)
    shirts = centers[:, :5] / WEIGHTS
    has_shorts = np.array([mask[labels == g, 5].any() for g in range(k)])

    # Co-occurrence of groups within isolated segments (each segment of n detections weighs ~n)
    M = np.zeros((k, k))
    segments = isolated_segments(frames, cfg)
    for seg in segments:
        gs = [p["appearance_group"] for p in seg if p["appearance_group"] is not None]
        if len(gs) >= cfg.segment_min_len:
            n = np.bincount(gs, minlength=k).astype(float)
            M += (np.outer(n, n) - np.diag(n)) / (len(gs) - 1)
    self_links = np.array([M[g, g] for g in range(k)])

    # b. "other": a colour neither kit has (chroma) that trajectories don't tie to the players
    chroma_other = {g for g in range(k) if model.classify(shirts[g]) == "other"}
    other = {g for g in chroma_other
             if M[g, [h for h in range(k) if h not in chroma_other]].sum() <= M[g, list(chroma_other)].sum()}
    team_groups = [g for g in range(k) if g not in other]
    linked = [g for g in team_groups if M[g, team_groups].sum() > 0]

    # Teams = the split of groups that isolated players cross the least (normalised cut)
    side, ncut = (None, float("inf"))
    if len(linked) >= 2:
        side, ncut = _best_split(M, linked, cfg.min_team_share)
    if side is None or ncut > cfg.max_ncut:
        why = ("too few isolated players to learn which kit colours go together" if side is None and len(linked) < 2
               else f"trajectories don't separate two teams (best split cut {ncut:.2f} > {cfg.max_ncut}): the colour "
                    f"groups are mixed along the same players, as with lighting changes")
        return TeamLabels([], False, None if side is None else ncut, why, {})
    # Groups never seen in an isolated segment: the team of the most similar linked group
    for g in team_groups:
        if g not in side:
            nearest = min(linked, key=lambda h: np.linalg.norm(centers[g] - centers[h]))
            side[g] = side[nearest]

    def rowsum(g):
        return M[g].sum()

    # Team A = the lighter kit (mean shirt lightness of each side, weighted by size)
    light = [np.average([shirts[g, 0] for g in side if side[g] == s], weights=[sizes[g] for g in side if side[g] == s])
             for s in (0, 1)]
    name = {0: "A", 1: "B"} if light[0] >= light[1] else {0: "B", 1: "A"}

    groups = []
    for g in range(k):
        if g in other:
            t, conf = "other", 1.0
        else:
            t = name[side[g]]
            opp = sum(M[g, h] for h in side if side[h] != side[g])
            conf = 1.0 - 2.0 * opp / rowsum(g) if rowsum(g) > 0 else 0.5
        groups.append(AppearanceGroup(id=g, size=int(sizes[g]), shirt=shirts[g].tolist(),
                                      shorts=(centers[g, 5:] / FULL_WEIGHTS[5:]).tolist() if has_shorts[g] else None,
                                      team=t, confidence=float(np.clip(conf, 0.0, 1.0)),
                                      evidence=float(rowsum(g) - self_links[g])))

    # c. Labels: confident groups label detections; isolated segments get one team each
    for p in dets:
        grp = groups[p["appearance_group"]]
        p["team_weight"] = round(grp.confidence, 3)
        p["team_frame"] = grp.team if grp.confidence >= cfg.label_min_conf else None
    for seg in segments:
        votes: Counter = Counter()
        for p in seg:
            if p["appearance_group"] is not None:
                grp = groups[p["appearance_group"]]
                votes[grp.team] += grp.confidence
        total = sum(votes.values())
        if total >= cfg.segment_min_weight:
            team, w = max(votes.items(), key=lambda kv: (kv[1], kv[0] in TEAMS))
            if w / total >= cfg.segment_min_share:
                for p in seg:
                    p["team_frame"], p["team_weight"] = team, 1.0

    # Display colour of each team: its biggest confident group
    colors = {}
    for t in TEAMS:
        cands = [g for g in groups if g.team == t]
        if cands:
            colors[t] = max(cands, key=lambda g: g.size * g.confidence).shirt[:3]
    return TeamLabels(groups, True, ncut, None, colors)


# ---------------------------------------------------------------------------
# Player teams
# ---------------------------------------------------------------------------

def vote_teams(frames: list[dict], min_votes: int = 3, src: str = "team_frame", weight: str = "team_weight",
               key: str = "track_id") -> dict[int, str | None]:
    """Confidence-weighted majority of labels for every ID (None with fewer than `min_votes` labels)."""
    counts: dict[int, Counter] = defaultdict(Counter)
    n_labels: Counter = Counter()
    ids = set()
    for fr in frames:
        for p in fr["players"]:
            ids.add(p[key])
            if p.get(src):
                counts[p[key]][p[src]] += p.get(weight, 1.0) or 0.0
                n_labels[p[key]] += 1
    teams = {}
    for tid in ids:
        c = counts.get(tid)
        if not c or n_labels[tid] < min_votes or sum(c.values()) <= 0:
            teams[tid] = None
        else:
            teams[tid] = max(c, key=lambda t: (c[t], t in TEAMS))   # ties go to a team, not "other"
    return teams
