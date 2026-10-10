"""
Identity assignment: one ID per player from the moment they appear, kept through duels and
short absences by position continuity and team.

The online tracker (BoT-SORT) is right most of the time. It goes wrong in two ways: when two
players cross it can hand the attacker's ID to the defender, and after an occlusion it often
starts a new ID. This module re-assigns identities over the whole clip, frame by frame, and
treats the tracker as the reference that only clear evidence can overrule:

  - Each identity remembers where its player was last seen, the tracker ID he had, how he was
    moving (camera motion removed) and his team (confidence-weighted majority of his labels).
  - In every frame, detections are matched one-to-one to identities (optimal assignment).
    Keeping the tracker's ID is always preferred, even after a box jump or a frame without
    detections. Distance to the predicted position (in body heights, so near and far players
    are treated alike) decides only between other options.
  - A detection labelled with the other team's kit costs an identity a strong penalty. So when
    the tracker swaps attacker and defender, the team-consistent assignment wins and both keep
    their IDs; a same-team swap (irrelevant for offside) is left as the tracker made it.
  - When the tracker loses a player and later starts a new ID for him, the player gets his
    identity back if he reappears within `max_gap_s` near where he was expected, never with the
    other team's kit, and only if no other option is almost as plausible: a new ID is safer
    than a wrong one.
  - Camera motion is estimated in every frame from the tracker's own correspondences (median
    displacement of the tracker IDs present in consecutive frames).

Input records need `bbox_xyxy`, `raw_track_id` and, optionally, a team label (`label_key`)
with its confidence (`weight_key`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:   # scipy is optional: fall back to a greedy assignment
    linear_sum_assignment = None

TEAMS = ("A", "B")
_INVALID = 1e6


@dataclass
class IdentityConfig:
    max_gap_s: float = 3.0          # how long an unseen player keeps their identity (seconds)
    base_dist: float = 1.0          # allowed distance from the prediction in the next frame (body heights)
    speed: float = 2.0              # extra allowed distance per second unseen (body heights; ~3.6 m/s)
    max_height_ratio: float = 1.5   # max ratio between box heights of the same player (new tracker IDs)
    tracker_bonus: float = 0.5      # preference for keeping the tracker's ID (body heights)...
    crowd_bonus: float = 0.15       # ...reduced to this when another player is predicted within crowd_dist:
    crowd_dist: float = 1.0         #    enough to ignore box noise, small enough for motion to decide a crossing
    rival_penalty: float = 2.0      # cost of giving an identity a detection with the other team's kit
    other_penalty: float = 0.5      # cost when a team label meets a referee/goalkeeper ("other") label
    ambiguity_margin: float = 0.3   # a player recovered under a new tracker ID needs a clearly best match
    velocity_frames: int = 12       # max frames of velocity extrapolation while unseen
    velocity_smoothing: float = 0.5 # weight of the previous velocity when updating it
    min_votes: float = 3.0          # label weight needed before an identity's team is used
    min_common: int = 3             # tracker IDs needed in two consecutive frames to estimate camera motion


@dataclass
class _Identity:
    id: int
    foot: np.ndarray
    h: float
    last_frame: int
    cam: np.ndarray
    raw: int
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    votes: Counter = field(default_factory=Counter)
    updates: int = 0

    def team(self, min_votes: float) -> str | None:
        if sum(self.votes.values()) < min_votes:
            return None
        return max(self.votes, key=lambda t: (self.votes[t], t in TEAMS))


def _foot(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, box[3]])


def _height(box) -> float:
    return max(box[3] - box[1], 1.0)


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        pairs = zip(rows.tolist(), cols.tolist())
    else:
        pairs, used_r, used_c = [], set(), set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), cost.shape[1])
            if r not in used_r and c not in used_c:
                pairs.append((r, c))
                used_r.add(r)
                used_c.add(c)
    return [(r, c) for r, c in pairs if cost[r, c] < _INVALID / 2]


def assign_identities(frames: list[dict], fps: float, cfg: IdentityConfig | None = None,
                      label_key: str = "team_frame", weight_key: str = "team_weight") -> tuple[list[dict], dict]:
    """Return (frames with `track_id` set to the player identity, events).

    Identities are numbered 1, 2, ... in order of first appearance. `events` lists the tracker
    ID changes that were overruled to keep a player's identity ("tracker_switches") and the
    players recovered under a new tracker ID after disappearing ("reidentifications").
    """
    cfg = cfg or IdentityConfig()
    frames = [{**fr, "players": [dict(p) for p in fr["players"]]} for fr in sorted(frames, key=lambda f: f["frame"])]
    max_gap = cfg.max_gap_s * fps
    identities: list[_Identity] = []
    cam = np.zeros(2)
    prev_feet: dict[int, np.ndarray] = {}
    events: dict[str, list] = {"tracker_switches": [], "reidentifications": []}

    for fr in frames:
        t, dets = fr["frame"], fr["players"]
        feet = [_foot(p["bbox_xyxy"]) for p in dets]
        hs = [_height(p["bbox_xyxy"]) for p in dets]
        labels = [p.get(label_key) for p in dets]
        raws = [p["raw_track_id"] for p in dets]

        # Camera motion from the tracker's own frame-to-frame correspondences
        cur = dict(zip(raws, feet))
        common = prev_feet.keys() & cur.keys()
        if len(common) >= cfg.min_common:
            cam = cam + np.median([cur[r] - prev_feet[r] for r in common], axis=0)
        if cur:
            prev_feet = cur

        active = [i for i in identities if t - i.last_frame <= max_gap]
        assigned: dict[int, _Identity] = {}
        if dets and active:
            n_i, n_d = len(active), len(dets)
            gaps = np.array([t - i.last_frame for i in active])
            preds = np.stack([i.foot + (cam - i.cam) + i.vel * min(g, cfg.velocity_frames)
                              for i, g in zip(active, gaps)])
            ih = np.array([i.h for i in active])
            dist = np.linalg.norm(preds[:, None, :] - np.stack(feet)[None], axis=2) / ((ih[:, None] + np.array(hs)[None]) / 2)
            allowed = cfg.base_dist + cfg.speed * (gaps - 1) / fps
            same_raw = np.array([[raws[d] == i.raw for d in range(n_d)] for i in active])
            recent = gaps <= 2
            cost = np.full((n_i, n_d), _INVALID)
            for a, ident in enumerate(active):
                team_i = ident.team(cfg.min_votes)
                for d in range(n_d):
                    if same_raw[a, d]:
                        others_close = recent & (dist[:, d] < cfg.crowd_dist)
                        others_close[a] = False
                        bonus = cfg.crowd_bonus if others_close.any() else cfg.tracker_bonus
                        c = min(dist[a, d], allowed[a]) - bonus   # the tracker's continuity
                    else:
                        if dist[a, d] > allowed[a]:
                            continue
                        if max(ident.h, hs[d]) > cfg.max_height_ratio * min(ident.h, hs[d]):
                            continue
                        c = dist[a, d]
                    lab = labels[d]
                    if team_i and lab and team_i != lab:
                        if team_i in TEAMS and lab in TEAMS:
                            if gaps[a] > 1 and not same_raw[a, d]:
                                continue   # never recover a player from the other team
                            c += cfg.rival_penalty
                        else:
                            c += cfg.other_penalty
                    cost[a, d] = c

            pairs = _assign(cost)
            matched_i = {a for a, _ in pairs}
            matched_d = {d for _, d in pairs}
            for a, d in pairs:
                ident = active[a]
                if not same_raw[a, d]:
                    if gaps[a] > 1:   # recovered under a new tracker ID: only with a clearly best match
                        extra = [cost[b, d] - cost[a, d] for b in range(n_i) if b != a and b not in matched_i]
                        extra += [cost[a, e] - cost[a, d] for e in range(n_d) if e != d and e not in matched_d]
                        extra += [cost[a, e] + cost[b, d] - cost[a, d] - cost[b, e] for b, e in pairs
                                  if b != a and cost[a, e] < _INVALID / 2 and cost[b, d] < _INVALID / 2]
                        if any(x < cfg.ambiguity_margin for x in extra if x < _INVALID / 2):
                            continue
                        events["reidentifications"].append({"frame": t, "id": ident.id, "gap_frames": int(gaps[a]),
                                                            "distance": round(float(dist[a, d]), 3),
                                                            "tracker_ids": [ident.raw, raws[d]]})
                    else:
                        events["tracker_switches"].append({"frame": t, "id": ident.id,
                                                           "tracker_ids": [ident.raw, raws[d]]})
                assigned[d] = ident

        for d, p in enumerate(dets):
            ident = assigned.get(d)
            if ident is None:   # a new player
                ident = _Identity(id=len(identities) + 1, foot=feet[d], h=hs[d], last_frame=t, cam=cam.copy(),
                                  raw=raws[d])
                identities.append(ident)
            else:
                gap = t - ident.last_frame
                own = ((feet[d] - ident.foot) - (cam - ident.cam)) / gap
                cap = 0.4 * hs[d]                       # no faster than ~0.4 body heights per frame
                speed = np.linalg.norm(own)
                if speed > cap:
                    own = own * cap / speed
                s = cfg.velocity_smoothing if ident.updates else 0.0
                ident.vel = s * ident.vel + (1 - s) * own
                ident.foot, ident.h, ident.last_frame = feet[d], 0.7 * ident.h + 0.3 * hs[d], t
                ident.cam, ident.raw = cam.copy(), raws[d]
                ident.updates += 1
            if labels[d]:
                ident.votes[labels[d]] += float(p.get(weight_key, 1.0) or 0.0)
            p["track_id"] = ident.id
    return frames, events
