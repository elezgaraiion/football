"""
Offline tracklet stitching: reconnect track IDs that the online tracker split.

The online tracker (BoT-SORT / ByteTrack) keeps a lost track alive for `track_buffer`
frames and re-associates it with the predicted position (Kalman filter + IoU). When a
player is missed for longer, or the prediction drifts (a sprint, a camera pan), the
player comes back with a new ID. Because the whole clip is available, this module repairs
those splits afterwards: a track that ends and another one that starts shortly after, at a
compatible position and size, are joined into one identity.

A track A is linked to a track B when:
  1. Time: A ends before B starts (they never share a frame) and the gap is at most
     `max_gap_s` seconds.
  2. Position: B's first foot point (bottom-centre of the box) is close to A's last one,
     after compensating camera motion with the median displacement of the tracks visible
     in both frames. Distance is measured in body heights (box height), so it means the
     same for near and far players, and the tolerance grows with the gap.
  3. Size: the box heights at the end of A and the start of B are similar.
  4. No ambiguity: if another, conflicting link is almost as plausible (e.g. two players
     close together at a set piece), neither is made. A wrong merge is worse than a split.

Links are accepted greedily from the closest match. Each track gets at most one
predecessor and one successor, so identities form chains that never overlap in time.

Image-space positions cannot bridge long absences (a player out of shot for several
seconds); that needs appearance/team cues or pitch coordinates.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


@dataclass
class StitchConfig:
    max_gap_s: float = 3.0         # longest gap (s) between the end of one track and the start of the next
    base_dist: float = 0.5         # allowed distance at zero gap, in body heights
    speed: float = 2.0             # extra allowed distance per second of gap, in body heights (~3.6 m/s for 1.8 m)
    max_height_ratio: float = 1.5  # max ratio between box heights at the end of A and the start of B
    ambiguity_ratio: float = 0.7   # keep a link only if its distance is < this fraction of a conflicting one's...
    ambiguity_margin: float = 0.3  # ...and at least this many body heights shorter (~0.5 m)
    edge_frames: int = 5           # frames used to estimate box height at a track's start / end
    min_common_tracks: int = 3     # tracks needed in both frames to estimate camera motion


@dataclass
class Link:
    from_id: int     # raw track ID that ends
    to_id: int       # raw track ID that starts later and continues it
    gap_frames: int
    distance: float  # in body heights, after camera-motion compensation


@dataclass
class _Track:
    tid: int
    frames: list[int] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]


def _foot(box: list[float]) -> tuple[float, float]:
    x1, _, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def _height(boxes: list[list[float]]) -> float:
    return max(statistics.median(b[3] - b[1] for b in boxes), 1.0)


def _overlap(a: _Track, b: _Track) -> bool:
    return a.start <= b.end and b.start <= a.end


def _collect(frames: list[dict]) -> tuple[dict[int, _Track], dict[int, dict[int, tuple[float, float]]]]:
    tracks: dict[int, _Track] = {}
    feet_by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    for fr in sorted(frames, key=lambda f: f["frame"]):
        feet = feet_by_frame.setdefault(fr["frame"], {})
        for p in fr["players"]:
            track = tracks.setdefault(p["track_id"], _Track(p["track_id"]))
            track.frames.append(fr["frame"])
            track.boxes.append(p["bbox_xyxy"])
            feet[p["track_id"]] = _foot(p["bbox_xyxy"])
    return tracks, feet_by_frame


def find_links(frames: list[dict], fps: float, cfg: StitchConfig | None = None) -> list[Link]:
    """Links between raw track IDs that most likely belong to the same person."""
    cfg = cfg or StitchConfig()
    tracks, feet = _collect(frames)
    max_gap = cfg.max_gap_s * fps
    by_start = sorted(tracks.values(), key=lambda t: t.start)
    shift_cache: dict[tuple[int, int], tuple[float, float]] = {}

    def camera_shift(f0: int, f1: int) -> tuple[float, float]:
        if (f0, f1) not in shift_cache:
            a, b = feet.get(f0, {}), feet.get(f1, {})
            common = a.keys() & b.keys()
            if len(common) < cfg.min_common_tracks:
                shift_cache[f0, f1] = (0.0, 0.0)
            else:
                shift_cache[f0, f1] = (statistics.median(b[t][0] - a[t][0] for t in common),
                                       statistics.median(b[t][1] - a[t][1] for t in common))
        return shift_cache[f0, f1]

    # 1. Candidate links that pass the time, size and position gates
    candidates: list[Link] = []
    for a in tracks.values():
        ax, ay = _foot(a.boxes[-1])
        a_h = _height(a.boxes[-cfg.edge_frames:])
        for b in by_start:
            gap = b.start - a.end
            if gap <= 0:
                continue
            if gap > max_gap:
                break
            b_h = _height(b.boxes[:cfg.edge_frames])
            if max(a_h, b_h) > cfg.max_height_ratio * min(a_h, b_h):
                continue
            dx, dy = camera_shift(a.end, b.start)
            bx, by = _foot(b.boxes[0])
            distance = math.hypot(bx - (ax + dx), by - (ay + dy)) / ((a_h + b_h) / 2.0)
            if distance <= cfg.base_dist + cfg.speed * gap / fps:
                candidates.append(Link(a.tid, b.tid, gap, round(distance, 3)))

    # 2. Drop ambiguous candidates. A rival is a link that cannot coexist with this one:
    #    another continuation of A that overlaps B in time, or another predecessor of B that
    #    overlaps A. (Non-overlapping alternatives can still be part of the same chain.)
    def rivals(c: Link):
        for o in candidates:
            if o is c:
                continue
            if o.from_id == c.from_id and _overlap(tracks[o.to_id], tracks[c.to_id]):
                yield o
            elif o.to_id == c.to_id and _overlap(tracks[o.from_id], tracks[c.from_id]):
                yield o

    unambiguous = [
        c for c in candidates
        if all(c.distance < cfg.ambiguity_ratio * o.distance and o.distance - c.distance >= cfg.ambiguity_margin
               for o in rivals(c))
    ]

    # 3. Greedy acceptance, closest first; at most one successor and one predecessor per track
    links: list[Link] = []
    has_succ: set[int] = set()
    has_pred: set[int] = set()
    for c in sorted(unambiguous, key=lambda c: (c.distance, c.gap_frames)):
        if c.from_id in has_succ or c.to_id in has_pred:
            continue
        links.append(c)
        has_succ.add(c.from_id)
        has_pred.add(c.to_id)
    return sorted(links, key=lambda l: (l.from_id, l.to_id))


def stitch_tracks(frames: list[dict], fps: float, cfg: StitchConfig | None = None) -> tuple[list[dict], list[Link]]:
    """Return (frames with stitched IDs, links made).

    Each player record keeps the tracker's ID in `raw_track_id`; `track_id` becomes the ID
    of the first track in its chain.
    """
    links = find_links(frames, fps, cfg)
    successor = {l.from_id: l.to_id for l in links}
    has_pred = {l.to_id for l in links}

    raw_ids = {p["track_id"] for fr in frames for p in fr["players"]}
    final_id: dict[int, int] = {}
    for root in raw_ids - has_pred:
        tid = root
        while tid is not None:
            final_id[tid] = root
            tid = successor.get(tid)

    stitched = [
        {**fr, "players": [{"track_id": final_id[p["track_id"]], "raw_track_id": p["track_id"],
                            **{k: v for k, v in p.items() if k != "track_id"}}
                           for p in fr["players"]]}
        for fr in frames
    ]
    return stitched, links
