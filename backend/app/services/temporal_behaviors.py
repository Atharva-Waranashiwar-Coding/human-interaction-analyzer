"""Geometry-only review suggestions from the existing timestamped pose stream.

No detector, frame reader, persistence, or appearance-based identity matching lives here.
All distances are in body heights; histories end at gaps, unsafe jumps or camera motion.
"""
from __future__ import annotations

from collections import deque
from itertools import combinations, product
import math
import os

ARMS = ((5, 7, 9), (6, 8, 10))
CAUTION = " Geometry-only review suggestion; contact and action are not confirmed."


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def norm(a):
    return math.hypot(*a)


def cosine(a, b):
    length = norm(a) * norm(b)
    return dot(a, b) / length if length > 1e-8 else -1.


def valid(p, *indices):
    k = p.get("pose", ())
    return all(i < len(k) and len(k[i]) >= 3 and all(math.isfinite(float(v)) for v in k[i][:3])
               and k[i][2] >= .6 for i in indices)


def torso(p):
    if not valid(p, 5, 6, 11, 12):
        return None
    k = p["pose"]
    return tuple(sum(float(k[j][i]) for j in (5, 6, 11, 12)) / 4 for i in (0, 1))


def arm(p, indices):
    if not valid(p, *indices):
        return None
    s, e, w = (p["pose"][i][:2] for i in indices)
    upper, lower = sub(e, s), sub(w, e)
    h = p["height"]
    if min(norm(upper), norm(lower)) < .06 * h:
        return None
    return s, e, w, cosine(upper, lower)


def reversals(values, threshold):
    """Count meaningful excursions, ignoring subthreshold keypoint jitter."""
    if not values:
        return 0
    anchor, direction, extreme, count = values[0], 0, values[0], 0
    for value in values[1:]:
        if not direction:
            if abs(value - anchor) >= threshold:
                direction = 1 if value > anchor else -1
                extreme = value
        elif direction * (value - extreme) >= 0:
            extreme = value
        elif direction * (extreme - value) >= threshold:
            count += 1
            direction *= -1
            extreme = value
    return count


class TemporalSuggestions:
    def __init__(self, fps, mode="all"):
        if mode not in {"all","relative","trajectory"}:raise ValueError("Invalid temporal mode")
        self.mode=mode
        self.fps = fps
        self.window = float(os.getenv("BEHAVIOR_WINDOW_SECONDS", "4"))
        self.hold = max(.6, float(os.getenv("BEHAVIOR_MIN_SECONDS", "1")))
        self.merge_gap = float(os.getenv("BEHAVIOR_MERGE_GAP_SECONDS", ".4"))
        if not 3 <= self.window <= 10 or not 0 <= self.merge_gap <= 1 or not .6 <= self.hold <= self.window:
            raise ValueError("Temporal window must be 3..10s, minimum duration .6..window, merge gap 0..1s")
        self.history = {}
        self.epochs = {}
        self.events = {}
        self.last_t = None

    def _drop(self, pid):
        self.history.pop(pid, None)
        self.epochs[pid] = self.epochs.get(pid, 0) + 1

    def step(self, t, people, camera_motion=False):
        if not math.isfinite(t) or t < 0 or (self.last_t is not None and t <= self.last_t):
            for pid in list(self.history):
                self._drop(pid)
            return
        self.last_t = t
        for pid in list(self.history):
            if pid not in people or camera_motion:
                self._drop(pid)
        if camera_motion:
            return
        for pid, p in people.items():
            h = float(p.get("height", 0))
            center = p.get("center", ())
            if not math.isfinite(h) or h <= 0 or len(center) != 2 or not all(math.isfinite(float(x)) for x in center):
                self._drop(pid)
                continue
            history = self.history.get(pid)
            if history:
                previous_t, previous = history[-1]
                if (t - previous_t > 1.5 / self.fps or
                    math.dist(center, previous["center"]) / h > .65 or
                    max(h, previous["height"]) / min(h, previous["height"]) > 1.6):
                    self._drop(pid)
            # Copy only small pose/track data; never retain a video frame.
            snapshot = dict(center=tuple(center), height=h, pose=[tuple(k) for k in p.get("pose", ())])
            history = self.history.setdefault(pid, deque())
            history.append((t, snapshot))
            while history and t - history[0][0] > self.window:
                history.popleft()
        if self.mode!="trajectory":
            for pid in sorted(self.history):self._gestures(pid)
        for a, b in combinations(sorted(self.history), 2):
            pair = self._pair(a, b)
            if len(pair) < 3:
                continue
            _, p, q = pair[-1]
            if math.dist(p["center"], q["center"]) > 3 * (p["height"] + q["height"]) / 2:
                continue
            if self.mode!="trajectory":
                self._handshake(a,b,pair)
                self._hug(a,b,pair)
            if self.mode!="relative":
                self._following(a,b,pair)
                self._following(b,a,[(t,q,p) for t,p,q in pair])
                self._push(a,b,pair)
                self._push(b,a,[(t,q,p) for t,p,q in pair])
        if self.mode!="relative":self._groups()

    def _tail(self, history, seconds=None):
        end = history[-1][0]
        return [x for x in history if end - x[0] <= (seconds or self.hold) + 1e-6]

    def _enough(self, samples, duration=None):
        return len(samples) >= 3 and samples[-1][0] - samples[0][0] >= (duration or self.hold) - 1e-6

    def _pair(self, a, b):
        right = {t: p for t, p in self.history[b]}
        matched = [(t, p, right[t]) for t, p in self.history[a] if t in right]
        continuous = []
        for t, p, q in matched:
            if max(p["height"], q["height"]) / min(p["height"], q["height"]) > 1.5:
                continuous = []
            else:
                continuous.append((t, p, q))
        return continuous

    def _emit(self, label, participants, start, end, evidence):
        if end <= start:
            return
        participants = tuple(participants)
        # Epochs prohibit merging across occlusion, camera movement and track jumps.
        key = label, participants, tuple(self.epochs.get(p, 0) for p in participants)
        events = self.events.setdefault(key, [])
        if events and start <= events[-1]["end_time"] + self.merge_gap:
            event = events[-1]
            event["start_time"] = round(min(start, event["start_time"]), 4)
            event["end_time"] = round(max(end, event["end_time"]), 4)
        else:
            events.append(dict(start_time=round(start, 4), end_time=round(end, 4), behavior=label,
                               participants=list(participants), description=evidence + CAUTION,
                               source="local_model", confidence=None, review_status="needs_review"))

    def _recent(self, label, participants, t):
        key = label, tuple(participants), tuple(self.epochs.get(p, 0) for p in participants)
        events = self.events.get(key)
        return bool(events and t - events[-1]["end_time"] <= 1.5 / self.fps)

    def _gestures(self, pid):
        history = list(self.history[pid])
        for indices in ARMS:
            wave = self._tail(history, max(1.6, self.hold))
            arms = [arm(p, indices) for _, p in wave]
            if self.fps >= 4 and self._enough(wave) and all(arms):
                xs = [(a[2][0] - a[0][0]) / p["height"] for a, (_, p) in zip(arms, wave)]
                raised = all(a[2][1] < a[0][1] + .03 * p["height"] and a[2][1] < a[1][1]
                             for a, (_, p) in zip(arms, wave))
                turns = reversals(xs, .07)
                if raised and turns >= 2 and max(xs) - min(xs) >= .15:
                    self._emit("waving", [pid], wave[0][0], wave[-1][0],
                               f"Raised wrist oscillated laterally relative to shoulder: {turns} direction reversals, excursion {max(xs)-min(xs):.2f} body heights.")
            point = self._tail(history)
            if not self._enough(point):
                continue
            # A suggestion requires a visible person along the extended arm's ray.
            for other in sorted(self.history):
                if other == pid:
                    continue
                targets = dict(self.history[other])
                directions = []
                for t, p in point:
                    a = arm(p, indices)
                    q = targets.get(t)
                    target = torso(q) if q else None
                    if not a or target is None:
                        break
                    s, _, w, straightness = a
                    reach = sub(w, s)
                    towards = sub(target, w)
                    if (straightness < .94 or norm(reach) < .3 * p["height"] or
                        abs(reach[1]) > .6 * abs(reach[0]) or cosine(reach, towards) < .92 or
                        norm(towards) > 2 * p["height"] or norm(towards) < .1 * p["height"]):
                        break
                    directions.append(reach)
                if len(directions) == len(point) and all(cosine(directions[0], d) > .96 for d in directions):
                    self._emit("pointing", [pid, other], point[0][0], point[-1][0],
                               "Sustained nearly straight lateral arm, stable direction, with another person's torso along the arm ray; finger direction is unavailable.")

    def _handshake(self, a, b, pair):
        if self.fps < 4:
            return
        for ia, ib in product(ARMS, repeat=2):
            contact = []
            onset = None
            for t, p, q in pair:
                aa, ab = arm(p, ia), arm(q, ib)
                tp, tq = torso(p), torso(q)
                if aa is None or ab is None or tp is None or tq is None:
                    contact, onset = [], None
                    continue
                h = (p["height"] + q["height"]) / 2
                s, _, w, _ = aa
                z, _, v, _ = ab
                gap = math.dist(w, v) / h
                if gap > .25:
                    onset = t
                # Shoulder-to-hip axis handles rotated/overhead bodies; above-head high-fives are excluded.
                low = True
                for own, shoulder, wrist in ((p,s,w),(q,z,v)):
                    k = own["pose"]
                    axis = ((k[11][0]+k[12][0]-k[5][0]-k[6][0])/2,
                            (k[11][1]+k[12][1]-k[5][1]-k[6][1])/2)
                    projection = dot(sub(wrist,shoulder),axis) / max(1e-8,norm(axis)) / h
                    low = low and norm(axis) > .12*h and .05 < projection < .55
                towards = dot(sub(w, s), sub(tq, tp)) > 0 and dot(sub(v, z), sub(tp, tq)) > 0
                between = sub(tq, tp)
                midpoint = ((w[0]+v[0])/2, (w[1]+v[1])/2)
                projection = dot(sub(midpoint,tp),between) / max(1e-8,dot(between,between))
                close = .35 < math.dist(tp, tq) / h < 1.3 and gap < .12 and .15 < projection < .85
                established = onset is not None or self._recent("handshaking", [a,b], pair[-1][0])
                if not (low and towards and close and established):
                    contact = []
                    continue
                contact.append((t, ((w[0]-s[0])/h,(w[1]-s[1])/h), ((v[0]-z[0])/h,(v[1]-z[1])/h)))
            if self._enough(contact):
                for axis in (0,1):
                    ys, zs = [x[1][axis] for x in contact], [x[2][axis] for x in contact]
                    dy, dz = [y-x for x,y in zip(ys,ys[1:])], [y-x for x,y in zip(zs,zs[1:])]
                    correlated = sum(x*y for x,y in zip(dy,dz)) / max(1e-8, math.sqrt(sum(x*x for x in dy)*sum(y*y for y in dz)))
                    if reversals(ys, .025) >= 1 and max(ys)-min(ys) > .05 and correlated > .8:
                        self._emit("handshaking", [a,b], contact[0][0], contact[-1][0],
                                   "Previously separated wrists converged between the bodies, below shoulder level along each torso axis, with sustained correlated oscillation; above-head contact is excluded.")
                        break

    def _hug_pose(self, p, q):
        tp, tq = torso(p), torso(q)
        if tp is None or tq is None:
            return False
        h = (p["height"] + q["height"]) / 2
        if not .12 < math.dist(tp, tq) / h < .5:
            return False
        for own, target, center in ((p, q, tq), (q, p, tp)):
            wrapped = False
            for indices in ARMS:
                a = arm(own, indices)
                if a:
                    s, _, w, bend = a
                    shoulder_y = (target["pose"][5][1] + target["pose"][6][1]) / 2
                    if (bend < .8 and norm(sub(w, s)) > .12*h and math.dist(w, center) < .3*h and
                        shoulder_y - .08*h < w[1] < shoulder_y + .28*h and
                        dot(sub(w, s), sub(center, torso(own))) > 0):
                        wrapped = True
            if not wrapped:
                return False
        return True

    def _hug(self, a, b, pair):
        tail = self._tail(pair)
        if not self._enough(tail) or not all(self._hug_pose(p,q) for _,p,q in tail):
            return
        # Require a visible transition into reciprocal arm wrapping, not mere proximity.
        before = [x for x in pair if x[0] < tail[0][0]]
        if any(torso(p) is not None and torso(q) is not None and
               all(arm(r, idx) is not None for r in (p,q) for idx in ARMS) and
               not self._hug_pose(p,q) for _,p,q in before) or self._recent("hugging", [a,b], pair[-1][0]):
            self._emit("hugging", [a,b], tail[0][0], tail[-1][0],
                       "Close torsos with reciprocally bent arms reaching the other's upper torso, sustained after a visible non-wrapped pose.")

    def _following(self, follower, leader, pair):
        if len(pair) < 7 or pair[-1][0]-pair[0][0] < 2.4:
            return
        h = sum(p["height"]+q["height"] for _,p,q in pair) / (2*len(pair))
        leader_path = [(t,q["center"]) for t,_,q in pair]
        for lag in (.6, 1., 1.4):
            matches = []
            for t,p,q in pair:
                earlier_t, earlier = min(leader_path, key=lambda x: abs(x[0]-(t-lag)))
                if abs(earlier_t-(t-lag)) <= .55/self.fps:
                    matches.append((t,p["center"],earlier,q["center"]))
            if len(matches) < 5 or matches[-1][0]-matches[0][0] < 1.4:
                continue
            errors = [math.dist(p,earlier)/h for _,p,earlier,_ in matches]
            current_gap = math.dist(matches[-1][1], matches[-1][3])/h
            first, mid, last = matches[0][2], matches[len(matches)//2][2], matches[-1][2]
            v1, v2 = sub(mid, first), sub(last, mid)
            # A shared turn and delayed path agreement reject parallel straight walking.
            if (max(errors) < .2 and .4 < current_gap < 2 and min(norm(v1),norm(v2))/h > .2
                and -.2 < cosine(v1,v2) < .85 and
                cosine(sub(matches[-1][1],matches[-2][1]), sub(matches[-1][3],matches[-2][3])) > .8):
                self._emit("following", [follower,leader], matches[0][0], matches[-1][0],
                           f"First participant retraced the second's turning path about {lag:.1f}s later, within {max(errors):.2f} body heights; intent is unknown.")
                return

    def _push(self, actor, recipient, pair):
        if len(pair) < 6:
            return
        end_t, last_p, last_q = pair[-1]
        for j in range(2, len(pair)-2):
            t, p, q = pair[j]
            if not .4 <= end_t-t <= 1.2:
                continue
            before = next((x for x in reversed(pair[:j]) if t-x[0] >= .35), None)
            if before is None:
                continue
            bt, bp, bq = before
            tp,tq,bp_t,bq_t,ep,eq = (torso(r) for r in (p,q,bp,bq,last_p,last_q))
            if any(x is None for x in (tp,tq,bp_t,bq_t,ep,eq)):
                continue
            h = (p["height"]+q["height"])/2
            direction = sub(tq,tp)
            d = norm(direction)
            if not .3 < d/h < 1.2:
                continue
            unit = (direction[0]/d,direction[1]/d)
            after = dot(sub(eq,tq),unit)/h
            prior_speed = dot(sub(tq,bq_t),unit)/h/(t-bt)
            if (after < .25 or after/(end_t-t) < max(.2,prior_speed+.2) or
                norm(sub(ep,tp))/h > .18):
                continue
            for indices in ARMS:
                old_arm,new_arm = arm(bp,indices),arm(p,indices)
                if old_arm is None or new_arm is None:
                    continue
                s,_,w,bend = new_arm
                old_reach = sub(old_arm[2],old_arm[0])
                reach = sub(w,s)
                if (bend > .9 and bend-old_arm[3] > .3 and
                    dot(sub(reach,old_reach),unit)/h > .12 and math.dist(w,tq)/h < .18 and
                    cosine(reach,direction) > .9):
                    self._emit("pushing", [actor,recipient], bt,end_t,
                               f"First participant extended a bent arm toward the second's torso, followed by {after:.2f} body-height displacement away and increased recipient speed; causation is unknown.")
                    return

    def _groups(self):
        # Connected proximity groups avoid enumerating every possible person subset.
        remaining = set(self.history)
        while remaining:
            group = {min(remaining)}
            pending = list(group)
            remaining -= group
            while pending:
                p = self.history[pending.pop()][-1][1]
                for pid in sorted(remaining):
                    q = self.history[pid][-1][1]
                    h = (p["height"]+q["height"])/2
                    if math.dist(p["center"],q["center"])/h < 1.2:
                        group.add(pid)
                        pending.append(pid)
                        remaining.remove(pid)
            if len(group) < 3:
                continue
            paths = {pid:self._tail(list(self.history[pid]), max(1.6,self.hold)) for pid in sorted(group)}
            if not all(self._enough(path, max(1.6,self.hold)) for path in paths.values()):
                continue
            times = [{t for t,_ in path} for path in paths.values()]
            if not all(s == times[0] for s in times):
                continue
            velocities = []
            for path in paths.values():
                first,last = path[0][1],path[-1][1]
                h = (first["height"]+last["height"])/2
                v = sub(last["center"],first["center"])
                if norm(v)/h < .35:
                    break
                velocities.append(v)
            else:
                if not all(cosine(u,v) > .9 for u,v in combinations(velocities,2)):
                    continue
                stable = True
                for a,b in combinations(paths,2):
                    distances = []
                    for (_,p),(_,q) in zip(paths[a],paths[b]):
                        h = (p["height"]+q["height"])/2
                        if max(p["height"],q["height"])/min(p["height"],q["height"]) > 1.5:
                            stable = False
                        distances.append(math.dist(p["center"],q["center"])/h)
                    if max(distances) > 2.4 or max(distances)-min(distances) > .2:
                        stable = False
                if stable:
                    path = next(iter(paths.values()))
                    self._emit("group activity", sorted(group), path[0][0],path[-1][0],
                               f"{len(group)} nearby tracks moved in aligned directions while preserving relative spacing; shared purpose is unknown.")

    def results(self):
        return sorted((event for events in self.events.values() for event in events),
                      key=lambda e:(e["start_time"],e["behavior"],e["participants"]))
