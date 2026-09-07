"""Closed-loop frame-budget controller stand-in.

Real contract, from backend.py's use of it:
    FrameServer(ladder=[preset,...], headroom=float)
    .serve_frame(session, render_fn, target_fps=) -> {"payload": render_fn(preset), "preset": preset}
    ._sessions[session] -> a controller object with the current rung
The ladder is coarsest-first; the controller drops a rung when a frame blows its budget and climbs
after several fast ones. This stand-in keeps the same shapes and a simplified version of that rule,
so the app's ADAPTIVE branch (the client's default) is actually exercised.
"""
import time


class _Controller:
    def __init__(self, ladder, headroom):
        self.ladder = list(ladder); self.headroom = float(headroom)
        self.i = max(0, len(self.ladder) // 2)
        self.fast = 0; self.last_ms = 0.0

    @property
    def preset(self):
        return self.ladder[self.i]

    @property
    def level(self):
        return self.i

    def current(self):
        return self.preset

    def stats(self):
        return {"last_ms": round(self.last_ms, 1), "budget_hits": self.fast}

    def observe(self, seconds, target_fps):
        budget = 1.0 / max(float(target_fps), 1.0)
        self.last_ms = seconds * 1000.0
        if seconds > budget * (1.0 + self.headroom):
            self.i = max(0, self.i - 1); self.fast = 0
        elif seconds < budget * 0.6:
            self.fast += 1
            if self.fast >= 3:
                self.i = min(len(self.ladder) - 1, self.i + 1); self.fast = 0
        return self.preset


class FrameServer:
    def __init__(self, ladder=None, headroom=0.15, **kw):
        self.ladder = list(ladder or [{"name": "medium", "scale": 0.55}])
        self.headroom = float(headroom)
        self._sessions = {}

    def _ctl(self, session):
        if session not in self._sessions:
            self._sessions[session] = _Controller(self.ladder, self.headroom)
        return self._sessions[session]

    def serve_frame(self, session, render_fn, target_fps=30, **kw):
        ctl = self._ctl(session)
        preset = ctl.preset
        t0 = time.time()
        try:
            payload = render_fn(preset)
        except Exception as e:                       # the route checks payload["error"]
            payload = {"error": str(e)}
        dt = time.time() - t0
        ctl.observe(dt, target_fps)
        # The route formats every one of these into X-Holostuff-Render, so the real serve_frame
        # returns them all -- a missing key here is a 500 on the client's DEFAULT preview path.
        return {"payload": payload, "preset": preset, "session": session,
                "frame_ms": dt * 1000.0,
                "budget_ms": 1000.0 / max(float(target_fps), 1.0),
                "stats": {"met_budget_frac": round(1.0 if dt <= 1.0 / max(float(target_fps), 1.0) else 0.0, 2),
                          "level": ctl.level, "last_ms": round(ctl.last_ms, 1)}}

    def stats(self, *a, **k):
        return self.sessions()

    def sessions(self):
        return {s: {"level": c.level, "preset": c.preset["name"], "last_ms": round(c.last_ms, 1)}
                for s, c in self._sessions.items()}
