"""
Undo/Redo manager for 3DMedicalPlanner.

Per-case action history with FIFO eviction (max 50 actions).
Each action stores serializable state snapshots — no in-memory mesh objects —
so the undo/redo API endpoints can replay them on demand.
"""

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
UPLOADS = DATA_DIR / "uploads"
OUTPUTS = DATA_DIR / "outputs"

MAX_ACTIONS = 50


class UndoAction:
    """Single undoable action."""

    def __init__(
        self,
        name: str,
        undo_payload: dict,
        redo_payload: dict,
        undo_fn: Optional[str] = None,
        redo_fn: Optional[str] = None,
    ):
        self.id: str = uuid.uuid4().hex[:12]
        self.name = name
        self.timestamp: float = time.time()
        self.undo_payload = undo_payload   # data needed to undo
        self.redo_payload = redo_payload   # data needed to redo
        self.undo_fn = undo_fn             # optional backend function name
        self.redo_fn = redo_fn             # optional backend function name

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp,
            "undo_fn": self.undo_fn,
            "redo_fn": self.redo_fn,
        }


class PerCaseState:
    """Snapshot helper — copies STL/JSON files to a versioned backup dir."""

    @staticmethod
    def snapshot_dir(case_id: str, version: str) -> Path:
        d = OUTPUTS / case_id / "_undo_history" / version
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def backup_current_state(case_id: str, version: str) -> dict:
        """Copy current input.stl + output files into a snapshot directory."""
        case_dir = UPLOADS / case_id
        out_dir = OUTPUTS / case_id
        snap = PerCaseState.snapshot_dir(case_id, version)
        backed = {}

        inp = case_dir / "input.stl"
        if inp.exists():
            dst = snap / "input.stl"
            shutil.copy2(inp, dst)
            backed["input_stl"] = True

        if out_dir.exists():
            for f in out_dir.iterdir():
                if f.name.startswith("_") or f.name.startswith("."):
                    continue
                dst = snap / f"output_{f.name}"
                shutil.copy2(f, dst)
                backed[f"output_{f.name.split('.')[0]}"] = True

        return backed

    @staticmethod
    def restore_state(case_id: str, version: str) -> dict:
        """Restore a snapshot back to the live case directory. Returns info about what was restored."""
        snap = OUTPUTS / case_id / "_undo_history" / version
        if not snap.exists():
            return {"restored": False, "reason": "snapshot not found"}

        case_dir = UPLOADS / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        out_dir = OUTPUTS / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        restored = {}

        # Restore input.stl if snap has it
        src_inp = snap / "input.stl"
        if src_inp.exists():
            shutil.copy2(src_inp, case_dir / "input.stl")
            restored["input_stl"] = True

        # Restore output files
        for f in snap.iterdir():
            if f.name.startswith("output_"):
                orig_name = f.name[len("output_"):]
                shutil.copy2(f, out_dir / orig_name)
                restored[orig_name] = True

        return {"restored": True, "files": restored}


class UndoManager:
    """Per-case undo/redo stack."""

    def __init__(self, case_id: str, max_actions: int = MAX_ACTIONS):
        self.case_id = case_id
        self.max_actions = max_actions
        self._undo_stack: list[UndoAction] = []
        self._redo_stack: list[UndoAction] = []
        self._persist_path: Path = OUTPUTS / case_id / "_undo_history" / "stack.json"

    # -- core operations --

    def push(self, action: UndoAction):
        """Record a new action. Clears redo stack. Evicts oldest if over limit."""
        # Snapshot the current state before the action happens
        action.undo_payload["_snapshot_version"] = uuid.uuid4().hex[:10]
        PerCaseState.backup_current_state(
            self.case_id, action.undo_payload["_snapshot_version"]
        )

        self._undo_stack.append(action)
        self._redo_stack.clear()

        # FIFO eviction
        while len(self._undo_stack) > self.max_actions:
            evicted = self._undo_stack.pop(0)
            # Clean up evicted snapshot
            evict_snap = (
                OUTPUTS / self.case_id / "_undo_history" / evicted.undo_payload.get("_snapshot_version", "")
            )
            if evict_snap.exists():
                shutil.rmtree(evict_snap, ignore_errors=True)

        self._persist()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> Optional[dict]:
        """Pop the last action and undo it. Returns the action info or None."""
        if not self._undo_stack:
            return None
        action = self._undo_stack.pop()
        result = PerCaseState.restore_state(
            self.case_id, action.undo_payload["_snapshot_version"]
        )

        # Re-derive snapshot_version for redo: snapshot current (now-restored) state
        action.redo_payload["_snapshot_version"] = uuid.uuid4().hex[:10]
        PerCaseState.backup_current_state(
            self.case_id, action.redo_payload["_snapshot_version"]
        )

        self._redo_stack.append(action)
        self._persist()
        return {
            "action": action.to_dict(),
            "restore_result": result,
        }

    def redo(self) -> Optional[dict]:
        """Pop the last undone action and redo it. Returns the action info or None."""
        if not self._redo_stack:
            return None
        action = self._redo_stack.pop()
        result = PerCaseState.restore_state(
            self.case_id, action.redo_payload["_snapshot_version"]
        )

        # Re-derive snapshot for new undo
        action.undo_payload["_snapshot_version"] = uuid.uuid4().hex[:10]
        PerCaseState.backup_current_state(
            self.case_id, action.undo_payload["_snapshot_version"]
        )

        self._undo_stack.append(action)
        self._persist()
        return {
            "action": action.to_dict(),
            "restore_result": result,
        }

    def history(self) -> dict:
        """Return serializable history."""
        return {
            "case_id": self.case_id,
            "can_undo": self.can_undo(),
            "can_redo": self.can_redo(),
            "undo_stack": [a.to_dict() for a in self._undo_stack],
            "redo_stack": [a.to_dict() for a in reversed(self._redo_stack)],
        }

    # -- persistence --

    def _persist(self):
        """Save stack metadata (not the file snapshots) to JSON."""
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "case_id": self.case_id,
            "undo_stack": [
                {
                    "id": a.id,
                    "name": a.name,
                    "timestamp": a.timestamp,
                    "undo_payload": {
                        k: v
                        for k, v in a.undo_payload.items()
                                               if k != "_snapshot_version"
                    },
                    "redo_payload": {
                        k: v
                        for k, v in a.redo_payload.items()
                        if k != "_snapshot_version"
                    },
                    "undo_fn": a.undo_fn,
                    "redo_fn": a.redo_fn,
                    "_undo_snapshot": a.undo_payload.get("_snapshot_version", ""),
                    "_redo_snapshot": a.redo_payload.get("_snapshot_version", ""),
                }
                for a in self._undo_stack
            ],
            "redo_stack": [
                {
                    "id": a.id,
                    "name": a.name,
                    "timestamp": a.timestamp,
                    "undo_payload": {
                        k: v
                        for k, v in a.undo_payload.items()
                        if k != "_snapshot_version"
                    },
                    "redo_payload": {
                        k: v
                        for k, v in a.redo_payload.items()
                        if k != "_snapshot_version"
                    },
                    "undo_fn": a.undo_fn,
                    "redo_fn": a.redo_fn,
                    "_undo_snapshot": a.undo_payload.get("_snapshot_version", ""),
                    "_redo_snapshot": a.redo_payload.get("_snapshot_version", ""),
                }
                for a in self._redo_stack
            ],
        }
        self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, case_id: str, max_actions: int = MAX_ACTIONS) -> "UndoManager":
        """Load a persisted UndoManager from disk, or create a fresh one."""
        mgr = cls(case_id, max_actions=max_actions)
        if mgr._persist_path.exists():
            try:
                data = json.loads(mgr._persist_path.read_text(encoding="utf-8"))
                mgr._undo_stack = [
                    _rehydrate_action(a) for a in data.get("undo_stack", [])
                ]
                mgr._redo_stack = [
                    _rehydrate_action(a) for a in data.get("redo_stack", [])
                ]
            except Exception:
                pass  # corrupt file — start fresh
        return mgr


def _rehydrate_action(d: dict) -> UndoAction:
    a = UndoAction(
        name=d["name"],
        undo_payload=dict(d.get("undo_payload", {})),
        redo_payload=dict(d.get("redo_payload", {})),
        undo_fn=d.get("undo_fn"),
        redo_fn=d.get("redo_fn"),
    )
    a.id = d.get("id", a.id)
    a.timestamp = d.get("timestamp", a.timestamp)
    a.undo_payload["_snapshot_version"] = d.get("_undo_snapshot", "")
    a.redo_payload["_snapshot_version"] = d.get("_redo_snapshot", "")
    return a


# -- global registry --

_managers: dict[str, UndoManager] = {}


def get_manager(case_id: str) -> UndoManager:
    """Get or create the UndoManager for a case."""
    if case_id not in _managers:
        _managers[case_id] = UndoManager.load(case_id)
    return _managers[case_id]
