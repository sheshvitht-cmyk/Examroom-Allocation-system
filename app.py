#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================================
  EXAM ROOM ALLOCATION SYSTEM  -  Graph Coloring + Greedy Algorithm  (DAA Mini Project)
  MALLA REDDY VISHVAVIDHYAPEETH            Student: THOKALA SHESHVITH
==========================================================================================

HOW TO RUN (VS Code terminal)
    1)  pip install flask
    2)  python app.py
    3)  The browser opens automatically at  http://127.0.0.1:5000
        (an internet connection is needed once per page load: React, Tailwind, Lucide icons
         and jsPDF are loaded from public CDNs.)

FILE LAYOUT (everything is in this one file so it can be run directly)
    SECTION 1  Database layer         - SQLite schema + helpers
    SECTION 2  ALGORITHM MODULE       - Greedy Graph Coloring + verification  (pure Python,
                                        no Flask / DB code -> easy to demonstrate in a viva)
    SECTION 3  Validation             - input validation for exams / rooms / slots / conflicts
    SECTION 4  REST API               - /api/exams, /api/rooms, /api/timeslots, /api/conflicts,
                                        /api/run-algorithm, /api/allocation, /api/analytics ...
    SECTION 5  Frontend               - React single-page app (served from "/")

MODEL
    vertex  = exam / class                 edge  = conflict (two exams must not run together)
    colour k = k-th examination time slot  (exams with the same colour run in the same slot)
    Inside a slot every exam gets a different room that is large enough (best-fit).
"""
import json
import os
import re
import sqlite3
import threading
import webbrowser
from datetime import datetime

from flask import Flask, Response, g, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "exam_allocation.db")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ==========================================================================================
# SECTION 1 - DATABASE LAYER
# ==========================================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS exams (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    subject   TEXT NOT NULL,
    students  INTEGER NOT NULL CHECK (students > 0),
    priority  TEXT NOT NULL DEFAULT 'Medium',
    duration  INTEGER NOT NULL DEFAULT 180
);
CREATE TABLE IF NOT EXISTS rooms (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    code      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    capacity  INTEGER NOT NULL CHECK (capacity > 0),
    building  TEXT NOT NULL,
    floor     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS timeslots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    label     TEXT NOT NULL UNIQUE COLLATE NOCASE,
    day       TEXT NOT NULL DEFAULT 'Day 1',
    start     TEXT NOT NULL,
    end       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conflicts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_a    INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    exam_b    INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    CHECK (exam_a <> exam_b),
    UNIQUE (exam_a, exam_b)
);
CREATE TABLE IF NOT EXISTS algorithm_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,
    ordering          TEXT NOT NULL,
    success           INTEGER NOT NULL,
    message           TEXT,
    order_json        TEXT,
    steps_json        TEXT,
    summary_json      TEXT,
    verification_json TEXT
);
CREATE TABLE IF NOT EXISTS allocations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL REFERENCES algorithm_runs(id) ON DELETE CASCADE,
    exam_id   INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    color     INTEGER,
    room_id   INTEGER REFERENCES rooms(id) ON DELETE SET NULL,
    slot_id   INTEGER REFERENCES timeslots(id) ON DELETE SET NULL,
    status    TEXT NOT NULL
);
"""


def get_db():
    """One SQLite connection per request (stored on flask.g)."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def rows(db, sql, args=()):
    return [dict(r) for r in db.execute(sql, args).fetchall()]


def invalidate_runs(db):
    """Any change to the input data makes the previous allocation stale -> remove it."""
    db.execute("DELETE FROM algorithm_runs")  # allocations are removed by ON DELETE CASCADE


def conflicts_list(db):
    return rows(
        db,
        """SELECT c.id, c.exam_a, c.exam_b, a.name AS a_name, b.name AS b_name
           FROM conflicts c
           JOIN exams a ON a.id = c.exam_a
           JOIN exams b ON b.id = c.exam_b
           ORDER BY c.id""",
    )


# ==========================================================================================
# SECTION 2 - ALGORITHM MODULE  (Graph Coloring + Greedy)
# ==========================================================================================
PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}

ORDERING_LABELS = {
    "degree_desc": "Descending degree (Welsh-Powell)",
    "degree_asc": "Ascending degree",
    "input": "Input order",
}


def to_minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def build_adjacency(exam_ids, edges):
    """Adjacency-list representation of the conflict graph: vertex -> set(neighbours)."""
    adj = {i: set() for i in exam_ids}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def order_vertices(exams, adj, strategy="degree_desc"):
    """
    VERTEX ORDERING RULE
      degree_desc : highest number of conflicts first  (default).
                    Ties are broken by priority (High first), then by student count
                    (larger first), then by input order.
      degree_asc  : lowest degree first (used only to demonstrate that order matters).
      input       : exactly the order the exams were entered.
    """
    if strategy == "input":
        return sorted(exams, key=lambda e: e["id"])
    if strategy == "degree_asc":
        return sorted(exams, key=lambda e: (len(adj[e["id"]]), e["id"]))
    return sorted(
        exams,
        key=lambda e: (-len(adj[e["id"]]), PRIORITY_RANK.get(e["priority"], 1), -e["students"], e["id"]),
    )


def pure_greedy_color_count(order_ids, adj):
    """Plain greedy colouring (no rooms / slots) - used only for the ordering comparison."""
    color = {}
    for v in order_ids:
        used = {color[u] for u in adj[v] if u in color}
        c = 1
        while c in used:
            c += 1
        color[v] = c
    return max(color.values(), default=0)


def chromatic_number(ids, adj, limit=14):
    """Exact minimum number of colours by back-tracking (only for small graphs)."""
    if len(ids) > limit:
        return None
    if not ids:
        return 0
    order = sorted(ids, key=lambda v: -len(adj[v]))
    for k in range(1, len(order) + 1):
        col = {}

        def dfs(i):
            if i == len(order):
                return True
            v = order[i]
            used = {col[n] for n in adj[v] if n in col}
            for c in range(1, k + 1):
                if c not in used:
                    col[v] = c
                    if dfs(i + 1):
                        return True
                    del col[v]
            return False

        if dfs(0):
            return k
    return len(order)


def greedy_graph_coloring(exams, rooms, slots, edges, strategy="degree_desc"):
    """
    RESOURCE-AWARE GREEDY GRAPH COLOURING

    colour k  <->  k-th time slot.   For every vertex v (in the chosen order):
        1. look at the colours already given to neighbours of v      (conflict check)
        2. try colour 1, 2, 3 ... in increasing order                (smallest colour first)
        3. colour c is REJECTED if
              - a neighbour already uses c                           (graph colouring rule)
              - the slot is shorter than the exam duration
              - no free room with enough seats exists in that slot   (resource validation)
        4. the first colour that is not rejected is assigned; the best-fit room
           (smallest room that is large enough) is booked for that slot.
    Every decision is recorded in `steps` so it can be shown / replayed in the UI.
    """
    exam_ids = [e["id"] for e in exams]
    adj = build_adjacency(exam_ids, edges)
    order = order_vertices(exams, adj, strategy)
    name_of = {e["id"]: e["name"] for e in exams}

    color = {}  # exam_id -> colour number (1-based)
    assignment = {}  # exam_id -> {color, room_id, slot_id}
    busy = set()  # (colour, room_id) already booked
    steps, failures = [], []
    max_capacity = max((r["capacity"] for r in rooms), default=0)
    kinds_seen = set()

    for step_no, ex in enumerate(order, start=1):
        v = ex["id"]
        neighbours = sorted(adj[v], key=lambda i: name_of[i])
        neighbour_info = [{"id": n, "name": name_of[n], "color": color.get(n)} for n in neighbours]
        neighbour_colors = sorted({color[n] for n in neighbours if n in color})
        rejected, chosen, chosen_room = [], None, None

        if ex["students"] > max_capacity:
            # No room in the whole campus is big enough -> impossible for every colour.
            reason = (
                f"Room capacity is insufficient for {ex['name']}: it needs {ex['students']} seats "
                f"but the largest room has {max_capacity}."
            )
            kinds_seen.add("capacity")
        else:
            reason = ""
            for c in range(1, len(slots) + 1):
                slot = slots[c - 1]
                blockers = [name_of[n] for n in neighbours if color.get(n) == c]
                if blockers:
                    rejected.append({
                        "color": c, "kind": "conflict",
                        "reason": f"Color {c} is unavailable because {ex['name']} conflicts with {', '.join(blockers)}",
                    })
                    continue
                if ex["duration"] > to_minutes(slot["end"]) - to_minutes(slot["start"]):
                    rejected.append({
                        "color": c, "kind": "duration",
                        "reason": f"Color {c} ({slot['label']}) is shorter than the {ex['duration']}-minute exam",
                    })
                    continue
                free = [r for r in rooms if r["capacity"] >= ex["students"] and (c, r["id"]) not in busy]
                if not free:
                    rejected.append({
                        "color": c, "kind": "room",
                        "reason": f"Color {c} ({slot['label']}) has no free room with at least {ex['students']} seats",
                    })
                    continue
                chosen = c
                chosen_room = min(free, key=lambda r: (r["capacity"], r["id"]))  # best-fit room
                break

            if chosen is not None:
                if not rejected:
                    if neighbour_colors:
                        reason = f"No neighbouring exam has been assigned Color {chosen}."
                    else:
                        reason = f"No neighbouring exam has been assigned any color yet, so Color {chosen} is feasible."
                else:
                    reason = ". ".join(r["reason"] for r in rejected) + f". Color {chosen} is the smallest feasible color."
            else:
                for r in rejected:
                    kinds_seen.add(r["kind"])
                reason = (
                    f"No available room/time resource satisfies the current constraints for {ex['name']}. "
                    + "; ".join(r["reason"] for r in rejected)
                )

        unavailable = sorted(set(neighbour_colors) | {r["color"] for r in rejected})
        step = {
            "step": step_no, "exam_id": v, "exam": ex["name"], "subject": ex["subject"],
            "students": ex["students"], "degree": len(adj[v]),
            "neighbours": neighbour_info, "neighbour_colors": neighbour_colors,
            "unavailable_colors": unavailable, "rejected": rejected,
            "selected_color": chosen, "room_id": None, "room": None, "building": None,
            "slot_id": None, "slot_label": None, "slot_start": None, "slot_end": None, "slot_day": None,
            "reason": reason, "success": chosen is not None,
        }
        if chosen is not None:
            slot = slots[chosen - 1]
            color[v] = chosen
            busy.add((chosen, chosen_room["id"]))
            assignment[v] = {"color": chosen, "room_id": chosen_room["id"], "slot_id": slot["id"]}
            step.update({
                "room_id": chosen_room["id"], "room": chosen_room["name"], "building": chosen_room["building"],
                "slot_id": slot["id"], "slot_label": slot["label"], "slot_start": slot["start"],
                "slot_end": slot["end"], "slot_day": slot["day"],
            })
        else:
            failures.append({"exam_id": v, "exam": ex["name"], "reason": reason})
        steps.append(step)

    suggestions = []
    if "capacity" in kinds_seen or "room" in kinds_seen:
        suggestions += ["Add another room", "Increase room capacity"]
    if "conflict" in kinds_seen or "room" in kinds_seen:
        suggestions.append("Add another time slot")
    if "conflict" in kinds_seen:
        suggestions.append("Modify exam constraints (remove a conflict)")
    if "duration" in kinds_seen:
        suggestions += ["Lengthen the time slots", "Shorten the exam duration"]
    seen = set()
    suggestions = [s for s in suggestions if not (s in seen or seen.add(s))]

    return {
        "order": [{"id": e["id"], "name": e["name"], "degree": len(adj[e["id"]])} for e in order],
        "steps": steps, "assignment": assignment, "failures": failures, "suggestions": suggestions,
    }


def verify_allocation(exams, rooms, slots, edges, assignment):
    """
    VERIFICATION - computed from the real allocation, never hard-coded.
        for every edge (A, B):  colour[A] == colour[B]  ->  conflict,  otherwise valid
    Also checks room capacity and that no room hosts two exams in the same slot.
    """
    name_of = {e["id"]: e["name"] for e in exams}
    exam_by_id = {e["id"]: e for e in exams}
    room_by_id = {r["id"]: r for r in rooms}

    edge_results, conflicts = [], []
    for a, b in edges:
        ca = assignment.get(a, {}).get("color")
        cb = assignment.get(b, {}).get("color")
        bad = ca is not None and cb is not None and ca == cb
        edge_results.append({"a": name_of[a], "b": name_of[b], "color_a": ca, "color_b": cb,
                             "valid": (not bad) and ca is not None and cb is not None})
        if bad:
            conflicts.append({"a": name_of[a], "b": name_of[b], "color": ca})

    cap_viol, clashes, seen = [], [], {}
    for eid, asg in assignment.items():
        room = room_by_id.get(asg["room_id"])
        if room is None or room["capacity"] < exam_by_id[eid]["students"]:
            cap_viol.append(name_of[eid])
        key = (asg["slot_id"], asg["room_id"])
        if key in seen:
            clashes.append({"exam": name_of[eid], "other": name_of[seen[key]]})
        else:
            seen[key] = eid
    unassigned = [e["name"] for e in exams if e["id"] not in assignment]
    passed = not conflicts and not cap_viol and not clashes and not unassigned
    return {
        "edges_checked": len(edges), "edge_results": edge_results,
        "conflicts": conflicts, "conflict_count": len(conflicts),
        "capacity_violations": cap_viol, "capacity_violation_count": len(cap_viol),
        "room_clashes": clashes, "room_clash_count": len(clashes),
        "unassigned": unassigned, "passed": passed,
    }


# ==========================================================================================
# SECTION 3 - VALIDATION
# ==========================================================================================
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def to_int(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(str(v).strip())
    except ValueError:
        return None
    return int(f) if f == int(f) else None


def clean(v):
    return str(v if v is not None else "").strip()


def bad_request(errors):
    errors = errors if isinstance(errors, list) else [errors]
    return jsonify({"error": errors[0], "errors": errors}), 400


def validate_exam(db, d, exam_id=None):
    errors = []
    name, subject = clean(d.get("name")), clean(d.get("subject"))
    students = to_int(d.get("students"))
    duration = to_int(d.get("duration", 180))
    priority = clean(d.get("priority") or "Medium").title()
    if not name:
        errors.append("Exam/Class name is required.")
    elif len(name) > 60:
        errors.append("Exam/Class name is too long (max 60 characters).")
    elif db.execute("SELECT 1 FROM exams WHERE lower(name)=lower(?) AND id IS NOT ?", (name, exam_id)).fetchone():
        errors.append(f"An exam named '{name}' already exists. Duplicate exam names are not allowed.")
    if not subject:
        errors.append("Subject is required.")
    if students is None or students <= 0:
        errors.append("Student count must be a positive whole number (negative or zero values are not allowed).")
    elif students > 5000:
        errors.append("Student count looks unrealistic (max 5000).")
    if priority not in PRIORITY_RANK:
        errors.append("Priority must be High, Medium or Low.")
    if duration is None or not (15 <= duration <= 600):
        errors.append("Duration must be between 15 and 600 minutes.")
    return errors, {"name": name, "subject": subject, "students": students, "priority": priority, "duration": duration}


def validate_room(db, d, room_id=None):
    errors = []
    code, name, building = clean(d.get("code")), clean(d.get("name")), clean(d.get("building"))
    capacity, floor = to_int(d.get("capacity")), to_int(d.get("floor", 0))
    if not code:
        errors.append("Room ID is required.")
    elif db.execute("SELECT 1 FROM rooms WHERE lower(code)=lower(?) AND id IS NOT ?", (code, room_id)).fetchone():
        errors.append(f"Room ID '{code}' already exists. Duplicate rooms are not allowed.")
    if not name:
        errors.append("Room name/number is required.")
    elif db.execute("SELECT 1 FROM rooms WHERE lower(name)=lower(?) AND id IS NOT ?", (name, room_id)).fetchone():
        errors.append(f"A room named '{name}' already exists. Duplicate rooms are not allowed.")
    if capacity is None or capacity <= 0:
        errors.append("Room capacity must be greater than zero.")
    elif capacity > 5000:
        errors.append("Room capacity looks unrealistic (max 5000).")
    if not building:
        errors.append("Building is required.")
    if floor is None or floor < 0 or floor > 100:
        errors.append("Floor must be a whole number between 0 and 100.")
    return errors, {"code": code, "name": name, "capacity": capacity, "building": building, "floor": floor}


def validate_slot(db, d, slot_id=None):
    errors = []
    label, day = clean(d.get("label")), clean(d.get("day")) or "Day 1"
    start, end = clean(d.get("start")), clean(d.get("end"))
    if not label:
        errors.append("Slot name is required.")
    elif db.execute("SELECT 1 FROM timeslots WHERE lower(label)=lower(?) AND id IS NOT ?", (label, slot_id)).fetchone():
        errors.append(f"A slot named '{label}' already exists.")
    if not TIME_RE.match(start) or not TIME_RE.match(end):
        errors.append("Start and end time must be valid times (HH:MM).")
    elif to_minutes(start) >= to_minutes(end):
        errors.append("End time must be after the start time.")
    else:
        for o in rows(db, "SELECT * FROM timeslots WHERE lower(day)=lower(?) AND id IS NOT ?", (day, slot_id)):
            if to_minutes(start) < to_minutes(o["end"]) and to_minutes(o["start"]) < to_minutes(end):
                errors.append(f"This slot overlaps with '{o['label']}' on {day}. Slots on the same day must not overlap.")
                break
    return errors, {"label": label, "day": day, "start": start, "end": end}


# ==========================================================================================
# SECTION 4 - REST API
# ==========================================================================================
def run_payload(db):
    """Latest algorithm run + allocation rows (read back from the database)."""
    row = db.execute("SELECT * FROM algorithm_runs ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return {"run": None, "allocation": []}
    summary = json.loads(row["summary_json"])
    run = {
        "id": row["id"], "created_at": row["created_at"], "ordering": row["ordering"],
        "ordering_label": ORDERING_LABELS.get(row["ordering"], row["ordering"]),
        "success": bool(row["success"]), "message": row["message"],
        "order": json.loads(row["order_json"]), "steps": json.loads(row["steps_json"]),
        "summary": summary, "verification": json.loads(row["verification_json"]),
        "failures": summary.get("failures", []), "suggestions": summary.get("suggestions", []),
    }
    alloc = rows(
        db,
        """SELECT a.exam_id, a.color, a.room_id, a.slot_id, a.status,
                  e.name AS exam, e.subject, e.students, e.priority,
                  r.name AS room, r.building, r.floor, r.capacity,
                  t.label AS slot, t.start AS slot_start, t.end AS slot_end, t.day AS slot_day
           FROM allocations a
           JOIN exams e ON e.id = a.exam_id
           LEFT JOIN rooms r ON r.id = a.room_id
           LEFT JOIN timeslots t ON t.id = a.slot_id
           WHERE a.run_id = ?
           ORDER BY a.color IS NULL, a.color, e.name""",
        (row["id"],),
    )
    return {"run": run, "allocation": alloc}


def execute_run(db, strategy):
    exams = rows(db, "SELECT * FROM exams ORDER BY id")
    rooms_ = rows(db, "SELECT * FROM rooms ORDER BY id")
    slots = rows(db, "SELECT * FROM timeslots ORDER BY id")
    edges = [(r["exam_a"], r["exam_b"]) for r in rows(db, "SELECT exam_a, exam_b FROM conflicts ORDER BY id")]

    result = greedy_graph_coloring(exams, rooms_, slots, edges, strategy)
    ver = verify_allocation(exams, rooms_, slots, edges, result["assignment"])
    asg = result["assignment"]

    success = not result["failures"]
    clean_status = "CONFLICT FREE" if ver["passed"] else "REQUIRES ATTENTION"
    summary = {
        "total_exams": len(exams), "total_conflicts": len(edges), "assigned": len(asg),
        "colors_used": len({a["color"] for a in asg.values()}),
        "unresolved_conflicts": ver["conflict_count"],
        "rooms_used": len({a["room_id"] for a in asg.values()}),
        "slots_used": len({a["slot_id"] for a in asg.values()}),
        "capacity_violations": ver["capacity_violation_count"], "room_clashes": ver["room_clash_count"],
        "unassigned": len(ver["unassigned"]), "status": clean_status,
        "failures": result["failures"], "suggestions": result["suggestions"],
    }
    message = "Allocation completed successfully." if success else (
        "Allocation Could Not Be Completed: " + " | ".join(f["reason"] for f in result["failures"]))

    db.execute("DELETE FROM algorithm_runs")
    cur = db.execute(
        """INSERT INTO algorithm_runs (created_at, ordering, success, message, order_json, steps_json,
                                       summary_json, verification_json) VALUES (?,?,?,?,?,?,?,?)""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), strategy, int(success), message,
         json.dumps(result["order"]), json.dumps(result["steps"]), json.dumps(summary), json.dumps(ver)),
    )
    run_id = cur.lastrowid

    # per-exam status: conflict free only if verified for that exam
    bad_exams = {c["a"] for c in ver["conflicts"]} | {c["b"] for c in ver["conflicts"]} | set(ver["capacity_violations"])
    bad_exams |= {c["exam"] for c in ver["room_clashes"]} | {c["other"] for c in ver["room_clashes"]}
    for e in exams:
        a = asg.get(e["id"])
        if a is None:
            db.execute("INSERT INTO allocations (run_id, exam_id, color, room_id, slot_id, status) VALUES (?,?,?,?,?,?)",
                       (run_id, e["id"], None, None, None, "Requires Attention"))
        else:
            status = "Requires Attention" if e["name"] in bad_exams else "Conflict Free"
            db.execute("INSERT INTO allocations (run_id, exam_id, color, room_id, slot_id, status) VALUES (?,?,?,?,?,?)",
                       (run_id, e["id"], a["color"], a["room_id"], a["slot_id"], status))
    db.commit()


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/api/state")
def api_state():
    db = get_db()
    return jsonify({
        "exams": rows(db, "SELECT * FROM exams ORDER BY id"),
        "rooms": rows(db, "SELECT * FROM rooms ORDER BY id"),
        "timeslots": rows(db, "SELECT * FROM timeslots ORDER BY id"),
        "conflicts": conflicts_list(db),
        **run_payload(db),
    })


# ---------- exams ----------
@app.get("/api/exams")
def exams_get():
    return jsonify(rows(get_db(), "SELECT * FROM exams ORDER BY id"))


@app.post("/api/exams")
def exams_post():
    db = get_db()
    errors, v = validate_exam(db, request.get_json(silent=True) or {})
    if errors:
        return bad_request(errors)
    cur = db.execute("INSERT INTO exams (name, subject, students, priority, duration) VALUES (?,?,?,?,?)",
                     (v["name"], v["subject"], v["students"], v["priority"], v["duration"]))
    invalidate_runs(db)
    db.commit()
    return jsonify({"id": cur.lastrowid, **v}), 201


@app.put("/api/exams/<int:eid>")
def exams_put(eid):
    db = get_db()
    if not db.execute("SELECT 1 FROM exams WHERE id=?", (eid,)).fetchone():
        return jsonify({"error": "Exam not found."}), 404
    errors, v = validate_exam(db, request.get_json(silent=True) or {}, eid)
    if errors:
        return bad_request(errors)
    db.execute("UPDATE exams SET name=?, subject=?, students=?, priority=?, duration=? WHERE id=?",
               (v["name"], v["subject"], v["students"], v["priority"], v["duration"], eid))
    invalidate_runs(db)
    db.commit()
    return jsonify({"id": eid, **v})


@app.delete("/api/exams/<int:eid>")
def exams_delete(eid):
    db = get_db()
    if not db.execute("DELETE FROM exams WHERE id=?", (eid,)).rowcount:
        return jsonify({"error": "Exam not found."}), 404
    invalidate_runs(db)
    db.commit()
    return jsonify({"ok": True})


# ---------- rooms ----------
@app.get("/api/rooms")
def rooms_get():
    return jsonify(rows(get_db(), "SELECT * FROM rooms ORDER BY id"))


@app.post("/api/rooms")
def rooms_post():
    db = get_db()
    errors, v = validate_room(db, request.get_json(silent=True) or {})
    if errors:
        return bad_request(errors)
    cur = db.execute("INSERT INTO rooms (code, name, capacity, building, floor) VALUES (?,?,?,?,?)",
                     (v["code"], v["name"], v["capacity"], v["building"], v["floor"]))
    invalidate_runs(db)
    db.commit()
    return jsonify({"id": cur.lastrowid, **v}), 201


@app.put("/api/rooms/<int:rid>")
def rooms_put(rid):
    db = get_db()
    if not db.execute("SELECT 1 FROM rooms WHERE id=?", (rid,)).fetchone():
        return jsonify({"error": "Room not found."}), 404
    errors, v = validate_room(db, request.get_json(silent=True) or {}, rid)
    if errors:
        return bad_request(errors)
    db.execute("UPDATE rooms SET code=?, name=?, capacity=?, building=?, floor=? WHERE id=?",
               (v["code"], v["name"], v["capacity"], v["building"], v["floor"], rid))
    invalidate_runs(db)
    db.commit()
    return jsonify({"id": rid, **v})


@app.delete("/api/rooms/<int:rid>")
def rooms_delete(rid):
    db = get_db()
    if not db.execute("DELETE FROM rooms WHERE id=?", (rid,)).rowcount:
        return jsonify({"error": "Room not found."}), 404
    invalidate_runs(db)
    db.commit()
    return jsonify({"ok": True})


# ---------- time slots ----------
@app.get("/api/timeslots")
def slots_get():
    return jsonify(rows(get_db(), "SELECT * FROM timeslots ORDER BY id"))


@app.post("/api/timeslots")
def slots_post():
    db = get_db()
    errors, v = validate_slot(db, request.get_json(silent=True) or {})
    if errors:
        return bad_request(errors)
    cur = db.execute("INSERT INTO timeslots (label, day, start, end) VALUES (?,?,?,?)",
                     (v["label"], v["day"], v["start"], v["end"]))
    invalidate_runs(db)
    db.commit()
    return jsonify({"id": cur.lastrowid, **v}), 201


@app.put("/api/timeslots/<int:sid>")
def slots_put(sid):
    db = get_db()
    if not db.execute("SELECT 1 FROM timeslots WHERE id=?", (sid,)).fetchone():
        return jsonify({"error": "Time slot not found."}), 404
    errors, v = validate_slot(db, request.get_json(silent=True) or {}, sid)
    if errors:
        return bad_request(errors)
    db.execute("UPDATE timeslots SET label=?, day=?, start=?, end=? WHERE id=?",
               (v["label"], v["day"], v["start"], v["end"], sid))
    invalidate_runs(db)
    db.commit()
    return jsonify({"id": sid, **v})


@app.delete("/api/timeslots/<int:sid>")
def slots_delete(sid):
    db = get_db()
    if not db.execute("DELETE FROM timeslots WHERE id=?", (sid,)).rowcount:
        return jsonify({"error": "Time slot not found."}), 404
    invalidate_runs(db)
    db.commit()
    return jsonify({"ok": True})


# ---------- conflicts (edges) ----------
@app.get("/api/conflicts")
def conflicts_get():
    return jsonify(conflicts_list(get_db()))


@app.post("/api/conflicts")
def conflicts_post():
    db = get_db()
    d = request.get_json(silent=True) or {}
    a, b = to_int(d.get("exam_a")), to_int(d.get("exam_b"))
    if a is None or b is None:
        return bad_request("Select both Exam A and Exam B.")
    if a == b:
        return bad_request("An exam cannot conflict with itself.")
    for i in (a, b):
        if not db.execute("SELECT 1 FROM exams WHERE id=?", (i,)).fetchone():
            return bad_request("Selected exam does not exist.")
    lo, hi = min(a, b), max(a, b)  # undirected edge stored once as (smaller id, larger id)
    if db.execute("SELECT 1 FROM conflicts WHERE exam_a=? AND exam_b=?", (lo, hi)).fetchone():
        return bad_request("This conflict already exists. Duplicate conflicts are not allowed.")
    cur = db.execute("INSERT INTO conflicts (exam_a, exam_b) VALUES (?,?)", (lo, hi))
    invalidate_runs(db)
    db.commit()
    return jsonify({"id": cur.lastrowid, "exam_a": lo, "exam_b": hi}), 201


@app.delete("/api/conflicts/<int:cid>")
def conflicts_delete(cid):
    db = get_db()
    if not db.execute("DELETE FROM conflicts WHERE id=?", (cid,)).rowcount:
        return jsonify({"error": "Conflict not found."}), 404
    invalidate_runs(db)
    db.commit()
    return jsonify({"ok": True})


# ---------- algorithm ----------
@app.post("/api/run-algorithm")
def run_algorithm():
    db = get_db()
    d = request.get_json(silent=True) or {}
    strategy = d.get("ordering", "degree_desc")
    if strategy not in ORDERING_LABELS:
        return bad_request("Unknown ordering strategy.")
    problems = []
    if not db.execute("SELECT 1 FROM exams").fetchone():
        problems.append("No exams added yet. Add at least one exam before running the algorithm.")
    if not db.execute("SELECT 1 FROM rooms").fetchone():
        problems.append("No rooms added yet. Add at least one examination room.")
    if not db.execute("SELECT 1 FROM timeslots").fetchone():
        problems.append("No time slots added yet. Add at least one examination time slot.")
    if problems:
        return bad_request(problems)
    execute_run(db, strategy)
    return jsonify(run_payload(db))


@app.get("/api/allocation")
def allocation_get():
    return jsonify(run_payload(get_db()))


@app.get("/api/ordering-comparison")
def ordering_comparison():
    """Runs plain greedy colouring with different vertex orders to show that order matters."""
    db = get_db()
    exams = rows(db, "SELECT * FROM exams ORDER BY id")
    edges = [(r["exam_a"], r["exam_b"]) for r in rows(db, "SELECT exam_a, exam_b FROM conflicts")]
    ids = [e["id"] for e in exams]
    adj = build_adjacency(ids, edges)
    out = []
    for key, label in ORDERING_LABELS.items():
        order = order_vertices(exams, adj, key)
        out.append({"key": key, "label": label,
                    "colors_used": pure_greedy_color_count([e["id"] for e in order], adj),
                    "order": [e["name"] for e in order]})
    max_deg = max((len(v) for v in adj.values()), default=0)
    return jsonify({"strategies": out, "max_degree": max_deg, "upper_bound": max_deg + 1 if ids else 0,
                    "optimal": chromatic_number(ids, adj)})


@app.get("/api/analytics")
def analytics():
    db = get_db()
    exams = rows(db, "SELECT * FROM exams ORDER BY id")
    rooms_ = rows(db, "SELECT * FROM rooms ORDER BY id")
    slots = rows(db, "SELECT * FROM timeslots ORDER BY id")
    edges = [(r["exam_a"], r["exam_b"]) for r in rows(db, "SELECT exam_a, exam_b FROM conflicts")]
    n, e = len(exams), len(edges)
    adj = build_adjacency([x["id"] for x in exams], edges)
    payload = run_payload(db)
    run, alloc = payload["run"], [a for a in payload["allocation"] if a["room_id"]]

    # Baseline "before optimisation": naive round-robin (exam i -> slot i mod S), no colouring.
    before = None
    if slots:
        naive = {x["id"]: i % len(slots) for i, x in enumerate(exams)}
        before = sum(1 for a, b in edges if naive[a] == naive[b])

    room_util = []
    for r in rooms_:
        mine = [a for a in alloc if a["room_id"] == r["id"]]
        used = len({a["slot_id"] for a in mine})
        room_util.append({
            "room": r["name"], "capacity": r["capacity"], "exams": len(mine), "slots_used": used,
            "usage_pct": round(used / len(slots) * 100, 1) if slots else 0,
            "seat_fill_pct": round(sum(a["students"] for a in mine) / (len(mine) * r["capacity"]) * 100, 1) if mine else 0,
        })
    slot_util = []
    for s in slots:
        mine = [a for a in alloc if a["slot_id"] == s["id"]]
        slot_util.append({"slot": s["label"], "exams": len(mine),
                          "usage_pct": round(len(mine) / len(rooms_) * 100, 1) if rooms_ else 0})
    colors = {}
    for a in alloc:
        colors[a["color"]] = colors.get(a["color"], 0) + 1
    return jsonify({
        "has_run": run is not None, "nodes": n, "edges": e,
        "density": round(2 * e / (n * (n - 1)) * 100, 1) if n > 1 else 0,
        "max_degree": max((len(v) for v in adj.values()), default=0),
        "total_exams": n, "total_conflicts": e,
        "colors_used": run["summary"]["colors_used"] if run else 0,
        "students_total": sum(x["students"] for x in exams),
        "students_scheduled": sum(a["students"] for a in alloc),
        "room_utilization": room_util, "slot_utilization": slot_util,
        "avg_room_utilization": round(sum(r["usage_pct"] for r in room_util) / len(room_util), 1) if room_util else 0,
        "avg_seat_fill": round(sum(a["students"] / a["capacity"] for a in alloc) / len(alloc) * 100, 1) if alloc else 0,
        "color_distribution": [{"color": c, "exams": colors[c]} for c in sorted(colors)],
        "conflicts_before": before,
        "conflicts_after": run["summary"]["unresolved_conflicts"] if run else None,
    })


# ---------- demo / reset ----------
def wipe(db):
    for t in ("allocations", "algorithm_runs", "conflicts", "exams", "rooms", "timeslots"):
        db.execute(f"DELETE FROM {t}")
    db.execute("DELETE FROM sqlite_sequence")


@app.post("/api/reset")
def reset():
    db = get_db()
    wipe(db)
    db.commit()
    return jsonify({"ok": True})


@app.post("/api/demo")
def demo():
    db = get_db()
    wipe(db)
    exams = [
        ("CSE-A", "Data Structures", 60, "High", 180),
        ("CSE-B", "Database Management Systems", 55, "Medium", 180),
        ("ECE-A", "Computer Networks", 45, "High", 180),
        ("CSE-C", "Operating Systems", 62, "High", 180),
        ("AIML-A", "Artificial Intelligence", 48, "Medium", 180),
        ("IT-A", "Software Engineering", 40, "Low", 180),
        ("CSE-D", "Python Programming", 35, "Low", 180),
        ("ECE-B", "Computer Organization", 50, "Medium", 180),
    ]
    ids = {}
    for name, subj, st, pr, du in exams:
        ids[name] = db.execute("INSERT INTO exams (name, subject, students, priority, duration) VALUES (?,?,?,?,?)",
                               (name, subj, st, pr, du)).lastrowid
    for r in [("R101", "Room 101", 60, "CSE Block", 1), ("R102", "Room 102", 50, "CSE Block", 1),
              ("R201", "Room 201", 70, "Main Block", 2), ("R202", "Room 202", 65, "Main Block", 2)]:
        db.execute("INSERT INTO rooms (code, name, capacity, building, floor) VALUES (?,?,?,?,?)", r)
    for s in [("Slot 1", "Day 1", "09:00", "12:00"), ("Slot 2", "Day 1", "13:00", "16:00"),
              ("Slot 3", "Next Day", "09:00", "12:00")]:
        db.execute("INSERT INTO timeslots (label, day, start, end) VALUES (?,?,?,?)", s)
    # shared students / faculty -> conflicts (edges of the graph)
    pairs = [("CSE-C", "CSE-A"), ("CSE-C", "CSE-B"), ("CSE-C", "ECE-A"), ("CSE-C", "AIML-A"),
             ("CSE-A", "ECE-A"), ("CSE-A", "AIML-A"), ("CSE-B", "IT-A"), ("CSE-B", "ECE-B"),
             ("ECE-A", "CSE-D"), ("AIML-A", "IT-A"), ("IT-A", "CSE-D"), ("CSE-D", "ECE-B")]
    for a, b in pairs:
        lo, hi = sorted((ids[a], ids[b]))
        db.execute("INSERT INTO conflicts (exam_a, exam_b) VALUES (?,?)", (lo, hi))
    db.commit()
    return jsonify({"ok": True})


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return Response("Not found", status=404)


@app.errorhandler(Exception)
def server_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": f"Server error: {e}"}), 500
    raise e


# ==========================================================================================
# SECTION 5 - FRONTEND (React single page app, served from "/")
# ==========================================================================================
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Exam Room Allocation System | Graph Coloring + Greedy</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = { theme: { extend: {
  fontFamily: { sans: ['Inter','system-ui','sans-serif'], mono: ['JetBrains Mono','ui-monospace','monospace'] },
  colors: { navy: { 950:'#060b18', 900:'#0a1226', 800:'#0f1b38', 700:'#162549' } }
}}};
</script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script src="https://unpkg.com/lucide-react@0.263.1/dist/umd/lucide-react.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.8.2/jspdf.plugin.autotable.min.js"></script>
<style>
  html { scroll-behavior: smooth; }
  body { background: radial-gradient(1200px 600px at 80% -10%, #16306b55, transparent), #060b18; color: #e2e8f0; font-family: Inter, system-ui, sans-serif; }
  .glass { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(12px); box-shadow: 0 12px 32px -14px rgba(0,0,0,.6); }
  select option { background: #0a1226; color: #e2e8f0; }
  ::-webkit-scrollbar { height: 8px; width: 8px; } ::-webkit-scrollbar-thumb { background: #243457; border-radius: 8px; }
  .page-in { animation: pageIn .35s ease both; }
  @keyframes pageIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  .toast-in { animation: toastIn .25s ease both; } @keyframes toastIn { from { opacity: 0; transform: translateX(16px); } to { opacity: 1; transform: none; } }
  .modal-in { animation: modalIn .22s ease both; } @keyframes modalIn { from { opacity: 0; transform: translateY(14px) scale(.98); } to { opacity: 1; transform: none; } }
  .bar-grow { transform-origin: bottom; animation: grow .7s cubic-bezier(.2,.8,.2,1) both; } @keyframes grow { from { transform: scaleY(0); } to { transform: scaleY(1); } }
  .net-node { animation: netPulse 4s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
  @keyframes netPulse { 0%,100% { transform: scale(1); opacity: .7; } 50% { transform: scale(1.7); opacity: 1; } }
  .net-edge { stroke-dasharray: 4 7; animation: dash 10s linear infinite; } @keyframes dash { to { stroke-dashoffset: -110; } }
  .node-active { animation: ring 1.1s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
  @keyframes ring { 0%,100% { transform: scale(1); } 50% { transform: scale(1.12); } }
  .pop { animation: pop .45s cubic-bezier(.2,1.4,.4,1) both; } @keyframes pop { from { transform: scale(.6); opacity: 0; } to { transform: scale(1); opacity: 1; } }
  .skeleton { background: linear-gradient(90deg, rgba(255,255,255,.04), rgba(255,255,255,.1), rgba(255,255,255,.04)); background-size: 200% 100%; animation: shimmer 1.4s infinite; }
  @keyframes shimmer { to { background-position: -200% 0; } }
  .spin { animation: spin .8s linear infinite; } @keyframes spin { to { transform: rotate(360deg); } }
  .print-only { display: none; }
  @media print {
    body { background: #fff !important; }
    #app-shell, .no-print { display: none !important; }
    .print-only { display: block !important; }
    table { border-collapse: collapse; width: 100%; } th, td { border: 1px solid #999; padding: 5px 7px; font-size: 11px; text-align: left; }
  }
  @media (prefers-reduced-motion: reduce) { * { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
</style>
</head>
<body>
<div id="root"></div>
<noscript>This application needs JavaScript enabled.</noscript>

<script type="text/babel">
const { useState, useEffect, useMemo, useRef, useCallback, createContext, useContext } = React;

/* ============================== helpers ============================== */
const PALETTE = ['#38bdf8','#a78bfa','#34d399','#fbbf24','#fb7185','#f472b6','#2dd4bf','#fb923c','#818cf8','#a3e635'];
const colorOf = c => (c ? PALETTE[(c - 1) % PALETTE.length] : '#64748b');
const fmt12 = t => { if (!t) return ''; const [h, m] = t.split(':').map(Number); const ap = h >= 12 ? 'PM' : 'AM'; return `${String(h % 12 || 12).padStart(2, '0')}:${String(m).padStart(2, '0')} ${ap}`; };
const dayTag = d => (d && d.trim().toLowerCase() !== 'day 1' ? ` (${d})` : '');
const slotRange = (s, e, d) => (s ? `${fmt12(s)} – ${fmt12(e)}${dayTag(d)}` : '—');
const rowSlot = r => slotRange(r.slot_start, r.slot_end, r.slot_day);
const pdfSlot = r => (r.slot_start ? `${fmt12(r.slot_start)} - ${fmt12(r.slot_end)}${dayTag(r.slot_day)}` : '-');

async function api(path, opts = {}) {
  const res = await fetch('/api' + path, { headers: { 'Content-Type': 'application/json' }, ...opts, body: opts.body ? JSON.stringify(opts.body) : undefined });
  let data = null; try { data = await res.json(); } catch (e) { /* no body */ }
  if (!res.ok) { const err = new Error((data && data.error) || `Request failed (${res.status})`); err.details = data && data.errors; throw err; }
  return data;
}

/* ---- readiness checks (shown before running the algorithm) ---- */
function readiness(state) {
  const { exams, rooms, timeslots, conflicts } = state; const out = [];
  out.push(exams.length ? { level: 'ok', text: `${exams.length} exam(s) configured` } : { level: 'error', text: 'No exams added yet' });
  out.push(rooms.length ? { level: 'ok', text: `${rooms.length} room(s) configured` } : { level: 'error', text: 'No rooms added yet' });
  out.push(timeslots.length ? { level: 'ok', text: `${timeslots.length} time slot(s) configured` } : { level: 'error', text: 'No time slots added yet' });
  const maxCap = Math.max(0, ...rooms.map(r => r.capacity));
  if (rooms.length) exams.filter(e => e.students > maxCap).forEach(e => out.push({ level: 'warn', text: `Room capacity is insufficient for ${e.name} (${e.students} students, largest room ${maxCap})` }));
  if (exams.length && !conflicts.length) out.push({ level: 'warn', text: 'No conflicts defined. Every exam may be placed in the same slot.' });
  else if (conflicts.length) out.push({ level: 'ok', text: `${conflicts.length} conflict edge(s) defined` });
  return out;
}

/* ============================== context ============================== */
const AppCtx = createContext(null);
const useApp = () => useContext(AppCtx);

/* ============================== UI primitives ============================== */
function Icon({ name, size = 18, className = '', strokeWidth = 2 }) {
  const L = window.LucideReact; const C = L && L[name];
  if (!C) return <span style={{ width: size, height: size, display: 'inline-block' }} />;
  return <C size={size} className={className} strokeWidth={strokeWidth} />;
}
const Spinner = () => <span className="spin inline-block h-4 w-4 rounded-full border-2 border-white/30 border-t-white" />;
const Card = ({ className = '', children, ...p }) => <div className={`glass rounded-2xl ${className}`} {...p}>{children}</div>;

function Btn({ variant = 'primary', icon, children, className = '', loading, ...p }) {
  const styles = {
    primary: 'bg-sky-500 hover:bg-sky-400 text-white shadow-lg shadow-sky-500/20',
    ghost: 'bg-white/5 hover:bg-white/10 text-slate-200 border border-white/10',
    danger: 'bg-rose-500/90 hover:bg-rose-500 text-white',
    success: 'bg-emerald-500 hover:bg-emerald-400 text-white shadow-lg shadow-emerald-500/20',
  };
  return (
    <button {...p} disabled={p.disabled || loading}
      className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition active:scale-[.97] disabled:cursor-not-allowed disabled:opacity-50 ${styles[variant]} ${className}`}>
      {loading ? <Spinner /> : icon ? <Icon name={icon} size={16} /> : null}{children}
    </button>
  );
}
const IconBtn = ({ icon, title, onClick, danger }) => (
  <button title={title} aria-label={title} onClick={onClick}
    className={`rounded-lg p-2 transition active:scale-90 ${danger ? 'text-rose-300 hover:bg-rose-500/15' : 'text-slate-300 hover:bg-white/10'}`}><Icon name={icon} size={16} /></button>
);
const inputCls = 'w-full rounded-xl border border-white/10 bg-navy-900/80 px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/60';
const Field = ({ label, error, children }) => (
  <label className="block"><span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-slate-400">{label}</span>{children}
    {error && <span className="mt-1 block text-xs text-rose-400">{error}</span>}</label>
);
const ColorChip = ({ c, short }) => c
  ? <span className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold" style={{ background: colorOf(c) + '22', color: colorOf(c), border: `1px solid ${colorOf(c)}55` }}>{short ? `C${c}` : `Color ${c}`}</span>
  : <span className="text-slate-500">—</span>;
const StatusBadge = ({ status }) => status === 'Conflict Free'
  ? <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2.5 py-1 text-xs font-semibold text-emerald-300">✓ Conflict Free</span>
  : <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 px-2.5 py-1 text-xs font-semibold text-amber-300">⚠ Requires Attention</span>;
const PriorityBadge = ({ p }) => {
  const m = { High: 'bg-rose-500/15 text-rose-300', Medium: 'bg-amber-500/15 text-amber-300', Low: 'bg-emerald-500/15 text-emerald-300' };
  return <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${m[p] || ''}`}>{p}</span>;
};

function PageHeader({ icon, title, subtitle, actions }) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 rounded-xl bg-sky-500/15 p-2.5 text-sky-300"><Icon name={icon} size={22} /></div>
        <div><h1 className="text-xl font-bold tracking-tight text-white sm:text-2xl">{title}</h1>{subtitle && <p className="mt-1 max-w-2xl text-sm text-slate-400">{subtitle}</p>}</div>
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}
function EmptyState({ icon = 'Inbox', title, text, action }) {
  return (
    <Card className="flex flex-col items-center px-6 py-12 text-center">
      <div className="mb-4 rounded-2xl bg-white/5 p-4 text-slate-400"><Icon name={icon} size={30} /></div>
      <h3 className="text-base font-semibold text-white">{title}</h3>
      {text && <p className="mt-1.5 max-w-md whitespace-pre-line text-sm text-slate-400">{text}</p>}
      {action && <div className="mt-5">{action}</div>}
    </Card>
  );
}
function Modal({ title, onClose, children }) {
  useEffect(() => { const h = e => e.key === 'Escape' && onClose(); window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h); }, [onClose]);
  return (
    <div className="no-print fixed inset-0 z-50 flex items-end justify-center bg-black/60 backdrop-blur-sm sm:items-center sm:p-4" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal-in max-h-[92vh] w-full overflow-y-auto rounded-t-3xl border border-white/10 bg-navy-800 p-6 shadow-2xl sm:max-w-lg sm:rounded-3xl">
        <div className="mb-5 flex items-center justify-between"><h2 className="text-lg font-bold text-white">{title}</h2><IconBtn icon="X" title="Close" onClick={onClose} /></div>
        {children}
      </div>
    </div>
  );
}
function ToastHost({ toasts }) {
  const cls = { error: 'border-rose-400/30 bg-rose-950/85 text-rose-100', success: 'border-emerald-400/30 bg-emerald-950/85 text-emerald-100', info: 'border-sky-400/30 bg-slate-900/90 text-sky-100' };
  return (
    <div className="no-print fixed bottom-4 right-4 z-[70] flex w-[min(92vw,380px)] flex-col gap-2">
      {toasts.map(t => (
        <div key={t.id} className={`toast-in flex items-start gap-3 rounded-xl border px-4 py-3 text-sm shadow-xl backdrop-blur ${cls[t.type] || cls.info}`}>
          <Icon name={t.type === 'error' ? 'AlertTriangle' : t.type === 'success' ? 'CheckCircle2' : 'Info'} size={18} className="mt-0.5 shrink-0" /><span>{t.msg}</span>
        </div>))}
    </div>
  );
}
function ConfirmDialog({ cfg, onClose }) {
  if (!cfg) return null;
  return (
    <Modal title={cfg.title} onClose={() => onClose(false)}>
      <p className="whitespace-pre-line text-sm text-slate-300">{cfg.message}</p>
      <div className="mt-6 flex justify-end gap-2">
        <Btn variant="ghost" onClick={() => onClose(false)}>Cancel</Btn>
        <Btn variant={cfg.danger ? 'danger' : 'primary'} onClick={() => onClose(true)}>{cfg.confirmText || 'Confirm'}</Btn>
      </div>
    </Modal>
  );
}
function StatCard({ icon, label, value, hint, tone = 'sky' }) {
  const tones = { sky: 'text-sky-300 bg-sky-500/15', violet: 'text-violet-300 bg-violet-500/15', emerald: 'text-emerald-300 bg-emerald-500/15', amber: 'text-amber-300 bg-amber-500/15', rose: 'text-rose-300 bg-rose-500/15' };
  return (
    <Card className="p-4 transition hover:-translate-y-0.5 hover:border-white/20">
      <div className="flex items-center justify-between"><span className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</span><span className={`rounded-lg p-1.5 ${tones[tone]}`}><Icon name={icon} size={16} /></span></div>
      <div className="mt-2 text-3xl font-extrabold tracking-tight text-white">{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </Card>
  );
}
function BarChart({ data, unit = '', max, height = 170 }) {
  const m = max ?? Math.max(1, ...data.map(d => d.value));
  return (
    <div className="flex items-end gap-3 overflow-x-auto pb-1" style={{ height: height + 40 }}>
      {data.map((d, i) => (
        <div key={i} className="flex min-w-[48px] flex-1 flex-col items-center justify-end gap-1.5">
          <span className="text-xs font-semibold text-slate-200">{d.value}{unit}</span>
          <div className="bar-grow w-full max-w-[64px] rounded-t-lg" style={{ height: Math.max(4, (d.value / m) * height), background: d.color || '#38bdf8' }} />
          <span className="max-w-full truncate text-[11px] text-slate-400" title={d.label}>{d.label}</span>
        </div>))}
    </div>
  );
}
const Skeleton = ({ className = '' }) => <div className={`skeleton rounded-2xl ${className}`} />;

/* ============================== export helpers ============================== */
function exportCSV(rows) {
  const head = ['Exam', 'Subject', 'Students', 'Room', 'Building', 'Time Slot', 'Color', 'Status'];
  const esc = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [head.map(esc).join(','), ...rows.map(r => [r.exam, r.subject, r.students, r.room || '', r.building || '', pdfSlot(r), r.color ? 'Color ' + r.color : '', r.status].map(esc).join(','))];
  const blob = new Blob(['\ufeff' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'exam_allocation.csv'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
}
function exportPDF(state, toast) {
  const lib = window.jspdf;
  if (!lib || !lib.jsPDF) { toast('error', 'PDF library could not be loaded (check your internet connection). Use Print instead.'); return; }
  const { run, allocation, rooms, timeslots } = state;
  const doc = new lib.jsPDF({ orientation: 'landscape' }); const W = doc.internal.pageSize.getWidth();
  doc.setFont('helvetica', 'bold'); doc.setFontSize(17); doc.text('MALLA REDDY VISHVAVIDHYAPEETH', W / 2, 15, { align: 'center' });
  doc.setFontSize(13); doc.text('EXAM ROOM ALLOCATION SYSTEM', W / 2, 23, { align: 'center' });
  doc.setFont('helvetica', 'normal'); doc.setFontSize(10);
  doc.text('Graph Coloring + Greedy Algorithm', W / 2, 29, { align: 'center' });
  doc.text('Student: THOKALA SHESHVITH', W / 2, 35, { align: 'center' });
  doc.text(`Generated: ${new Date().toLocaleString()}   |   Exams: ${run.summary.total_exams}   Conflicts: ${run.summary.total_conflicts}   Colors used: ${run.summary.colors_used}   Status: ${run.summary.status}`, W / 2, 41, { align: 'center' });
  doc.autoTable({
    startY: 46, styles: { fontSize: 9 }, headStyles: { fillColor: [15, 27, 56] },
    head: [['Exam', 'Subject', 'Students', 'Room', 'Building', 'Time Slot', 'Color', 'Status']],
    body: allocation.map(r => [r.exam, r.subject, r.students, r.room || '-', r.building || '-', pdfSlot(r), r.color ? 'Color ' + r.color : '-', r.status]),
  });
  doc.addPage();
  doc.setFont('helvetica', 'bold'); doc.setFontSize(13); doc.text('Timetable', 14, 16);
  doc.autoTable({
    startY: 21, styles: { fontSize: 9 }, headStyles: { fillColor: [15, 27, 56] },
    head: [['Time', ...rooms.map(r => r.name)]],
    body: timeslots.map(s => [`${s.label}\n${fmt12(s.start)} - ${fmt12(s.end)}${dayTag(s.day)}`, ...rooms.map(r => { const a = allocation.find(x => x.slot_id === s.id && x.room_id === r.id); return a ? `${a.exam}\n${a.subject}` : '-'; })]),
  });
  doc.save('exam_timetable.pdf');
}

/* ============================== conflict graph canvas ============================== */
const GW = 800, GH = 520;
function layoutCircle(ids) {
  const cx = GW / 2, cy = GH / 2, r = Math.min(GW, GH) / 2 - 80, out = {};
  ids.forEach((id, i) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(ids.length, 1); out[id] = { x: cx + r * 1.3 * Math.cos(a), y: cy + r * Math.sin(a) }; });
  return out;
}
function GraphCanvas({ exams, conflicts, colors = {}, activeId = null, selectedId = null, highlightIds = [], onSelect }) {
  const { graphPos, setGraphPos } = useApp();
  const svgRef = useRef(null); const drag = useRef(null);
  const ids = exams.map(e => e.id);
  const base = useMemo(() => layoutCircle(ids), [ids.join(',')]);
  const P = id => graphPos[id] || base[id] || { x: GW / 2, y: GH / 2 };
  const focus = activeId ?? selectedId;
  const toSvg = e => { const pt = svgRef.current.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svgRef.current.getScreenCTM().inverse()); };
  const down = (e, id) => { e.preventDefault(); svgRef.current.setPointerCapture(e.pointerId); drag.current = id; onSelect && onSelect(id); };
  const move = e => { if (drag.current == null) return; const p = toSvg(e); const id = drag.current; setGraphPos(prev => ({ ...prev, [id]: { x: Math.max(30, Math.min(GW - 30, p.x)), y: Math.max(30, Math.min(GH - 40, p.y)) } })); };
  const up = () => { drag.current = null; };
  return (
    <div>
      <svg ref={svgRef} viewBox={`0 0 ${GW} ${GH}`} className="h-auto w-full select-none rounded-2xl bg-navy-900/60" onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
        <defs><pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="rgba(255,255,255,.04)" /></pattern></defs>
        <rect width={GW} height={GH} fill="url(#grid)" />
        {conflicts.map(c => {
          const a = P(c.exam_a), b = P(c.exam_b); const hot = focus === c.exam_a || focus === c.exam_b;
          const bad = colors[c.exam_a] && colors[c.exam_a] === colors[c.exam_b];
          return <line key={c.id} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={bad ? '#f43f5e' : hot ? '#e2e8f0' : '#475569'} strokeWidth={hot || bad ? 2.4 : 1.4} style={{ transition: 'stroke .3s' }} />;
        })}
        {exams.map(ex => {
          const p = P(ex.id), c = colors[ex.id], isActive = activeId === ex.id, isSel = selectedId === ex.id, isNb = highlightIds.includes(ex.id);
          return (
            <g key={ex.id} transform={`translate(${p.x},${p.y})`} style={{ cursor: 'grab', touchAction: 'none' }} onPointerDown={e => down(e, ex.id)}>
              <g className={isActive ? 'node-active' : ''}>
                {(isActive || isSel) && <circle r="34" fill="none" stroke={c ? colorOf(c) : '#e2e8f0'} strokeOpacity=".55" strokeWidth="3" />}
                {isNb && <circle r="33" fill="none" stroke="#fbbf24" strokeDasharray="4 4" strokeWidth="2" />}
                <circle r="27" style={{ fill: c ? colorOf(c) : '#0f1b38', stroke: c ? 'rgba(255,255,255,.7)' : '#64748b', transition: 'fill .5s, stroke .3s' }} strokeWidth="2" strokeDasharray={c ? '0' : '5 4'} />
                <text textAnchor="middle" dy="4" fontSize={ex.name.length > 6 ? 10 : 12} fontWeight="700" fill={c ? '#06101f' : '#e2e8f0'} pointerEvents="none">{ex.name.length > 9 ? ex.name.slice(0, 8) + '…' : ex.name}</text>
                {c && <g className="pop"><circle cx="21" cy="-21" r="10" fill="#06101f" stroke="#fff" strokeOpacity=".6" /><text x="21" y="-17.5" textAnchor="middle" fontSize="10" fontWeight="700" fill="#fff" pointerEvents="none">{c}</text></g>}
              </g>
              <text y="46" textAnchor="middle" fontSize="11" fill="#94a3b8" pointerEvents="none">{ex.subject.length > 20 ? ex.subject.slice(0, 19) + '…' : ex.subject}</text>
            </g>);
        })}
      </svg>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-slate-400">
        <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rounded-full border border-dashed border-slate-400 bg-navy-800" />Unassigned</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rounded-full bg-sky-400" />Assigned (fill = color)</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-5 bg-slate-500" />Conflict edge</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-5 bg-rose-500" />Violation (same color)</span>
        <span className="text-slate-500">Drag nodes to rearrange · tap a node for details</span>
      </div>
    </div>
  );
}
function NodeDetails({ examId }) {
  const { state } = useApp();
  const ex = state.exams.find(e => e.id === examId);
  if (!ex) return <Card className="p-5 text-sm text-slate-400">Select a node in the graph to see its details.</Card>;
  const nbrs = state.conflicts.filter(c => c.exam_a === ex.id || c.exam_b === ex.id).map(c => (c.exam_a === ex.id ? c.b_name : c.a_name));
  const al = state.allocation.find(a => a.exam_id === ex.id);
  const row = (k, v) => <div className="flex justify-between gap-3 border-b border-white/5 py-2 text-sm"><span className="text-slate-400">{k}</span><span className="text-right font-medium text-slate-100">{v}</span></div>;
  return (
    <Card className="p-5">
      <h3 className="mb-1 text-base font-bold text-white">{ex.name}</h3>
      {row('Subject', ex.subject)}{row('Students', ex.students)}{row('Degree (conflicts)', nbrs.length)}
      <div className="border-b border-white/5 py-2 text-sm"><div className="mb-1.5 text-slate-400">Conflicting exams</div>
        <div className="flex flex-wrap gap-1.5">{nbrs.length ? nbrs.map(n => <span key={n} className="rounded-md bg-white/10 px-2 py-0.5 text-xs">{n}</span>) : <span className="text-slate-500">None</span>}</div></div>
      {row('Assigned color / slot', al && al.color ? <><ColorChip c={al.color} /> <span className="text-slate-400">· {al.slot}</span></> : 'Not assigned')}
      {row('Assigned room', al && al.room ? `${al.room} (${al.building})` : 'Not assigned')}
    </Card>
  );
}

/* ============================== conflict manager ============================== */
function ConflictManager() {
  const { state, refresh, toast } = useApp();
  const [a, setA] = useState(''); const [b, setB] = useState(''); const [err, setErr] = useState('');
  const add = async () => {
    setErr('');
    if (!a || !b) return setErr('Select both Exam A and Exam B.');
    if (a === b) return setErr('An exam cannot conflict with itself.');
    const x = +a, y = +b;
    if (state.conflicts.some(c => (c.exam_a === x && c.exam_b === y) || (c.exam_a === y && c.exam_b === x))) return setErr('This conflict already exists. Duplicate conflicts are not allowed.');
    try { await api('/conflicts', { method: 'POST', body: { exam_a: x, exam_b: y } }); setA(''); setB(''); await refresh(); toast('success', 'Conflict added.'); }
    catch (e) { setErr(e.message); }
  };
  const del = async id => { try { await api('/conflicts/' + id, { method: 'DELETE' }); await refresh(); toast('info', 'Conflict removed.'); } catch (e) { toast('error', e.message); } };
  return (
    <Card className="p-5">
      <div className="mb-4 flex items-center justify-between"><h3 className="font-bold text-white">Create conflict</h3><span className="rounded-full bg-sky-500/15 px-2.5 py-1 text-xs font-semibold text-sky-300">{state.conflicts.length} edge(s)</span></div>
      {state.exams.length < 2 ? <p className="text-sm text-slate-400">Add at least two exams to create a conflict.</p> : (
        <div className="space-y-3">
          <Field label="Exam A"><select className={inputCls} value={a} onChange={e => setA(e.target.value)}><option value="">Select exam…</option>{state.exams.map(e => <option key={e.id} value={e.id}>{e.name} · {e.subject}</option>)}</select></Field>
          <Field label="Exam B"><select className={inputCls} value={b} onChange={e => setB(e.target.value)}><option value="">Select exam…</option>{state.exams.map(e => <option key={e.id} value={e.id}>{e.name} · {e.subject}</option>)}</select></Field>
          {err && <p className="flex items-center gap-1.5 text-sm text-rose-400"><Icon name="AlertTriangle" size={14} />{err}</p>}
          <Btn icon="Link2" onClick={add} className="w-full">ADD CONFLICT</Btn>
        </div>)}
      <div className="mt-5 max-h-64 space-y-1.5 overflow-y-auto pr-1">
        {state.conflicts.length === 0 ? <p className="whitespace-pre-line rounded-xl bg-white/5 p-3 text-sm text-slate-400">No conflicts defined.{'\n'}Add conflicts to build the graph.</p> :
          state.conflicts.map(c => (
            <div key={c.id} className="flex items-center justify-between rounded-lg bg-white/5 px-3 py-2 text-sm"><span>{c.a_name} <span className="text-slate-500">────</span> {c.b_name}</span><IconBtn icon="Trash2" title="Remove conflict" danger onClick={() => del(c.id)} /></div>))}
      </div>
    </Card>
  );
}

/* ============================== generic CRUD page ============================== */
function CrudPage({ icon, title, subtitle, endpoint, items, fields, columns, defaults, validate, emptyTitle, emptyText, addLabel, deleteMessage }) {
  const { refresh, toast, confirm } = useApp();
  const [editing, setEditing] = useState(null);
  const [vals, setVals] = useState({}); const [errs, setErrs] = useState({}); const [serverErrs, setServerErrs] = useState([]); const [saving, setSaving] = useState(false);
  const open = item => { setEditing(item || {}); setVals(item ? { ...item } : { ...defaults }); setErrs({}); setServerErrs([]); };
  const save = async e => {
    e.preventDefault();
    const v = validate(vals, items, editing.id); setErrs(v);
    if (Object.keys(v).length) return;
    setSaving(true); setServerErrs([]);
    try {
      await api(endpoint + (editing.id ? '/' + editing.id : ''), { method: editing.id ? 'PUT' : 'POST', body: vals });
      await refresh(); toast('success', editing.id ? 'Changes saved. Re-run the algorithm to refresh the allocation.' : 'Added successfully.'); setEditing(null);
    } catch (er) { setServerErrs(er.details || [er.message]); } finally { setSaving(false); }
  };
  const del = async item => {
    const ok = await confirm({ title: 'Delete item?', message: deleteMessage(item), confirmText: 'Delete', danger: true });
    if (!ok) return;
    try { await api(endpoint + '/' + item.id, { method: 'DELETE' }); await refresh(); toast('info', 'Deleted.'); } catch (er) { toast('error', er.message); }
  };
  return (
    <div>
      <PageHeader icon={icon} title={title} subtitle={subtitle} actions={<Btn icon="Plus" onClick={() => open()}>{addLabel}</Btn>} />
      {items.length === 0 ? <EmptyState icon={icon} title={emptyTitle} text={emptyText} action={<Btn icon="Plus" onClick={() => open()}>{addLabel}</Btn>} /> : (<>
        <Card className="hidden overflow-x-auto md:block">
          <table className="w-full text-left text-sm"><thead><tr className="border-b border-white/10 text-xs uppercase tracking-wide text-slate-400">
            {columns.map(c => <th key={c.label} className="px-4 py-3 font-medium">{c.label}</th>)}<th className="px-4 py-3 text-right font-medium">Actions</th></tr></thead>
            <tbody>{items.map(it => (
              <tr key={it.id} className="border-b border-white/5 transition hover:bg-white/[.03]">
                {columns.map(c => <td key={c.label} className="px-4 py-3">{c.render(it)}</td>)}
                <td className="px-4 py-3 text-right"><IconBtn icon="Pencil" title="Edit" onClick={() => open(it)} /><IconBtn icon="Trash2" title="Delete" danger onClick={() => del(it)} /></td></tr>))}</tbody></table>
        </Card>
        <div className="grid gap-3 md:hidden">{items.map(it => (
          <Card key={it.id} className="p-4">
            <div className="flex items-start justify-between"><div className="text-base font-bold text-white">{columns[0].render(it)}</div><div className="-mr-2 -mt-2 flex"><IconBtn icon="Pencil" title="Edit" onClick={() => open(it)} /><IconBtn icon="Trash2" title="Delete" danger onClick={() => del(it)} /></div></div>
            <div className="mt-2 space-y-1.5">{columns.slice(1).map(c => <div key={c.label} className="flex justify-between gap-3 text-sm"><span className="text-slate-400">{c.label}</span><span className="text-right">{c.render(it)}</span></div>)}</div>
          </Card>))}</div>
      </>)}
      {editing && (
        <Modal title={(editing.id ? 'Edit ' : 'Add ') + title.replace(/s$/, '').replace('Time Slot', 'Time Slot')} onClose={() => setEditing(null)}>
          <form onSubmit={save} className="space-y-4" noValidate>
            {serverErrs.length > 0 && <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 p-3 text-sm text-rose-200">{serverErrs.map((m, i) => <div key={i}>• {m}</div>)}</div>}
            {fields.map(f => (
              <Field key={f.key} label={f.label} error={errs[f.key]}>
                {f.type === 'select'
                  ? <select className={inputCls} value={vals[f.key] ?? ''} onChange={e => setVals({ ...vals, [f.key]: e.target.value })}>{f.options.map(o => <option key={o}>{o}</option>)}</select>
                  : <input className={inputCls} type={f.type || 'text'} min={f.min} placeholder={f.placeholder} value={vals[f.key] ?? ''} onChange={e => setVals({ ...vals, [f.key]: e.target.value })} />}
              </Field>))}
            <div className="flex justify-end gap-2 pt-2"><Btn type="button" variant="ghost" onClick={() => setEditing(null)}>Cancel</Btn><Btn type="submit" icon="Check" loading={saving}>{editing.id ? 'Save changes' : addLabel}</Btn></div>
          </form>
        </Modal>)}
    </div>
  );
}
const dupBy = (items, key, v, id) => items.some(i => i.id !== id && String(i[key]).trim().toLowerCase() === String(v).trim().toLowerCase());
const isPosInt = v => /^\d+$/.test(String(v).trim()) && +v > 0;

function ExamsPage() {
  const { state } = useApp();
  const deg = id => state.conflicts.filter(c => c.exam_a === id || c.exam_b === id).length;
  return <CrudPage icon="BookOpen" title="Exams" endpoint="/exams" items={state.exams} addLabel="Add Exam"
    subtitle="Each exam/class is a vertex of the conflict graph."
    emptyTitle="No exams added yet." emptyText={'No exams added yet.\nAdd your first exam to begin building the conflict graph.'}
    defaults={{ name: '', subject: '', students: '', priority: 'Medium', duration: 180 }}
    fields={[
      { key: 'name', label: 'Exam / Class Name', placeholder: 'e.g. CSE-A' },
      { key: 'subject', label: 'Subject', placeholder: 'e.g. Data Structures' },
      { key: 'students', label: 'Student Count', type: 'number', min: 1, placeholder: 'e.g. 60' },
      { key: 'priority', label: 'Priority', type: 'select', options: ['High', 'Medium', 'Low'] },
      { key: 'duration', label: 'Duration (minutes)', type: 'number', min: 15, placeholder: 'e.g. 180' }]}
    columns={[
      { label: 'Exam / Class', render: e => <span className="font-semibold text-white">{e.name}</span> },
      { label: 'Subject', render: e => e.subject },
      { label: 'Students', render: e => e.students },
      { label: 'Priority', render: e => <PriorityBadge p={e.priority} /> },
      { label: 'Duration', render: e => `${e.duration} min` },
      { label: 'Degree', render: e => <span className="font-mono text-sky-300">{deg(e.id)}</span> }]}
    validate={(v, items, id) => {
      const er = {};
      if (!String(v.name).trim()) er.name = 'Exam/Class name is required.'; else if (dupBy(items, 'name', v.name, id)) er.name = 'An exam with this name already exists.';
      if (!String(v.subject).trim()) er.subject = 'Subject is required.';
      if (!isPosInt(v.students)) er.students = 'Student count must be a positive whole number.';
      if (!isPosInt(v.duration) || +v.duration < 15 || +v.duration > 600) er.duration = 'Duration must be 15–600 minutes.';
      return er;
    }}
    deleteMessage={e => `Delete exam "${e.name}"?\nAll conflicts involving it will be removed and the current allocation will be cleared.`} />;
}
function RoomsPage() {
  const { state } = useApp();
  return <CrudPage icon="DoorOpen" title="Rooms" endpoint="/rooms" items={state.rooms} addLabel="Add Room"
    subtitle="Rooms are the physical resources. An exam is only placed in a room with enough seats."
    emptyTitle="No rooms added yet." emptyText={'No rooms added yet.\nAdd examination rooms so exams can be placed in them.'}
    defaults={{ code: '', name: '', capacity: '', building: '', floor: 0 }}
    fields={[
      { key: 'code', label: 'Room ID', placeholder: 'e.g. R101' },
      { key: 'name', label: 'Room Name / Number', placeholder: 'e.g. Room 101' },
      { key: 'capacity', label: 'Capacity (seats)', type: 'number', min: 1, placeholder: 'e.g. 60' },
      { key: 'building', label: 'Building', placeholder: 'e.g. CSE Block' },
      { key: 'floor', label: 'Floor', type: 'number', min: 0, placeholder: 'e.g. 1' }]}
    columns={[
      { label: 'Room', render: r => <span className="font-semibold text-white">{r.name}</span> },
      { label: 'Room ID', render: r => <span className="font-mono text-slate-300">{r.code}</span> },
      { label: 'Capacity', render: r => `${r.capacity} seats` },
      { label: 'Building', render: r => r.building },
      { label: 'Floor', render: r => r.floor }]}
    validate={(v, items, id) => {
      const er = {};
      if (!String(v.code).trim()) er.code = 'Room ID is required.'; else if (dupBy(items, 'code', v.code, id)) er.code = 'This Room ID already exists.';
      if (!String(v.name).trim()) er.name = 'Room name is required.'; else if (dupBy(items, 'name', v.name, id)) er.name = 'A room with this name already exists.';
      if (!isPosInt(v.capacity)) er.capacity = 'Capacity must be greater than zero.';
      if (!String(v.building).trim()) er.building = 'Building is required.';
      if (!/^\d+$/.test(String(v.floor).trim())) er.floor = 'Floor must be 0 or a positive number.';
      return er;
    }}
    deleteMessage={r => `Delete room "${r.name}"?\nThe current allocation will be cleared.`} />;
}
function SlotsPage() {
  const { state } = useApp();
  const mins = t => +t.slice(0, 2) * 60 + +t.slice(3, 5);
  return <CrudPage icon="Clock" title="Time Slots" endpoint="/timeslots" items={state.timeslots} addLabel="Add Time Slot"
    subtitle="The k-th time slot is the k-th color of the graph. Slots on the same day must not overlap."
    emptyTitle="No time slots added yet." emptyText={'No time slots added yet.\nDefine examination periods (colors) for the algorithm to use.'}
    defaults={{ label: '', day: 'Day 1', start: '09:00', end: '12:00' }}
    fields={[
      { key: 'label', label: 'Slot Name', placeholder: 'e.g. Slot 1' },
      { key: 'day', label: 'Day', placeholder: 'e.g. Day 1 or Next Day' },
      { key: 'start', label: 'Start Time', type: 'time' },
      { key: 'end', label: 'End Time', type: 'time' }]}
    columns={[
      { label: 'Slot', render: s => <span className="font-semibold text-white">{s.label}</span> },
      { label: 'Day', render: s => s.day },
      { label: 'Time', render: s => `${fmt12(s.start)} – ${fmt12(s.end)}` },
      { label: 'Length', render: s => `${mins(s.end) - mins(s.start)} min` }]}
    validate={(v, items, id) => {
      const er = {};
      if (!String(v.label).trim()) er.label = 'Slot name is required.'; else if (dupBy(items, 'label', v.label, id)) er.label = 'A slot with this name already exists.';
      if (!v.start) er.start = 'Start time is required.';
      if (!v.end) er.end = 'End time is required.'; else if (v.start && mins(v.end) <= mins(v.start)) er.end = 'End time must be after start time.';
      return er;
    }}
    deleteMessage={s => `Delete time slot "${s.label}"?\nThe current allocation will be cleared.`} />;
}

/* ============================== shared result components ============================== */
const NoRun = ({ navigate }) => <EmptyState icon="Play" title="No allocation yet" text="Run the Greedy Graph Coloring algorithm to generate an allocation." action={<Btn icon="Play" onClick={() => navigate('run')}>Go to Run Algorithm</Btn>} />;

function FailureBanner({ run }) {
  if (!run || run.success) return null;
  return (
    <div className="mb-6 rounded-2xl border border-rose-400/30 bg-rose-500/10 p-5">
      <div className="flex items-center gap-2 text-lg font-bold text-rose-200"><Icon name="AlertTriangle" size={22} />Allocation Could Not Be Completed</div>
      <ul className="mt-3 space-y-1.5 text-sm text-rose-100/90">{run.failures.map(f => <li key={f.exam_id}>• {f.reason}</li>)}</ul>
      {run.suggestions.length > 0 && <div className="mt-4"><div className="text-xs font-semibold uppercase tracking-wide text-rose-200/70">Suggested actions</div>
        <div className="mt-2 flex flex-wrap gap-2">{run.suggestions.map(s => <span key={s} className="rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-white">{s}</span>)}</div></div>}
    </div>
  );
}
function AllocationLog({ steps }) {
  if (!steps.length) return <EmptyState icon="ListOrdered" title="No decisions recorded yet" text="Run the Greedy Graph Coloring algorithm to generate an allocation." />;
  const th = 'px-3 py-3 font-medium whitespace-nowrap';
  return (
    <div className="overflow-x-auto rounded-2xl border border-white/10 glass">
      <table className="w-full min-w-[1000px] text-left text-sm">
        <thead><tr className="border-b border-white/10 text-xs uppercase tracking-wide text-slate-400">
          {['Step', 'Exam', 'Degree', 'Conflicting Exams', 'Unavailable Colors', 'Selected Color', 'Room', 'Time Slot', 'Reason'].map(h => <th key={h} className={th}>{h}</th>)}</tr></thead>
        <tbody>{steps.map(s => (
          <tr key={s.step} className={`border-b border-white/5 align-top ${s.success ? '' : 'bg-rose-500/10'}`}>
            <td className="px-3 py-3 font-mono text-slate-300">{s.step}</td>
            <td className="px-3 py-3"><div className="font-semibold text-white">{s.exam}</div><div className="text-xs text-slate-500">{s.subject}</div></td>
            <td className="px-3 py-3 font-mono text-sky-300">{s.degree}</td>
            <td className="px-3 py-3 text-slate-300">{s.neighbours.length ? s.neighbours.map(n => n.name).join(', ') : 'None'}</td>
            <td className="px-3 py-3"><div className="flex flex-wrap gap-1">{s.unavailable_colors.length ? s.unavailable_colors.map(c => <ColorChip key={c} c={c} short />) : <span className="text-slate-500">None</span>}</div></td>
            <td className="px-3 py-3">{s.selected_color ? <ColorChip c={s.selected_color} /> : <span className="font-semibold text-rose-300">Failed</span>}</td>
            <td className="px-3 py-3 whitespace-nowrap">{s.room || '—'}</td>
            <td className="px-3 py-3 whitespace-nowrap">{s.slot_start ? slotRange(s.slot_start, s.slot_end, s.slot_day) : '—'}</td>
            <td className="min-w-[260px] px-3 py-3 text-xs text-slate-400">{s.reason}</td>
          </tr>))}</tbody>
      </table>
    </div>
  );
}
function ResultPanel({ run }) {
  const s = run.summary, v = run.verification, free = s.status === 'CONFLICT FREE';
  const items = [['Total Exams', s.total_exams], ['Total Conflicts', s.total_conflicts], ['Colors Used', s.colors_used], ['Unresolved Conflicts', s.unresolved_conflicts], ['Rooms Used', s.rooms_used], ['Time Slots Used', s.slots_used], ['Room Capacity Violations', s.capacity_violations]];
  return (
    <div className="space-y-6">
      <Card className="p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><h3 className="text-lg font-bold text-white">Optimization Result</h3>
          <span className={`rounded-full px-4 py-1.5 text-sm font-bold tracking-wide ${free ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-300'}`}>Allocation Status: {s.status}</span></div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">{items.map(([k, val]) => (
          <div key={k} className="rounded-xl bg-white/5 p-3"><div className="text-[11px] uppercase tracking-wide text-slate-400">{k}</div><div className="mt-1 text-2xl font-extrabold text-white">{val}</div></div>))}</div>
      </Card>
      <Card className="p-5">
        <h3 className="mb-3 text-lg font-bold text-white">Verification Result</h3>
        {v.passed ? <p className="flex items-center gap-2 text-base font-semibold text-emerald-300"><Icon name="CheckCircle2" size={20} />✓ All conflict constraints satisfied</p>
          : <div className="space-y-1 text-sm font-semibold text-amber-300"><p>{v.conflict_count > 0 ? `✕ ${v.conflict_count} conflicts detected` : `✓ No edge conflicts (0 conflicts detected)`}</p>
            {v.capacity_violation_count > 0 && <p>✕ {v.capacity_violation_count} room capacity violation(s)</p>}{v.room_clash_count > 0 && <p>✕ {v.room_clash_count} room double-booking(s)</p>}
            {v.unassigned.length > 0 && <p>✕ Unassigned: {v.unassigned.join(', ')}</p>}</div>}
        <p className="mt-2 text-xs text-slate-500">For every edge (A, B): if color[A] == color[B] → conflict, otherwise valid. {v.edges_checked} edge(s) checked against the generated allocation.</p>
        {v.edge_results.length > 0 && (
          <details className="mt-4 rounded-xl bg-white/5 p-3"><summary className="cursor-pointer text-sm font-medium text-slate-300">Show edge-by-edge verification</summary>
            <div className="mt-3 overflow-x-auto"><table className="w-full min-w-[460px] text-left text-sm"><thead><tr className="text-xs uppercase text-slate-400"><th className="py-1.5">Edge (A — B)</th><th>color[A]</th><th>color[B]</th><th>Result</th></tr></thead>
              <tbody>{v.edge_results.map((e, i) => <tr key={i} className="border-t border-white/5"><td className="py-1.5">{e.a} — {e.b}</td><td><ColorChip c={e.color_a} short /></td><td><ColorChip c={e.color_b} short /></td><td className={e.valid ? 'text-emerald-300' : 'text-rose-300'}>{e.valid ? '✓ valid' : '✕ conflict'}</td></tr>)}</tbody></table></div></details>)}
      </Card>
    </div>
  );
}
function WhyGreedy() {
  return (
    <Card className="border-sky-400/20 bg-sky-500/5 p-5">
      <div className="mb-2 flex items-center gap-2 font-bold text-sky-200"><Icon name="Lightbulb" size={18} />Why Greedy?</div>
      <p className="text-sm leading-relaxed text-slate-300">The algorithm makes the best locally feasible decision at each step. For every exam, it chooses the first available color that does not conflict with already colored neighboring exams.</p>
      <p className="mt-2 rounded-lg bg-white/5 p-3 text-sm font-medium text-slate-100">Greedy is fast and simple, but the number of colors used can depend on the order in which exams are processed.</p>
    </Card>
  );
}

/* ============================== DASHBOARD ============================== */
function HeroNetwork() {
  const nodes = [[60, 60], [200, 140], [340, 50], [480, 150], [620, 70], [760, 140], [900, 60], [140, 260], [300, 230], [460, 280], [640, 240], [820, 270]];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [1, 8], [8, 9], [3, 9], [9, 10], [10, 5], [7, 1], [7, 8], [10, 11], [5, 11]];
  return (
    <svg className="absolute inset-0 h-full w-full opacity-40" viewBox="0 0 960 320" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      {edges.map(([a, b], i) => <line key={i} x1={nodes[a][0]} y1={nodes[a][1]} x2={nodes[b][0]} y2={nodes[b][1]} stroke="#38bdf8" strokeOpacity=".4" strokeWidth="1.2" className="net-edge" />)}
      {nodes.map(([x, y], i) => <circle key={i} cx={x} cy={y} r="5" fill={PALETTE[i % 5]} className="net-node" style={{ animationDelay: `${i * 0.35}s` }} />)}
    </svg>
  );
}
function Dashboard() {
  const { state, navigate, loadDemo } = useApp();
  const { exams, rooms, timeslots, conflicts, run, allocation } = state;
  const cfree = allocation.filter(a => a.status === 'Conflict Free').length;
  const flow = [
    ['Database', 'Input Data', 'Exams, rooms, slots'], ['BookOpen', 'Exams / Classes', 'Vertices of the graph'], ['Link2', 'Conflict Detection', 'Shared students / staff'], ['Network', 'Conflict Graph', 'Edges = conflicts'], ['ArrowDownWideNarrow', 'Greedy Vertex Ordering', 'Highest degree first'],
    ['Palette', 'Graph Coloring', 'Smallest feasible color'], ['ShieldCheck', 'Resource Validation', 'Capacity + slot checks'], ['DoorOpen', 'Room + Time Assignment', 'Color → slot, best-fit room'], ['CalendarDays', 'Final Timetable', 'Verified allocation'], ['BarChart3', 'Analytics', 'Utilization + metrics']];
  const pages = ['input', 'exams', 'graph', 'graph', 'run', 'visualizer', 'run', 'final', 'timetable', 'analytics'];
  return (
    <div className="space-y-8">
      <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-navy-700 via-navy-800 to-navy-900 p-6 shadow-2xl sm:p-10">
        <HeroNetwork />
        <div className="relative z-10 max-w-3xl">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-sky-400/30 bg-sky-500/10 px-3 py-1 text-xs font-semibold tracking-wide text-sky-200"><Icon name="GraduationCap" size={14} />MALLA REDDY VISHVAVIDHYAPEETH</div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-5xl">Smart Examination Scheduling</h1>
          <p className="mt-3 text-lg font-semibold text-sky-200">EXAM ROOM ALLOCATION SYSTEM · Graph Coloring + Greedy Algorithm</p>
          <p className="mt-2 max-w-xl text-slate-300">Optimize examination room allocation using Graph Coloring and Greedy Algorithms.</p>
          <p className="mt-1 text-sm text-slate-400">Developed by <span className="font-semibold text-white">THOKALA SHESHVITH</span></p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Btn icon="SlidersHorizontal" onClick={() => navigate('input')}>Configure Data</Btn>
            <Btn variant="success" icon="Play" onClick={() => navigate('run')}>Run Algorithm</Btn>
            <Btn variant="ghost" icon="ClipboardCheck" onClick={() => navigate('final')}>View Final Allocation</Btn>
          </div>
        </div>
      </section>
      <section className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
        <StatCard icon="BookOpen" label="Total Exams" value={exams.length} hint="Vertices (V)" />
        <StatCard icon="DoorOpen" label="Total Rooms" value={rooms.length} tone="violet" />
        <StatCard icon="Clock" label="Available Time Slots" value={timeslots.length} tone="amber" hint="Max colors available" />
        <StatCard icon="Link2" label="Total Conflicts" value={conflicts.length} tone="rose" hint="Edges (E)" />
        <StatCard icon="Palette" label="Colors / Slots Used" value={run ? `${run.summary.colors_used}/${timeslots.length}` : '—'} tone="sky" hint={run ? 'Chromatic result' : 'Run the algorithm'} />
        <StatCard icon="ShieldCheck" label="Conflict-Free Assignments" value={run ? `${cfree}/${exams.length}` : '—'} tone="emerald" hint={run ? `Unresolved conflicts: ${run.summary.unresolved_conflicts}` : 'Run the algorithm'} />
      </section>
      {exams.length === 0 && <Card className="flex flex-wrap items-center justify-between gap-4 p-5"><div><div className="font-bold text-white">Nothing configured yet</div><p className="text-sm text-slate-400">Load realistic demo data (8 exams, 4 rooms, 3 slots) or start from scratch.</p></div><Btn icon="Sparkles" onClick={loadDemo}>LOAD DEMO DATA</Btn></Card>}
      <section>
        <h2 className="mb-3 text-lg font-bold text-white">Algorithm Process Flow</h2>
        <Card className="p-5"><div className="flex flex-wrap items-center gap-2">
          {flow.map(([ic, t, d], i) => (
            <React.Fragment key={t}>
              <button onClick={() => navigate(pages[i])} className="group flex min-w-[150px] flex-1 items-center gap-3 rounded-xl border border-white/10 bg-white/5 p-3 text-left transition hover:-translate-y-0.5 hover:border-sky-400/40 hover:bg-sky-500/10">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-sky-500/15 text-sky-300"><Icon name={ic} size={17} /></span>
                <span><span className="block text-[11px] font-semibold text-sky-400">STEP {i + 1}</span><span className="block text-sm font-semibold text-white">{t}</span><span className="block text-xs text-slate-500">{d}</span></span>
              </button>
              {i < flow.length - 1 && <Icon name="ChevronRight" size={18} className="hidden text-slate-600 xl:block" />}
            </React.Fragment>))}
        </div></Card>
      </section>
      <section className="grid gap-6 lg:grid-cols-2">
        {run ? <ResultPanel run={run} /> : <EmptyState icon="Play" title="No allocation yet" text="Run the Greedy Graph Coloring algorithm to generate an allocation." action={<Btn icon="Play" onClick={() => navigate('run')}>Run Algorithm</Btn>} />}
        <WhyGreedy />
      </section>
    </div>
  );
}

/* ============================== INPUT CONFIGURATION ============================== */
function InputConfig() {
  const { state, navigate, loadDemo, resetProject } = useApp();
  const checks = readiness(state);
  const lv = { ok: ['CheckCircle2', 'text-emerald-300'], warn: ['AlertTriangle', 'text-amber-300'], error: ['XCircle', 'text-rose-300'] };
  const cards = [['exams', 'BookOpen', 'Exams', state.exams.length, 'Vertices of the conflict graph'], ['rooms', 'DoorOpen', 'Rooms', state.rooms.length, 'Capacity-limited resources'], ['slots', 'Clock', 'Time Slots', state.timeslots.length, 'Available colors']];
  return (
    <div>
      <PageHeader icon="SlidersHorizontal" title="Input Configuration" subtitle="Define exams, rooms, time slots and conflicts. Every change is validated and stored in SQLite."
        actions={<><Btn icon="Sparkles" onClick={loadDemo}>LOAD DEMO DATA</Btn><Btn variant="ghost" icon="RotateCcw" onClick={resetProject}>RESET PROJECT</Btn></>} />
      <div className="grid gap-4 sm:grid-cols-3">{cards.map(([p, ic, t, n, d]) => (
        <button key={p} onClick={() => navigate(p)} className="text-left"><Card className="p-5 transition hover:-translate-y-0.5 hover:border-sky-400/40">
          <div className="flex items-center justify-between"><span className="rounded-xl bg-sky-500/15 p-2.5 text-sky-300"><Icon name={ic} size={20} /></span><Icon name="ArrowRight" size={16} className="text-slate-500" /></div>
          <div className="mt-3 text-3xl font-extrabold text-white">{n}</div><div className="font-semibold text-slate-200">{t}</div><div className="text-xs text-slate-500">{d}</div></Card></button>))}</div>
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card className="p-5"><h3 className="mb-3 font-bold text-white">Data readiness</h3>
          <ul className="space-y-2">{checks.map((c, i) => <li key={i} className="flex items-start gap-2 text-sm"><Icon name={lv[c.level][0]} size={16} className={`mt-0.5 shrink-0 ${lv[c.level][1]}`} /><span className="text-slate-300">{c.text}</span></li>)}</ul>
          <Btn className="mt-5" icon="Play" onClick={() => navigate('run')}>Continue to Run Algorithm</Btn></Card>
        <ConflictManager />
      </div>
    </div>
  );
}

/* ============================== CONFLICT GRAPH ============================== */
function GraphPage() {
  const { state, navigate, setGraphPos } = useApp();
  const [sel, setSel] = useState(null);
  const colors = useMemo(() => { const m = {}; state.allocation.forEach(a => { if (a.color) m[a.exam_id] = a.color; }); return m; }, [state.allocation]);
  const maxE = state.exams.length * (state.exams.length - 1) / 2;
  if (!state.exams.length) return <div><PageHeader icon="Network" title="CONFLICT GRAPH" subtitle="Exams are vertices, conflicts are edges." /><EmptyState icon="Network" title="No exams added yet." text={'No exams added yet.\nAdd your first exam to begin building the conflict graph.'} action={<Btn icon="Plus" onClick={() => navigate('exams')}>Add Exam</Btn>} /></div>;
  return (
    <div>
      <PageHeader icon="Network" title="CONFLICT GRAPH" subtitle="Each exam is a vertex; each conflict is an edge. After running the algorithm nodes are filled with their assigned color."
        actions={<Btn variant="ghost" icon="Shuffle" onClick={() => setGraphPos({})}>Reset layout</Btn>} />
      <div className="mb-4 grid grid-cols-3 gap-3">
        <StatCard icon="Circle" label="Vertices" value={state.exams.length} /><StatCard icon="Link2" label="Edges" value={state.conflicts.length} tone="rose" /><StatCard icon="Percent" label="Density" value={maxE ? Math.round(state.conflicts.length / maxE * 100) + '%' : '0%'} tone="amber" />
      </div>
      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="p-3 sm:p-4 xl:col-span-2">
          {!state.conflicts.length && <p className="mb-3 whitespace-pre-line rounded-xl bg-amber-500/10 p-3 text-sm text-amber-200">No conflicts defined.{'\n'}Add conflicts to build the graph.</p>}
          <GraphCanvas exams={state.exams} conflicts={state.conflicts} colors={colors} selectedId={sel} onSelect={setSel} />
        </Card>
        <div className="space-y-6"><NodeDetails examId={sel} /><ConflictManager /></div>
      </div>
    </div>
  );
}

/* ============================== RUN ALGORITHM ============================== */
function RunPage() {
  const { state, settings, setSettings, refresh, toast, navigate } = useApp();
  const [running, setRunning] = useState(false);
  const checks = readiness(state); const blocked = checks.some(c => c.level === 'error'); const run = state.run;
  const lv = { ok: ['CheckCircle2', 'text-emerald-300'], warn: ['AlertTriangle', 'text-amber-300'], error: ['XCircle', 'text-rose-300'] };
  const go = async () => {
    setRunning(true);
    try {
      await new Promise(r => setTimeout(r, 450));
      const res = await api('/run-algorithm', { method: 'POST', body: { ordering: settings.ordering } });
      await refresh();
      toast(res.run.success ? 'success' : 'error', res.run.success ? `Allocation generated: ${res.run.summary.status}` : 'Allocation Could Not Be Completed. See the explanation below.');
    } catch (e) { toast('error', e.message); } finally { setRunning(false); }
  };
  return (
    <div className="space-y-6">
      <PageHeader icon="Play" title="Run Algorithm" subtitle="Greedy Graph Coloring with resource validation (room capacity + time slot)."
        actions={<><Btn icon="Play" variant="success" onClick={go} loading={running} disabled={blocked}>RUN ALGORITHM</Btn>{run && <Btn variant="ghost" icon="Eye" onClick={() => navigate('visualizer')}>Open Visualizer</Btn>}</>} />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5">
          <h3 className="mb-3 font-bold text-white">How the greedy algorithm works</h3>
          <ol className="list-decimal space-y-1.5 pl-5 text-sm text-slate-300">
            <li><b>Order</b> the vertices (exams): highest degree (most conflicts) first.</li>
            <li>For each exam, collect the colors used by its <b>conflicting neighbours</b>.</li>
            <li>Try colors 1, 2, 3… and pick the <b>smallest</b> one that is not used by a neighbour and still has a free room with enough seats.</li>
            <li>Color k = k-th time slot; the smallest room that fits is booked (best-fit).</li>
            <li><b>Verify</b>: for every edge (A, B) the colors must differ.</li>
          </ol>
          <div className="mt-4"><Field label="Vertex ordering rule">
            <select className={inputCls} value={settings.ordering} onChange={e => setSettings({ ordering: e.target.value })}>
              <option value="degree_desc">Descending degree (default, Welsh–Powell)</option><option value="degree_asc">Ascending degree (for comparison)</option><option value="input">Input order (for comparison)</option></select></Field>
            <p className="mt-2 text-xs text-slate-500">Default rule: exams with the highest number of conflicts are processed first; ties → priority (High first) → student count → input order.</p></div>
        </Card>
        <Card className="p-5"><h3 className="mb-3 font-bold text-white">Pre-run checks</h3>
          <ul className="space-y-2">{checks.map((c, i) => <li key={i} className="flex items-start gap-2 text-sm"><Icon name={lv[c.level][0]} size={16} className={`mt-0.5 shrink-0 ${lv[c.level][1]}`} /><span className="text-slate-300">{c.text}</span></li>)}</ul>
          {blocked && <p className="mt-4 rounded-xl bg-rose-500/10 p-3 text-sm text-rose-200">Invalid or empty data cannot be processed. <button className="font-semibold underline" onClick={() => navigate('input')}>Go to Input Configuration</button></p>}</Card>
      </div>
      {!run ? <NoRun navigate={navigate} /> : (<>
        <FailureBanner run={run} />
        <Card className="p-5"><h3 className="mb-3 font-bold text-white">Greedy vertex ordering <span className="text-sm font-normal text-slate-400">({run.ordering_label})</span></h3>
          <div className="flex flex-wrap gap-2">{run.order.map((o, i) => { const st = run.steps[i]; return (
            <div key={o.id} className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm"><span className="mr-2 font-mono text-xs text-slate-500">#{i + 1}</span><span className="font-semibold text-white">{o.name}</span><span className="ml-2 text-xs text-sky-300">deg {o.degree}</span>{st && st.selected_color && <span className="ml-2"><ColorChip c={st.selected_color} short /></span>}</div>); })}</div></Card>
        <ResultPanel run={run} />
        <div><h3 className="mb-3 text-lg font-bold text-white">Allocation Log</h3><AllocationLog steps={run.steps} /></div>
      </>)}
    </div>
  );
}

/* ============================== ALGORITHM VISUALIZER ============================== */
function Visualizer() {
  const { state, settings, navigate } = useApp();
  const run = state.run; const steps = run ? run.steps : [];
  const [idx, setIdx] = useState(0); const [playing, setPlaying] = useState(false);
  useEffect(() => { setIdx(0); setPlaying(false); }, [run && run.id]);
  useEffect(() => {
    if (!playing) return;
    if (idx >= steps.length) { setPlaying(false); return; }
    const t = setTimeout(() => setIdx(i => i + 1), settings.speed); return () => clearTimeout(t);
  }, [playing, idx, steps.length, settings.speed]);
  if (!run) return <div><PageHeader icon="Eye" title="GREEDY ALGORITHM VISUALIZER" subtitle="Watch the algorithm color the conflict graph step by step." /><NoRun navigate={navigate} /></div>;
  const applied = steps.slice(0, idx); const cur = idx > 0 ? steps[idx - 1] : null;
  const colors = {}; applied.forEach(s => { if (s.selected_color) colors[s.exam_id] = s.selected_color; });
  const done = idx >= steps.length;
  const start = () => { setPlaying(false); setIdx(i => (i === 0 ? 1 : i)); };
  const auto = () => { if (idx >= steps.length) setIdx(0); setPlaying(true); };
  return (
    <div className="space-y-6">
      <PageHeader icon="Eye" title="GREEDY ALGORITHM VISUALIZER" subtitle={`Ordering: ${run.ordering_label}. Highlighted node = vertex being processed; dashed amber rings = its conflicting neighbours.`} />
      <Card className="flex flex-wrap items-center gap-2 p-3">
        <Btn icon="Play" variant="ghost" onClick={start} disabled={idx > 0}>Start</Btn>
        <Btn icon="Pause" variant="ghost" onClick={() => setPlaying(false)} disabled={!playing}>Pause</Btn>
        <Btn icon="SkipBack" variant="ghost" onClick={() => { setPlaying(false); setIdx(i => Math.max(0, i - 1)); }} disabled={idx === 0}>Previous Step</Btn>
        <Btn icon="SkipForward" variant="ghost" onClick={() => { setPlaying(false); setIdx(i => Math.min(steps.length, i + 1)); }} disabled={done}>Next Step</Btn>
        <Btn icon="RotateCcw" variant="ghost" onClick={() => { setPlaying(false); setIdx(0); }} disabled={idx === 0}>Reset</Btn>
        <Btn icon="FastForward" onClick={auto} disabled={playing}>Run Automatically</Btn>
        <span className="ml-auto rounded-full bg-white/10 px-3 py-1 text-xs font-semibold text-slate-300">Step {idx} / {steps.length}</span>
      </Card>
      <div className="grid gap-6 xl:grid-cols-5">
        <Card className="p-3 sm:p-4 xl:col-span-3">
          <GraphCanvas exams={state.exams} conflicts={state.conflicts} colors={colors} activeId={cur ? cur.exam_id : null} highlightIds={cur ? cur.neighbours.map(n => n.id) : []} />
        </Card>
        <div className="space-y-4 xl:col-span-2">
          {!cur ? (
            <Card className="p-5"><h3 className="font-bold text-white">Ready</h3><p className="mt-2 text-sm text-slate-400">Press <b>Start</b> to go step by step, or <b>Run Automatically</b> to play the whole algorithm.</p>
              <p className="mt-3 text-sm text-slate-400">Vertices are processed in this order:</p>
              <div className="mt-2 flex flex-wrap gap-1.5">{run.order.map((o, i) => <span key={o.id} className="rounded-lg bg-white/10 px-2 py-1 text-xs">{i + 1}. {o.name} <span className="text-sky-300">(deg {o.degree})</span></span>)}</div></Card>
          ) : (
            <Card className="p-5" key={cur.step}>
              <div className="pop mb-3 flex items-center justify-between"><span className="text-xs font-bold tracking-widest text-sky-300">STEP {cur.step}</span>{cur.success ? <ColorChip c={cur.selected_color} /> : <span className="text-sm font-semibold text-rose-300">No feasible color</span>}</div>
              <dl className="space-y-3 text-sm">
                <div><dt className="text-xs uppercase tracking-wide text-slate-500">Current exam</dt><dd className="font-semibold text-white">{cur.exam} <span className="font-normal text-slate-400">· {cur.subject} · degree {cur.degree}</span></dd></div>
                <div><dt className="mb-1 text-xs uppercase tracking-wide text-slate-500">Conflicting exams</dt><dd className="flex flex-wrap gap-1.5">{cur.neighbours.length ? cur.neighbours.map(n => <span key={n.id} className="inline-flex items-center gap-1.5 rounded-md bg-white/10 px-2 py-0.5 text-xs">{n.name}{n.color && <span className="h-2 w-2 rounded-full" style={{ background: colorOf(n.color) }} />}</span>) : <span className="text-slate-500">None</span>}</dd></div>
                <div><dt className="mb-1 text-xs uppercase tracking-wide text-slate-500">Colors used by neighbours</dt><dd className="flex flex-wrap gap-1.5">{cur.neighbour_colors.length ? cur.neighbour_colors.map(c => <ColorChip key={c} c={c} />) : <span className="text-slate-400">None</span>}</dd></div>
                <div><dt className="text-xs uppercase tracking-wide text-slate-500">Selected color</dt><dd className="font-semibold text-white">{cur.selected_color ? `Color ${cur.selected_color}` : 'None'}{cur.room && <span className="ml-2 font-normal text-slate-400">→ {cur.room} · {slotRange(cur.slot_start, cur.slot_end, cur.slot_day)}</span>}</dd></div>
                <div><dt className="text-xs uppercase tracking-wide text-slate-500">Reason</dt><dd className="mt-1 rounded-xl bg-white/5 p-3 italic text-slate-200">"{cur.reason}"</dd></div>
              </dl>
            </Card>)}
          {done && steps.length > 0 && <Card className={`p-4 text-sm font-semibold ${run.success ? 'text-emerald-300' : 'text-rose-300'}`}>{run.success ? `✓ All ${steps.length} exams colored using ${run.summary.colors_used} color(s). ${run.summary.status}.` : 'Allocation Could Not Be Completed. Check the failed step.'}</Card>}
          <Card className="p-4"><div className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-400">Pseudocode</div>
            <pre className="overflow-x-auto font-mono text-xs leading-relaxed text-slate-300">{`order = sort(vertices, by degree desc)
for v in order:
    used = { color[u] : u in neighbours(v) }
    c = 1
    while c in used or no room free in slot c:
        c = c + 1
    color[v] = c        # smallest feasible`}</pre></Card>
        </div>
      </div>
      <Card className="p-4"><div className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-400">Vertex ordering progress</div>
        <div className="flex flex-wrap gap-2">{run.order.map((o, i) => { const st = steps[i]; const doneV = i < idx; const act = cur && cur.exam_id === o.id; return (
          <div key={o.id} className={`rounded-xl border px-3 py-2 text-sm transition ${act ? 'border-sky-400 bg-sky-500/15' : 'border-white/10 bg-white/5'}`} style={doneV && st && st.selected_color ? { borderColor: colorOf(st.selected_color) + '88' } : {}}>
            <span className="mr-1.5 font-mono text-xs text-slate-500">#{i + 1}</span>{o.name}<span className="ml-1.5 text-xs text-sky-300">d={o.degree}</span>{doneV && st && st.selected_color && <span className="ml-2"><ColorChip c={st.selected_color} short /></span>}</div>); })}</div></Card>
      <div><h3 className="mb-3 text-lg font-bold text-white">Allocation Log (up to current step)</h3><AllocationLog steps={applied} /></div>
    </div>
  );
}

/* ============================== FINAL ALLOCATION ============================== */
function ExportBar() {
  const { state, toast } = useApp();
  return (<>
    <Btn icon="FileText" onClick={() => exportPDF(state, toast)}>Export Timetable as PDF</Btn>
    <Btn variant="ghost" icon="Download" onClick={() => { exportCSV(state.allocation); toast('success', 'CSV downloaded.'); }}>Export Allocation as CSV</Btn>
    <Btn variant="ghost" icon="Printer" onClick={() => window.print()}>Print Timetable</Btn>
  </>);
}
function FinalPage() {
  const { state, navigate } = useApp(); const { run, allocation } = state;
  if (!run) return <div><PageHeader icon="ClipboardCheck" title="FINAL EXAM ALLOCATION" /><NoRun navigate={navigate} /></div>;
  const s = run.summary, ok = s.status === 'CONFLICT FREE';
  return (
    <div className="space-y-6">
      <PageHeader icon="ClipboardCheck" title="FINAL EXAM ALLOCATION" subtitle="Verified result of the Greedy Graph Coloring allocation." actions={<ExportBar />} />
      <FailureBanner run={run} />
      <Card className={`flex flex-wrap items-center justify-between gap-4 p-5 ${ok ? 'border-emerald-400/30' : 'border-amber-400/30'}`}>
        <div className="flex items-center gap-3"><div className={`rounded-2xl p-3 ${ok ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-300'}`}><Icon name={ok ? 'ShieldCheck' : 'AlertTriangle'} size={28} /></div>
          <div><div className="text-2xl font-extrabold text-white">{s.unresolved_conflicts} Scheduling Conflicts</div><div className={`text-sm font-semibold ${ok ? 'text-emerald-300' : 'text-amber-300'}`}>Allocation Status: {s.status}</div></div></div>
        <div className="text-sm text-slate-400">{run.verification.passed ? '✓ All conflict constraints satisfied' : `✕ ${run.verification.conflict_count} conflicts detected`}</div>
      </Card>
      <Card className="hidden overflow-x-auto md:block">
        <table className="w-full text-left text-sm"><thead><tr className="border-b border-white/10 text-xs uppercase tracking-wide text-slate-400">{['Exam', 'Subject', 'Students', 'Room', 'Building', 'Time Slot', 'Color', 'Status'].map(h => <th key={h} className="px-4 py-3 font-medium">{h}</th>)}</tr></thead>
          <tbody>{allocation.map(a => <tr key={a.exam_id} className="border-b border-white/5 hover:bg-white/[.03]">
            <td className="px-4 py-3 font-semibold text-white">{a.exam}</td><td className="px-4 py-3">{a.subject}</td><td className="px-4 py-3">{a.students}</td><td className="px-4 py-3">{a.room || '—'}</td><td className="px-4 py-3">{a.building || '—'}</td>
            <td className="px-4 py-3 whitespace-nowrap">{rowSlot(a)}</td><td className="px-4 py-3"><ColorChip c={a.color} /></td><td className="px-4 py-3"><StatusBadge status={a.status} /></td></tr>)}</tbody></table>
      </Card>
      <div className="grid gap-3 sm:grid-cols-2 md:hidden">{allocation.map(a => (
        <Card key={a.exam_id} className="p-4" style={{ borderLeft: `4px solid ${colorOf(a.color)}` }}>
          <div className="flex items-start justify-between gap-2"><div><div className="font-bold text-white">{a.exam}</div><div className="text-xs text-slate-400">{a.subject} · {a.students} students</div></div><ColorChip c={a.color} /></div>
          <div className="mt-3 space-y-1 text-sm text-slate-300"><div>{a.room ? `${a.room} · ${a.building}` : 'No room'}</div><div>{rowSlot(a)}</div></div><div className="mt-3"><StatusBadge status={a.status} /></div></Card>))}</div>
    </div>
  );
}

/* ============================== TIMETABLE ============================== */
function TimetablePage() {
  const { state, navigate } = useApp(); const { run, allocation, rooms, timeslots } = state;
  if (!run) return <div><PageHeader icon="CalendarDays" title="Timetable" /><NoRun navigate={navigate} /></div>;
  const used = [...new Set(allocation.filter(a => a.color).map(a => a.color))].sort((a, b) => a - b);
  return (
    <div className="space-y-6">
      <PageHeader icon="CalendarDays" title="Timetable" subtitle="Rows are time slots (colors), columns are rooms. Cell color = graph color." actions={<ExportBar />} />
      <FailureBanner run={run} />
      <Card className="overflow-x-auto p-2 sm:p-4">
        <table className="w-full min-w-[640px] border-separate border-spacing-2 text-sm">
          <thead><tr><th className="p-2 text-left text-xs uppercase tracking-wide text-slate-400">Time</th>{rooms.map(r => <th key={r.id} className="p-2 text-left"><div className="font-semibold text-white">{r.name}</div><div className="text-xs font-normal text-slate-500">{r.capacity} seats · {r.building}</div></th>)}</tr></thead>
          <tbody>{timeslots.map((s, i) => (
            <tr key={s.id}><td className="whitespace-nowrap rounded-xl bg-white/5 p-3 align-top"><div className="font-semibold text-white">{s.label}</div><div className="text-xs text-slate-400">{fmt12(s.start)} – {fmt12(s.end)}</div>{s.day && s.day.toLowerCase() !== 'day 1' && <div className="text-xs text-amber-300">{s.day}</div>}<div className="mt-1"><ColorChip c={i + 1} short /></div></td>
              {rooms.map(r => { const a = allocation.find(x => x.slot_id === s.id && x.room_id === r.id);
                return <td key={r.id} className="rounded-xl align-top" style={a ? { background: colorOf(a.color) + '22', border: `1px solid ${colorOf(a.color)}77` } : { border: '1px dashed rgba(255,255,255,.1)' }}>
                  {a ? <div className="p-3"><div className="font-bold text-white">{a.subject}</div><div className="text-xs text-slate-300">{a.exam} · {a.students}/{a.capacity}</div></div> : <div className="p-3 text-slate-600">—</div>}</td>; })}</tr>))}</tbody></table>
      </Card>
      <div className="flex flex-wrap gap-2">{used.map(c => <span key={c} className="flex items-center gap-1.5 text-xs text-slate-400"><ColorChip c={c} /> = {(timeslots[c - 1] || {}).label}</span>)}</div>
    </div>
  );
}

/* ============================== ANALYTICS ============================== */
function Analytics() {
  const { state, navigate } = useApp(); const [d, setD] = useState(null);
  useEffect(() => { api('/analytics').then(setD).catch(() => setD(null)); }, [state.run && state.run.id, state.exams.length, state.conflicts.length, state.rooms.length, state.timeslots.length]);
  if (!d) return <div><PageHeader icon="BarChart3" title="Analytics" /><div className="grid gap-4 sm:grid-cols-3"><Skeleton className="h-28" /><Skeleton className="h-28" /><Skeleton className="h-28" /></div></div>;
  if (!state.exams.length) return <div><PageHeader icon="BarChart3" title="Analytics" /><EmptyState icon="BarChart3" title="No data to analyse" text={'No exams added yet.\nAdd your first exam to begin building the conflict graph.'} action={<Btn onClick={() => navigate('exams')}>Add Exam</Btn>} /></div>;
  const stats = [
    ['Total Exams', d.total_exams], ['Total Conflicts', d.total_conflicts], ['Colors Used', d.has_run ? d.colors_used : '—'], ['Average Room Utilization', d.has_run ? d.avg_room_utilization + '%' : '—'],
    ['Average Seat Fill', d.has_run ? d.avg_seat_fill + '%' : '—'], ['Students Scheduled', d.has_run ? `${d.students_scheduled}/${d.students_total}` : '—'],
    ['Conflicts Before Optimization', d.conflicts_before ?? '—'], ['Conflicts After Allocation', d.has_run ? d.conflicts_after : '—']];
  return (
    <div className="space-y-6">
      <PageHeader icon="BarChart3" title="Analytics" subtitle="Graph statistics and resource utilization computed from the stored allocation." />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">{stats.map(([k, v]) => <Card key={k} className="p-4"><div className="text-[11px] uppercase tracking-wide text-slate-400">{k}</div><div className="mt-1 text-2xl font-extrabold text-white">{v}</div></Card>)}</div>
      <p className="text-xs text-slate-500">“Before optimization” = conflicts produced by a naive round-robin assignment (exam i → slot i mod S) that ignores the conflict graph. “After” = verified conflicts of the greedy result.</p>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5"><h3 className="mb-4 font-bold text-white">Graph Statistics · nodes vs edges</h3><BarChart data={[{ label: 'Nodes (V)', value: d.nodes, color: '#38bdf8' }, { label: 'Edges (E)', value: d.edges, color: '#fb7185' }, { label: 'Max degree', value: d.max_degree, color: '#fbbf24' }]} /></Card>
        {d.has_run ? <>
          <Card className="p-5"><h3 className="mb-4 font-bold text-white">Room Utilization · % of slots used</h3><BarChart max={100} unit="%" data={d.room_utilization.map((r, i) => ({ label: r.room, value: r.usage_pct, color: PALETTE[i % PALETTE.length] }))} /></Card>
          <Card className="p-5"><h3 className="mb-4 font-bold text-white">Time Slot Utilization · exams per slot</h3><BarChart data={d.slot_utilization.map((s, i) => ({ label: s.slot, value: s.exams, color: PALETTE[i % PALETTE.length] }))} /></Card>
          <Card className="p-5"><h3 className="mb-4 font-bold text-white">Color Distribution · exams per color</h3><BarChart data={d.color_distribution.map(c => ({ label: 'Color ' + c.color, value: c.exams, color: colorOf(c.color) }))} /></Card>
        </> : <Card className="lg:col-span-1"><NoRun navigate={navigate} /></Card>}
      </div>
      {d.has_run && <Card className="overflow-x-auto"><table className="w-full min-w-[480px] text-left text-sm"><thead><tr className="border-b border-white/10 text-xs uppercase tracking-wide text-slate-400"><th className="px-4 py-3">Room</th><th className="px-4 py-3">Capacity</th><th className="px-4 py-3">Exams hosted</th><th className="px-4 py-3">Slot usage</th><th className="px-4 py-3">Avg seat fill</th></tr></thead>
        <tbody>{d.room_utilization.map(r => <tr key={r.room} className="border-b border-white/5"><td className="px-4 py-3 font-semibold text-white">{r.room}</td><td className="px-4 py-3">{r.capacity}</td><td className="px-4 py-3">{r.exams}</td><td className="px-4 py-3">{r.usage_pct}%</td><td className="px-4 py-3">{r.seat_fill_pct}%</td></tr>)}</tbody></table></Card>}
    </div>
  );
}

/* ============================== ALGORITHM ANALYSIS ============================== */
function AnalysisPage() {
  const { state } = useApp(); const [cmp, setCmp] = useState(null);
  useEffect(() => { api('/ordering-comparison').then(setCmp).catch(() => setCmp(null)); }, [state.exams.length, state.conflicts.length]);
  const V = state.exams.length, E = state.conflicts.length, C = state.run ? state.run.summary.colors_used : '—';
  const Pro = ({ t, items, tone }) => <div><div className={`mb-2 text-xs font-bold uppercase tracking-wide ${tone}`}>{t}</div><ul className="space-y-1.5 text-sm text-slate-300">{items.map(i => <li key={i}>• {i}</li>)}</ul></div>;
  return (
    <div className="space-y-6">
      <PageHeader icon="Sigma" title="ALGORITHM ANALYSIS" subtitle="Graph Coloring and the Greedy strategy: purpose, complexity and limitations." />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard icon="Circle" label="Vertices (V)" value={V} /><StatCard icon="Link2" label="Edges (E)" value={E} tone="rose" /><StatCard icon="Palette" label="Colors Used (C)" value={C} tone="violet" />
        <Card className="p-4"><div className="text-xs font-medium uppercase tracking-wide text-slate-400">Estimated Complexity</div><div className="mt-2 font-mono text-2xl font-extrabold text-emerald-300">O(V²)</div><div className="mt-1 text-xs text-slate-500">straightforward implementation</div></Card>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5"><h3 className="text-lg font-bold text-white">Graph Coloring</h3><p className="mt-1 text-xs font-semibold uppercase tracking-wide text-sky-300">Purpose</p>
          <p className="mt-1 text-sm text-slate-300">Represent examination conflicts and assign different colors to connected vertices. Each exam is a vertex, each conflict an edge, and each color a time slot. Adjacent vertices must never share a color.</p></Card>
        <Card className="p-5"><h3 className="text-lg font-bold text-white">Greedy Strategy</h3><p className="mt-1 text-xs font-semibold uppercase tracking-wide text-sky-300">Purpose</p>
          <p className="mt-1 text-sm text-slate-300">Select the smallest feasible color at each step. The result depends on the vertex ordering, and the greedy method does not necessarily guarantee the minimum possible number of colors.</p></Card>
      </div>
      <Card className="overflow-x-auto"><table className="w-full min-w-[520px] text-left text-sm"><thead><tr className="border-b border-white/10 text-xs uppercase tracking-wide text-slate-400"><th className="px-4 py-3">Measure</th><th className="px-4 py-3">Value</th><th className="px-4 py-3">Explanation</th></tr></thead><tbody>
        {[['Time complexity', 'O(V²)  (O(V + E) with adjacency lists)', 'Each vertex inspects its neighbours to find used colors; with an adjacency matrix that is O(V) per vertex → O(V²). Sorting by degree adds O(V log V). Room selection adds O(V·S·R) for S slots and R rooms.'],
          ['Space complexity', 'O(V + E)  (O(V²) for a matrix)', 'Adjacency list, color array, room-booking table and the step log.'],
          ['Colors guaranteed', '≤ Δ + 1', 'Greedy never needs more than (maximum degree + 1) colors.']].map(r => <tr key={r[0]} className="border-b border-white/5 align-top"><td className="px-4 py-3 font-semibold text-white">{r[0]}</td><td className="px-4 py-3 font-mono text-emerald-300">{r[1]}</td><td className="px-4 py-3 text-slate-300">{r[2]}</td></tr>)}</tbody></table></Card>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5"><Pro t="Advantages" tone="text-emerald-300" items={['Very fast and simple to implement', 'Produces a valid conflict-free coloring', 'Easy to trace step by step (transparent decisions)', 'Works for any graph size, deterministic output']} /></Card>
        <Card className="p-5"><Pro t="Limitations" tone="text-amber-300" items={['Not guaranteed to use the minimum number of colors', 'Result depends on the vertex ordering', 'Local decisions cannot be undone (no backtracking)', 'Finding the true chromatic number is NP-hard']} /></Card>
      </div>
      <Card className="p-5"><h3 className="mb-1 text-lg font-bold text-white">Does ordering matter? (live experiment)</h3>
        <p className="mb-4 text-sm text-slate-400">Plain greedy coloring (without room limits) run on your current graph with different vertex orders.</p>
        {!cmp || !V ? <p className="text-sm text-slate-500">Add exams and conflicts to see the comparison.</p> : (<>
          <div className="overflow-x-auto"><table className="w-full min-w-[480px] text-left text-sm"><thead><tr className="border-b border-white/10 text-xs uppercase tracking-wide text-slate-400"><th className="py-2">Ordering</th><th>Colors used</th><th>Order of processing</th></tr></thead>
            <tbody>{cmp.strategies.map(s => <tr key={s.key} className="border-b border-white/5 align-top"><td className="py-2 pr-3 font-medium text-white">{s.label}</td><td className="pr-3 font-mono text-sky-300">{s.colors_used}</td><td className="py-2 text-xs text-slate-400">{s.order.join(' → ')}</td></tr>)}</tbody></table></div>
          <div className="mt-4 flex flex-wrap gap-3 text-sm"><span className="rounded-lg bg-white/5 px-3 py-1.5">Max degree Δ = <b>{cmp.max_degree}</b></span><span className="rounded-lg bg-white/5 px-3 py-1.5">Upper bound Δ + 1 = <b>{cmp.upper_bound}</b></span>
            <span className="rounded-lg bg-white/5 px-3 py-1.5">Exact minimum (back-tracking) = <b>{cmp.optimal ?? 'n/a (graph too large)'}</b></span></div></>)}
      </Card>
      <WhyGreedy />
    </div>
  );
}

/* ============================== ABOUT / SETTINGS ============================== */
function AboutPage() {
  const rows = [['Project Title', 'Exam Room Allocation System'], ['Algorithms', 'Graph Coloring + Greedy'], ['Technology', 'Python / Flask / SQLite / React'], ['Institution', 'MALLA REDDY VISHVAVIDHYAPEETH'], ['Student', 'THOKALA SHESHVITH']];
  return (
    <div className="space-y-6">
      <PageHeader icon="Info" title="About Project" subtitle="Design and Analysis of Algorithms (DAA) project." />
      <Card className="p-6"><dl className="divide-y divide-white/5">{rows.map(([k, v]) => <div key={k} className="flex flex-col gap-1 py-3 sm:flex-row sm:gap-8"><dt className="w-40 shrink-0 text-xs font-semibold uppercase tracking-wide text-slate-400">{k}</dt><dd className="font-medium text-white">{v}</dd></div>)}
        <div className="flex flex-col gap-1 py-3 sm:flex-row sm:gap-8"><dt className="w-40 shrink-0 text-xs font-semibold uppercase tracking-wide text-slate-400">Purpose</dt><dd className="text-slate-200">To demonstrate how Graph Coloring and Greedy algorithms can be applied to a real-world examination scheduling and room allocation problem.</dd></div></dl></Card>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5"><h3 className="mb-3 font-bold text-white">Problem model</h3><ul className="space-y-1.5 text-sm text-slate-300"><li>• Exam / class → <b>vertex</b></li><li>• Conflict between two exams → <b>edge</b></li><li>• Time slot → <b>color</b> (k-th slot = color k)</li><li>• Conflicting exams must never receive the same color</li><li>• Inside a slot, each exam gets a different room with enough seats</li></ul></Card>
        <Card className="p-5"><h3 className="mb-3 font-bold text-white">Architecture</h3><ul className="space-y-1.5 text-sm text-slate-300"><li>• <b>Flask</b> REST API (/api/exams, /api/rooms, /api/timeslots, /api/conflicts, /api/run-algorithm, /api/allocation, /api/analytics, /api/reset)</li><li>• <b>SQLite</b> tables: exams, rooms, timeslots, conflicts, allocations, algorithm_runs</li><li>• Algorithm module: <code className="rounded bg-white/10 px-1">greedy_graph_coloring()</code> and <code className="rounded bg-white/10 px-1">verify_allocation()</code></li><li>• <b>React</b> + Tailwind frontend with SVG graph visualization</li></ul></Card>
      </div>
    </div>
  );
}
function SettingsPage() {
  const { settings, setSettings, resetProject, loadDemo } = useApp();
  return (
    <div className="space-y-6">
      <PageHeader icon="Settings" title="Settings" subtitle="Preferences are stored in this browser." />
      <Card className="space-y-5 p-5">
        <Field label="Default vertex ordering"><select className={inputCls} value={settings.ordering} onChange={e => setSettings({ ordering: e.target.value })}><option value="degree_desc">Descending degree (recommended)</option><option value="degree_asc">Ascending degree</option><option value="input">Input order</option></select></Field>
        <Field label={`Visualizer speed: ${(settings.speed / 1000).toFixed(1)} s per step`}><input type="range" min="300" max="3000" step="100" value={settings.speed} onChange={e => setSettings({ speed: +e.target.value })} className="w-full accent-sky-500" /></Field>
      </Card>
      <Card className="flex flex-wrap gap-3 p-5"><Btn icon="Sparkles" onClick={loadDemo}>LOAD DEMO DATA</Btn><Btn variant="danger" icon="RotateCcw" onClick={resetProject}>RESET PROJECT</Btn></Card>
    </div>
  );
}

/* ============================== PRINT SHEET (only visible when printing) ============================== */
function PrintSheet({ state }) {
  const { run, allocation, rooms, timeslots } = state; if (!run) return null;
  return (
    <div className="print-only" style={{ color: '#000', background: '#fff', padding: 16, fontFamily: 'Inter, Arial, sans-serif' }}>
      <h1 style={{ textAlign: 'center', fontSize: 20, margin: 0 }}>MALLA REDDY VISHVAVIDHYAPEETH</h1>
      <h2 style={{ textAlign: 'center', fontSize: 16, margin: '4px 0' }}>EXAM ROOM ALLOCATION SYSTEM</h2>
      <p style={{ textAlign: 'center', fontSize: 12, margin: '2px 0 12px' }}>Graph Coloring + Greedy Algorithm · Student: THOKALA SHESHVITH · Status: {run.summary.status}</p>
      <h3 style={{ fontSize: 14 }}>Final Allocation</h3>
      <table><thead><tr>{['Exam', 'Subject', 'Students', 'Room', 'Building', 'Time Slot', 'Color', 'Status'].map(h => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>{allocation.map(a => <tr key={a.exam_id}><td>{a.exam}</td><td>{a.subject}</td><td>{a.students}</td><td>{a.room || '-'}</td><td>{a.building || '-'}</td><td>{pdfSlot(a)}</td><td>{a.color ? 'Color ' + a.color : '-'}</td><td>{a.status}</td></tr>)}</tbody></table>
      <h3 style={{ fontSize: 14, marginTop: 18 }}>Timetable</h3>
      <table><thead><tr><th>Time</th>{rooms.map(r => <th key={r.id}>{r.name}</th>)}</tr></thead>
        <tbody>{timeslots.map(s => <tr key={s.id}><td>{s.label}: {fmt12(s.start)} - {fmt12(s.end)}{dayTag(s.day)}</td>{rooms.map(r => { const a = allocation.find(x => x.slot_id === s.id && x.room_id === r.id); return <td key={r.id}>{a ? `${a.subject} (${a.exam})` : '-'}</td>; })}</tr>)}</tbody></table>
    </div>
  );
}

/* ============================== APP SHELL ============================== */
const NAV = [
  ['dashboard', 'Dashboard', 'LayoutDashboard'], ['input', 'Input Configuration', 'SlidersHorizontal'], ['exams', 'Exams', 'BookOpen'], ['rooms', 'Rooms', 'DoorOpen'], ['slots', 'Time Slots', 'Clock'],
  ['graph', 'Conflict Graph', 'Network'], ['run', 'Run Algorithm', 'Play'], ['visualizer', 'Algorithm Visualizer', 'Eye'], ['final', 'Final Allocation', 'ClipboardCheck'], ['timetable', 'Timetable', 'CalendarDays'],
  ['analytics', 'Analytics', 'BarChart3'], ['analysis', 'Algorithm Analysis', 'Sigma'], ['about', 'About Project', 'Info'], ['settings', 'Settings', 'Settings']];
const PAGES = { dashboard: Dashboard, input: InputConfig, exams: ExamsPage, rooms: RoomsPage, slots: SlotsPage, graph: GraphPage, run: RunPage, visualizer: Visualizer, final: FinalPage, timetable: TimetablePage, analytics: Analytics, analysis: AnalysisPage, about: AboutPage, settings: SettingsPage };

function App() {
  const [page, setPage] = useState(() => (location.hash || '#dashboard').slice(1));
  const [menu, setMenu] = useState(false);
  const [state, setState] = useState(null); const [loadError, setLoadError] = useState('');
  const [toasts, setToasts] = useState([]); const [confirmCfg, setConfirmCfg] = useState(null);
  const [graphPos, setGraphPos] = useState({});
  const [settings, setSettingsState] = useState(() => { try { return { ordering: 'degree_desc', speed: 1200, ...JSON.parse(localStorage.getItem('era_settings') || '{}') }; } catch (e) { return { ordering: 'degree_desc', speed: 1200 }; } });
  const setSettings = patch => setSettingsState(s => { const n = { ...s, ...patch }; try { localStorage.setItem('era_settings', JSON.stringify(n)); } catch (e) { /* ignore */ } return n; });

  useEffect(() => { const h = () => { setPage((location.hash || '#dashboard').slice(1)); setMenu(false); window.scrollTo(0, 0); }; window.addEventListener('hashchange', h); return () => window.removeEventListener('hashchange', h); }, []);
  const navigate = useCallback(id => { location.hash = id; }, []);
  const toast = useCallback((type, msg) => { const id = Math.random().toString(36).slice(2); setToasts(t => [...t, { id, type, msg }]); setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4500); }, []);
  const confirm = useCallback(opts => new Promise(res => setConfirmCfg({ ...opts, res })), []);
  const closeConfirm = v => { if (confirmCfg) confirmCfg.res(v); setConfirmCfg(null); };
  const refresh = useCallback(async () => { try { setState(await api('/state')); setLoadError(''); } catch (e) { setLoadError(e.message); } }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const loadDemo = useCallback(async () => {
    if (state && (state.exams.length || state.rooms.length || state.timeslots.length)) {
      const ok = await confirm({ title: 'Load demo data?', message: 'This replaces all current exams, rooms, time slots and conflicts with the demo data set.', confirmText: 'Load demo data' }); if (!ok) return;
    }
    try { await api('/demo', { method: 'POST' }); setGraphPos({}); await refresh(); toast('success', 'Demo data loaded: 8 exams, 4 rooms, 3 time slots. Now click RUN ALGORITHM.'); } catch (e) { toast('error', e.message); }
  }, [state, confirm, refresh, toast]);
  const resetProject = useCallback(async () => {
    const ok = await confirm({ title: 'Reset project?', message: 'All exams, rooms, time slots, conflicts and the generated allocation will be permanently deleted.', confirmText: 'Reset project', danger: true }); if (!ok) return;
    try { await api('/reset', { method: 'POST' }); setGraphPos({}); await refresh(); toast('info', 'Project reset. You can start again.'); navigate('dashboard'); } catch (e) { toast('error', e.message); }
  }, [confirm, refresh, toast, navigate]);

  const ctx = { state, refresh, toast, confirm, navigate, settings, setSettings, graphPos, setGraphPos, loadDemo, resetProject };
  const Page = PAGES[page] || Dashboard;

  return (
    <AppCtx.Provider value={ctx}>
      <div id="app-shell" className="min-h-screen">
        {menu && <div className="fixed inset-0 z-30 bg-black/60 lg:hidden" onClick={() => setMenu(false)} />}
        <aside className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-white/10 bg-navy-900/95 backdrop-blur transition-transform lg:translate-x-0 ${menu ? 'translate-x-0' : '-translate-x-full'}`}>
          <div className="border-b border-white/10 p-5">
            <div className="flex items-center gap-3"><div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-sky-400 to-indigo-500 text-white shadow-lg shadow-sky-500/30"><Icon name="Network" size={22} /></div>
              <div><div className="text-[11px] font-bold leading-tight tracking-wide text-sky-300">MALLA REDDY<br />VISHVAVIDHYAPEETH</div></div></div>
            <div className="mt-3 text-sm font-semibold text-white">Exam Room Allocation System</div>
          </div>
          <nav className="flex-1 space-y-0.5 overflow-y-auto p-3">{NAV.map(([id, label, ic]) => (
            <a key={id} href={'#' + id} className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${page === id ? 'bg-sky-500/15 text-sky-200 shadow-inner' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}><Icon name={ic} size={18} />{label}</a>))}</nav>
          <div className="border-t border-white/10 p-4 text-xs text-slate-500">Developed by<div className="font-semibold text-slate-300">THOKALA SHESHVITH</div></div>
        </aside>
        <div className="lg:pl-72">
          <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-white/10 bg-navy-950/80 px-4 py-3 backdrop-blur sm:px-6">
            <button className="rounded-lg p-2 text-slate-300 hover:bg-white/10 lg:hidden" aria-label="Open menu" onClick={() => setMenu(true)}><Icon name="Menu" size={22} /></button>
            <div className="min-w-0 flex-1"><div className="truncate text-sm font-extrabold tracking-wide text-white sm:text-base">EXAM ROOM ALLOCATION SYSTEM</div><div className="truncate text-xs text-slate-400">Graph Coloring &amp; Greedy Algorithm Visualization · Student: THOKALA SHESHVITH</div></div>
            <Btn icon="Sparkles" onClick={loadDemo} className="hidden !px-3 !py-2 sm:inline-flex">LOAD DEMO DATA</Btn>
            <Btn variant="ghost" icon="RotateCcw" onClick={resetProject} className="!px-3 !py-2"><span className="hidden sm:inline">RESET PROJECT</span></Btn>
          </header>
          <main className="mx-auto max-w-7xl p-4 sm:p-6">
            {loadError && !state ? <EmptyState icon="WifiOff" title="Could not reach the backend" text={loadError} action={<Btn onClick={refresh}>Try again</Btn>} />
              : !state ? <div className="space-y-4"><Skeleton className="h-56" /><div className="grid grid-cols-2 gap-4 xl:grid-cols-6">{[...Array(6)].map((_, i) => <Skeleton key={i} className="h-28" />)}</div><Skeleton className="h-64" /></div>
              : <div key={page} className="page-in"><Page /></div>}
          </main>
        </div>
        <ToastHost toasts={toasts} />
        <ConfirmDialog cfg={confirmCfg} onClose={closeConfirm} />
      </div>
      {state && <PrintSheet state={state} />}
    </AppCtx.Provider>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
</script>
</body>
</html>
"""

# ==========================================================================================
# ENTRY POINT
# ==========================================================================================
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    url = f"http://127.0.0.1:{port}"
    print("=" * 70)
    print("  EXAM ROOM ALLOCATION SYSTEM  |  MALLA REDDY VISHVAVIDHYAPEETH")
    print("  Graph Coloring + Greedy Algorithm  |  Student: THOKALA SHESHVITH")
    print(f"  Open: {url}    (press CTRL+C to stop)")
    print("=" * 70)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False)
