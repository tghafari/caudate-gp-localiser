#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face working-memory task for PsychoPy.

Features:
- Sternberg-style encoding -> delay -> probe
- Adjustable trial counts, timings, and set sizes
- Subject-level counterbalanced response mapping
- No scanner trigger synchronization
- Saves trial-level CSV data

Expected stimuli folder:
    stimuli/
        face01.jpg
        face02.jpg
        ...

Change CONFIG below to match your lab setup.
"""

from __future__ import annotations

import csv
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from psychopy import core, data, event, gui, visual


# =========================
# CONFIG
# =========================
CONFIG = {
    # Experiment identifiers
    "exp_name": "face_wm_task",
    "stimuli_dir": "stimuli",          # folder containing face images
    "output_dir": "data",              # where CSV files will be written

    # Trial structure
    "n_trials": 40,
    "set_sizes": [1, 2, 3, 4],         # possible numbers of encoding faces
    "trial_random_seed": None,         # set integer for reproducible runs, or None

    # Timing (seconds)
    "fixation_before_encoding": 2.5,   # initial fixation / blink period
    "item_duration": 0.3,              # duration each encoding face is shown
    "iti_between_items": 1.25,         # blank interval between encoding faces
    "delay_duration": 2.7,             # maintenance interval before probe
    "probe_duration": 0.3,             # probe display duration
    "feedback_duration": 0.5,          # feedback display duration
    "iti_after_trial": 0.5,            # optional blank after feedback

    # Response mapping
    # Assumes a 2-button response box / keyboard mapping.
    # Edit these if your hardware uses different key names.
    "button_yes_key": "4",
    "button_no_key": "5",

    # Counterbalancing rule:
    # odd participant numeric IDs -> YES=button_yes_key, NO=button_no_key
    # even participant numeric IDs -> YES=button_no_key, NO=button_yes_key
    "counterbalance_by_participant_parity": True,

    # Stimulus display
    "window_size": [1280, 720],
    "fullscr": True,
    "bg_color": "black",
    "text_color": "white",
    "stim_size": (0.35, 0.35),         # image size in PsychoPy norm units
    "face_y": 0.0,
    "fixation_text": "+",

    # Misc
    "show_instructions": True,
    "allow_escape": True,
    "practice_trials": 0,              # set >0 if you want practice
}


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class TrialSpec:
    trial_num: int
    set_size: int
    target_files: List[str]
    probe_file: str
    probe_is_match: bool


# =========================
# HELPERS
# =========================
def get_participant_id() -> str:
    info = {
        "participant": "",
        "session": "001",
    }
    dlg = gui.DlgFromDict(info, title=CONFIG["exp_name"])
    if not dlg.OK:
        core.quit()
    return str(info["participant"]).strip(), str(info["session"]).strip()


def participant_parity(participant_id: str) -> int:
    """
    Returns 0 for even, 1 for odd.
    Uses numeric digits in participant ID if present; otherwise derives from string.
    """
    digits = re.findall(r"\d+", participant_id)
    if digits:
        value = int("".join(digits))
    else:
        value = sum(ord(c) for c in participant_id)
    return value % 2


def get_response_mapping(participant_id: str) -> Dict[str, str]:
    """
    Returns mapping dict:
        {
            "yes": key_name,
            "no": key_name
        }
    """
    yes_key = CONFIG["button_yes_key"]
    no_key = CONFIG["button_no_key"]

    if CONFIG["counterbalance_by_participant_parity"]:
        if participant_parity(participant_id) == 0:
            # even
            return {"yes": no_key, "no": yes_key}
        else:
            # odd
            return {"yes": yes_key, "no": no_key}
    return {"yes": yes_key, "no": no_key}


def load_stimulus_paths(stimuli_dir: str) -> List[Path]:
    stim_path = Path(stimuli_dir)
    if not stim_path.exists():
        raise FileNotFoundError(f"Stimuli directory not found: {stim_path.resolve()}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    files = sorted([p for p in stim_path.iterdir() if p.suffix.lower() in exts])

    if len(files) < 4:
        raise ValueError("Need at least 4 face images in the stimuli folder.")

    return files


def build_trials(all_files: Sequence[Path], n_trials: int, set_sizes: Sequence[int]) -> List[TrialSpec]:
    """
    Creates randomized trials.
    Each trial:
      - samples a set size
      - selects unique target faces
      - probe is either a target face (match) or a novel face (non-match)
    """
    trials: List[TrialSpec] = []

    # Keep a pool for non-match probes
    all_file_names = [p.name for p in all_files]

    for t in range(1, n_trials + 1):
        set_size = random.choice(list(set_sizes))
        target_paths = random.sample(list(all_files), k=set_size)
        target_names = [p.name for p in target_paths]

        probe_is_match = random.choice([True, False])

        if probe_is_match:
            probe_file = random.choice(target_names)
        else:
            distractors = [f for f in all_file_names if f not in target_names]
            if not distractors:
                # very small stimulus set fallback
                probe_file = random.choice(all_file_names)
                probe_is_match = probe_file in target_names
            else:
                probe_file = random.choice(distractors)

        trials.append(
            TrialSpec(
                trial_num=t,
                set_size=set_size,
                target_files=target_names,
                probe_file=probe_file,
                probe_is_match=probe_is_match,
            )
        )

    random.shuffle(trials)
    # Reset trial numbering after shuffle for cleaner output
    for i, tr in enumerate(trials, start=1):
        tr.trial_num = i

    return trials


def wait_for_key(keys: Sequence[str]) -> str | None:
    keys_pressed = event.waitKeys(keyList=list(keys))
    return keys_pressed[0] if keys_pressed else None


def draw_centered_text(win: visual.Window, text: str, height: float = 0.06):
    stim = visual.TextStim(
        win,
        text=text,
        color=CONFIG["text_color"],
        height=height,
        wrapWidth=1.4,
        alignText="center",
        anchorHoriz="center",
        anchorVert="center",
    )
    stim.draw()
    win.flip()


def draw_fixation(win: visual.Window, text: str = "+", height: float = 0.08):
    fix = visual.TextStim(
        win,
        text=text,
        color=CONFIG["text_color"],
        height=height,
        anchorHoriz="center",
        anchorVert="center",
    )
    fix.draw()
    win.flip()


# =========================
# MAIN EXPERIMENT
# =========================
def main():
    if CONFIG["trial_random_seed"] is not None:
        random.seed(CONFIG["trial_random_seed"])

    participant_id, session_id = get_participant_id()
    response_map = get_response_mapping(participant_id)

    stim_files = load_stimulus_paths(CONFIG["stimuli_dir"])
    trials = build_trials(stim_files, CONFIG["n_trials"], CONFIG["set_sizes"])

    out_dir = Path(CONFIG["output_dir"]) / f"sub-{participant_id}" / f"ses-{session_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"sub-{participant_id}_ses-{session_id}_{CONFIG['exp_name']}.csv"

    # Window
    win = visual.Window(
        size=CONFIG["window_size"],
        fullscr=CONFIG["fullscr"],
        color=CONFIG["bg_color"],
        units="norm",
    )

    # Preload image stimuli for all faces in folder
    face_stimuli = {}
    for p in stim_files:
        face_stimuli[p.name] = visual.ImageStim(
            win,
            image=str(p),
            size=CONFIG["stim_size"],
            pos=(0.0, CONFIG["face_y"]),
        )

    probe_stimuli = {}
    for p in stim_files:
        probe_stimuli[p.name] = visual.ImageStim(
            win,
            image=str(p),
            size=CONFIG["stim_size"],
            pos=(0.0, CONFIG["face_y"]),
        )

    # Instruction screen
    if CONFIG["show_instructions"]:
        instr = (
            "Face Working Memory Task\n\n"
            f"YES = {response_map['yes']} key\n"
            f"NO  = {response_map['no']} key\n\n"
            "Remember the faces shown in each trial.\n"
            "Then decide whether the probe face was present in the set.\n\n"
            "Press any key to begin."
        )
        draw_centered_text(win, instr, height=0.05)
        event.waitKeys()

    # Data header
    fieldnames = [
        "participant",
        "session",
        "trial_num",
        "set_size",
        "target_files",
        "probe_file",
        "probe_is_match",
        "response_key",
        "response_label",
        "correct_response",
        "accuracy",
        "rt",
        "yes_key",
        "no_key",
    ]

    rows = []
    global_clock = core.Clock()

    # Optional practice could be inserted here if needed
    # practice_trials = build_trials(stim_files, CONFIG["practice_trials"], CONFIG["set_sizes"])

    for trial in trials:
        # Trial clock
        trial_clock = core.Clock()
        trial_start = global_clock.getTime()

        # Initial fixation / blink period
        draw_fixation(win, CONFIG["fixation_text"])
        core.wait(CONFIG["fixation_before_encoding"])

        # Encoding sequence
        event.clearEvents(eventType="keyboard")

        for idx, fname in enumerate(trial.target_files):
            face_stimuli[fname].draw()
            win.flip()
            core.wait(CONFIG["item_duration"])

            # Blank between items, but not after the last item
            if idx < len(trial.target_files) - 1:
                draw_fixation(win, CONFIG["fixation_text"])
                core.wait(CONFIG["iti_between_items"])

        # Delay period
        draw_fixation(win, CONFIG["fixation_text"])
        core.wait(CONFIG["delay_duration"])

        # Probe
        probe_stimuli[trial.probe_file].draw()
        win.flip()

        probe_onset = global_clock.getTime()
        resp_clock = core.Clock()
        response_key = None
        rt = None

        # Response window
        keys = [response_map["yes"], response_map["no"]]
        keys = list(dict.fromkeys(keys))  # preserve order, remove duplicates if any

        got_response = False
        while resp_clock.getTime() < CONFIG["probe_duration"]:
            keys_pressed = event.getKeys(keyList=keys + (["escape"] if CONFIG["allow_escape"] else []), timeStamped=resp_clock)
            if keys_pressed:
                k, t = keys_pressed[0]
                if CONFIG["allow_escape"] and k == "escape":
                    win.close()
                    core.quit()

                response_key = k
                rt = float(t)
                got_response = True
                break

        # Finish probe interval if response was early
        remaining = CONFIG["probe_duration"] - resp_clock.getTime()
        if remaining > 0:
            core.wait(remaining)

        # Determine correctness
        if response_key == response_map["yes"]:
            response_label = "yes"
        elif response_key == response_map["no"]:
            response_label = "no"
        else:
            response_label = ""

        correct_response = "yes" if trial.probe_is_match else "no"
        accuracy = int(response_label == correct_response) if got_response else 0

        # Feedback
        if got_response:
            fb_text = "Correct" if accuracy == 1 else "Incorrect"
        else:
            fb_text = "No response"

        draw_centered_text(win, fb_text, height=0.07)
        core.wait(CONFIG["feedback_duration"])

        # Optional ITI after trial
        draw_fixation(win, CONFIG["fixation_text"])
        core.wait(CONFIG["iti_after_trial"])

        rows.append(
            {
                "participant": participant_id,
                "session": session_id,
                "trial_num": trial.trial_num,
                "set_size": trial.set_size,
                "target_files": ";".join(trial.target_files),
                "probe_file": trial.probe_file,
                "probe_is_match": int(trial.probe_is_match),
                "response_key": response_key if response_key is not None else "",
                "response_label": response_label,
                "correct_response": correct_response,
                "accuracy": accuracy,
                "rt": rt if rt is not None else "",
                "yes_key": response_map["yes"],
                "no_key": response_map["no"],
            }
        )

    # Save data
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # End screen
    end_msg = f"Task complete.\n\nData saved to:\n{out_file}\n\nPress any key to exit."
    draw_centered_text(win, end_msg, height=0.05)
    event.waitKeys()

    win.close()
    core.quit()


if __name__ == "__main__":
    main()
