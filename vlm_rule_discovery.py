#!/usr/bin/env python3
"""
vlm_rule_discovery_viz.py

VLM-based rule-discovery pipeline with visualizations.

- Parses episode logs (episodes/*.txt or trajectories*.jsonl)
- Builds symbolic frames, prompts, queries VLM (dry/http/openai)
- Parses VLM responses and normalizes rules
- Verifies rules across episodes (support / matches / precision)
- Clusters similar rules and picks canonical representatives
- Optionally refines low-precision rules with VLM
- Produces visualizations:
    - bar chart: support & precision per discovered rule
    - cluster summary plot (cluster sizes)
    - heatmap of pairwise similarity (text-sim)
    - per-rule example images (renders ascii snapshots + meta)
- Saves:
    - rules jsonl (raw VLM responses)
    - canonical jsonl
    - rules_summary.json (verification + clusters + refinements)
    - plots and images under <out_dir>/vlm_viz/

Usage examples:

# Dry-run (writes prompts, doesn't call any model)
python vlm_rule_discovery_viz.py --run_dir ./runs/fixed_law0_base/run_numlaws_3 --vlm dry --out rules_vlm.jsonl --out_summary rules_summary.json

# HTTP VLM (local server)
python vlm_rule_discovery_viz.py --run_dir ./runs/fixed_law0_base/run_numlaws_3 --vlm http --vlm_url http://localhost:8000/predict --out rules_vlm.jsonl --out_summary rules_summary.json

# OpenAI (requires OPENAI_API_KEY)
python vlm_rule_discovery_viz.py --run_dir ./runs/fixed_law0_base/run_numlaws_3 --vlm openai --model gpt-4o-mini --out rules_vlm.jsonl --out_summary rules_summary.json
"""
from __future__ import annotations

import argparse, json, os, re, time, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import difflib
import math
import textwrap
import logging
import itertools

# optional libs
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

# plotting, images, tqdm
HAS_PLOTTING = False
HAS_PIL = False
HAS_IMAGEIO = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    HAS_PLOTTING = True
    HAS_PIL = True
    HAS_IMAGEIO = True
except Exception:
    HAS_PLOTTING = False
    HAS_PIL = False
    HAS_IMAGEIO = False

try:
    from tqdm import tqdm, trange
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("vlm_rule_discovery")

# -------------------------
# Episode parsing utilities
# -------------------------
EPISODE_SNAPSHOT_BLOCK_RE = re.compile(r"(?P<grid>(?:[ .A-Za-z]{2,}.*\n)+)(?P<meta>step=.*\n)", flags=re.MULTILINE)

def find_episode_files(run_dir: str) -> List[Path]:
    p = Path(run_dir)
    ep_dir = p / "episodes"
    if ep_dir.exists() and ep_dir.is_dir():
        episodes = sorted(ep_dir.glob("episode_*.txt"))
        return episodes
    # fallback to trajectories jsonl
    trajs = sorted(p.glob("trajectories*.jsonl"))
    return trajs

def parse_meta_from_block(meta_line: str) -> Dict[str,Any]:
    meta = {}
    parts = [p.strip() for p in re.split(r"\|", meta_line)]
    for p in parts:
        if "=" in p:
            k,v = p.split("=",1); meta[k.strip()] = v.strip()
        elif p.lower().startswith("step "):
            m = re.match(r"Step\s+(\d+)", p, flags=re.IGNORECASE)
            if m: meta["step"] = int(m.group(1))
        else:
            if "action=" in p:
                k,v = p.split("=",1); meta[k.strip()] = v.strip()
    am = re.search(r"action=([\w_+-]+)", meta_line)
    if am: meta["action"] = am.group(1)
    rm = re.search(r"reward=([-\d\.]+)", meta_line)
    if rm:
        try: meta["reward"] = float(rm.group(1))
        except: pass
    return meta

def parse_episode_file(path: Path) -> Dict:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    frames = []
    matches = list(EPISODE_SNAPSHOT_BLOCK_RE.finditer(txt))
    if matches:
        for m in matches:
            ascii_block = m.group("grid").rstrip("\n")
            meta_line = m.group("meta").strip()
            parsed_meta = parse_meta_from_block(meta_line)
            frames.append({"ascii": ascii_block, "meta_line": meta_line, "parsed_meta": parsed_meta})
    else:
        # fallback: split by ----
        parts = re.split(r"-{5,}", txt)
        for part in parts:
            part = part.strip()
            if not part: continue
            lines = part.splitlines()
            meta_line = None; ascii_lines = []
            for i,ln in enumerate(lines):
                if "step=" in ln or ln.startswith("Step "):
                    meta_line = ln.strip(); ascii_lines = lines[:i]; break
            if meta_line:
                ascii_block = "\n".join(ascii_lines).rstrip("\n")
                parsed_meta = parse_meta_from_block(meta_line)
                frames.append({"ascii": ascii_block, "meta_line": meta_line, "parsed_meta": parsed_meta})
    # rule_events
    rule_events = []
    for m in re.finditer(r"RULE_EVENT:\s*([\w_+-]+)\s*-\s*(.+)", txt):
        rule_events.append({"event": m.group(1), "desc": m.group(2).strip()})
    # JSONL RULE_EVENT embedded
    for m in re.finditer(r"RULE_EVENT:\s*(\{.+?\})", txt, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(1)); rule_events.append(obj)
        except Exception:
            pass
    return {"path": str(path), "frames": frames, "rule_events": rule_events, "raw": txt[:2000]}

def parse_trajectories_jsonl(path: Path) -> List[Dict]:
    outs=[]
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip(): continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        frames=[]
        for step in obj.get("trajectory", []):
            meta = f"step={step.get('step')} agent={tuple(step.get('agent_pos'))} inv={step.get('inventory')} action={step.get('action_meaning')} reward={step.get('reward')}"
            frames.append({"ascii": "", "meta_line": meta, "parsed_meta": parse_meta_from_block(meta)})
        outs.append({"path": str(path), "frames": frames, "rule_events": obj.get("rule_events", [])})
    return outs

# -------------------------
# Symbolic abstraction (kept from original + hardened)
# -------------------------
def ascii_to_state_dict(ascii_block: str, meta: Dict[str,Any]=None) -> Dict[str,Any]:
    if not ascii_block or ascii_block.strip() == "":
        if meta is None: return {}
        st = {}
        ag = re.search(r"agent=\(?\s*([0-9]+)\s*,\s*([0-9]+)\)?", meta.get("meta_line", "") if isinstance(meta, dict) else "")
        if ag: st["agent_pos"] = [int(ag.group(1)), int(ag.group(2))]
        inv = meta.get("inv") if isinstance(meta, dict) else None
        st["has_key"] = 1 if inv and "key" in str(inv) else 0
        st["has_sword"] = 1 if inv and "sword" in str(inv) else 0
        return st

    lines = ascii_block.splitlines()
    height = len(lines)
    width = max(len(line)//2 for line in lines) if lines else 0

    agent = key = sword = lock = monster = None
    monster_alive = False
    has_key = False; has_sword = False

    for y,ln in enumerate(lines):
        if len(ln) < 2*width:
            ln = ln.ljust(2*width)
        for x in range(width):
            cell = ln[2*x:2*x+2].strip()
            if not cell or cell == '.': continue
            if cell.startswith('A'):
                agent = (x,y)
                if 'k' in cell: has_key=True
                if 's' in cell: has_sword=True
            elif 'K' in cell:
                key = (x,y)
            elif 'S' in cell:
                sword = (x,y)
            elif 'L' in cell:
                lock = (x,y)
            elif 'M' in cell:
                monster = (x,y); monster_alive=True
            elif 'm' in cell:
                monster = (x,y); monster_alive=False

    state = {
        'agent_pos': list(agent) if agent is not None else None,
        'key_pos': list(key) if key is not None else None,
        'sword_pos': list(sword) if sword is not None else None,
        'lock_pos': list(lock) if lock is not None else None,
        'monster_pos': list(monster) if monster is not None else None,
        'monster_alive': bool(monster_alive),
        'has_key': int(has_key or (meta and meta.get('inv') and 'key' in str(meta.get('inv')))) if (meta or has_key) else 0,
        'has_sword': int(has_sword or (meta and meta.get('inv') and 'sword' in str(meta.get('inv')))) if (meta or has_sword) else 0
    }
    return state

def state_to_facts(state: Dict[str,Any]) -> List[str]:
    facts=[]
    if not state: return facts
    a = state.get('agent_pos'); 
    if a: facts.append(f"At(agent,{a[0]},{a[1]})")
    k = state.get('key_pos')
    if k: facts.append(f"At(key,{k[0]},{k[1]})")
    s = state.get('sword_pos')
    if s: facts.append(f"At(sword,{s[0]},{s[1]})")
    L = state.get('lock_pos')
    if L: facts.append(f"At(lock,{L[0]},{L[1]})")
    m = state.get('monster_pos')
    if m:
        facts.append(f"At(monster,{m[0]},{m[1]})")
        facts.append("MonsterAlive(True)" if state.get('monster_alive') else "MonsterAlive(False)")
    if state.get('has_key'): facts.append("Has(agent,key)")
    if state.get('has_sword'): facts.append("Has(agent,sword)")
    if a and m:
        if abs(a[0]-m[0]) + abs(a[1]-m[1]) == 1: facts.append("Adjacent(agent,monster)")
    return facts

# -------------------------
# VLM prompt & callers (unchanged, but kept safe)
# -------------------------
PDDL_FEWSHOT = [
    {"input": "FRAME: At(agent,1,1); At(key,2,1); action=pick -> reward=0.1",
     "output": {"id":"pick_key", "rule_text":"If agent on key tile and picks -> agent gets key and key removed","type":"deterministic","condition":"At(agent,X,Y) and At(key,X,Y) and action==pick","effect":"Has(agent,key); At(key,X,Y)=None","confidence":0.9}},
    {"input": "FRAME: At(agent,3,2); Has(agent,key); At(lock,3,2); action=use -> reward=success, done",
     "output": {"id":"use_key_lock", "rule_text":"If agent at lock and has key and uses -> terminal success","type":"terminal","condition":"At(agent,X,Y) and At(lock,X,Y) and Has(agent,key) and action==use","effect":"done=True; reward=success","confidence":0.95}},
]

PROMPT_HEADER = """You are an assistant that extracts compact PDDL-like rules describing a small gridworld's dynamics.
Output MUST be JSON: an array of rule objects. Each rule object must contain:
 - id (string),
 - rule_text (human-readable),
 - type (one of deterministic, probabilistic, terminal, movement, interaction),
 - condition (a symbolic condition using predicates like At(agent,X,Y), At(key,X,Y), Has(agent,key), Adjacent(agent,monster), action==pick),
 - effect (symbolic effects like Has(agent,key); At(key,X,Y)=None; MonsterAlive=False; reward=...; done=True),
 - probability (optional for probabilistic rules),
 - examples (list of short example descriptors: "ep12:step5"),
 - confidence (0..1 estimate).

FEW-SHOT EXAMPLES:
"""

def build_prompt_for_episode_symbolic(ep_info: Dict, max_frames: int=40) -> str:
    frames = ep_info.get("frames", [])
    if len(frames) > max_frames:
        take_last = max_frames // 2
        take_early = max_frames - take_last
        use = frames[:take_early] + frames[-take_last:]
    else:
        use = frames
    ex_text = "\n\n".join([f"INPUT: {e['input']}\nOUTPUT: {json.dumps(e['output'])}" for e in PDDL_FEWSHOT])
    frames_text=[]
    for i,f in enumerate(use):
        state = ascii_to_state_dict(f.get("ascii",""), f.get("parsed_meta",{}))
        facts = state_to_facts(state)
        action = f.get("parsed_meta",{}).get("action") or f.get("parsed_meta",{}).get("act") or ""
        frames_text.append(f"FRAME {i+1}: facts: {', '.join(facts)} ; action={action} ; meta={f.get('meta_line')}")
    frames_joined = "\n".join(frames_text)
    prompt = PROMPT_HEADER + ex_text + "\n\nEPISODE FRAMES:\n" + frames_joined + "\n\nPlease output a JSON array of rules that explain the dynamics observed. Only output valid JSON."
    return prompt[:15000]

def call_vlm_openai(prompt: str, model: str="gpt-4o-mini", temp: float=0.0, max_tokens: int=1024) -> str:
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = key
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"You are a rule extraction assistant."},
                      {"role":"user","content":prompt}],
            temperature=float(temp),
            max_tokens=int(max_tokens)
        )
        return resp["choices"][0]["message"]["content"]
    except Exception:
        resp = openai.Completion.create(engine=model, prompt=prompt, temperature=float(temp), max_tokens=int(max_tokens))
        return resp["choices"][0]["text"]

def call_vlm_http(prompt: str, url: str, timeout: int=60) -> str:
    if not HAS_REQUESTS:
        raise RuntimeError("requests not installed")
    data = {"prompt": prompt, "max_tokens": 1024}
    r = requests.post(url, json=data, timeout=timeout)
    r.raise_for_status()
    try:
        j = r.json()
        if isinstance(j, dict) and "text" in j: return j["text"]
        if isinstance(j, dict) and "output" in j: return j["output"]
        return json.dumps(j)
    except Exception:
        return r.text

# -------------------------
# Parsing VLM output & DSL
# -------------------------
def extract_json_from_text(text: str) -> Any:
    if not text or not text.strip(): return None
    try: return json.loads(text)
    except Exception:
        m = re.search(r"(\[|\{)", text)
        if not m: return None
        try:
            candidate = text[m.start():].strip()
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)
            return json.loads(candidate)
        except Exception:
            start = m.start(); sub = text[start:]
            if sub.startswith("["):
                cnt = 0
                for i,ch in enumerate(sub):
                    if ch == '[': cnt += 1
                    elif ch == ']': cnt -= 1
                    if cnt == 0:
                        try: return json.loads(sub[:i+1])
                        except: break
            return None

def parse_rule_dsl(line: str) -> Optional[Dict]:
    try:
        part = line.strip()
        if part.lower().startswith("if "): part = part[3:]
        if "->" in part: cond, eff = part.split("->",1)
        elif "=>" in part: cond, eff = part.split("=>",1)
        else: return None
        cond = cond.strip(); eff = eff.strip(); prob = None
        pm = re.search(r"\[p\s*=\s*([0-9\.]+)\]", eff)
        if pm:
            prob = float(pm.group(1)); eff = re.sub(r"\[p\s*=\s*([0-9\.]+)\]", "", eff).strip()
        return {"condition": cond, "effect": eff, "probability": prob}
    except Exception:
        return None

# -------------------------
# Rule verification engine (unchanged)
# -------------------------
def check_condition_on_state(condition: str, state: Dict[str,Any], action: Optional[str]=None) -> bool:
    cond = condition.strip()
    parts = [p.strip() for p in re.split(r"\bAND\b", cond, flags=re.IGNORECASE)]
    for p in parts:
        if not p: continue
        m = re.match(r"action\s*==\s*([A-Za-z0-9_+-]+)", p)
        if m:
            if action is None: return False
            if action != m.group(1): return False
            else: continue
        m = re.match(r"At\(\s*([a-zA-Z_]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", p)
        if m:
            ent = m.group(1); x = int(m.group(2)); y = int(m.group(3))
            key = None
            if ent == "agent": key = state.get('agent_pos')
            elif ent == "key": key = state.get('key_pos')
            elif ent == "sword": key = state.get('sword_pos')
            elif ent == "lock": key = state.get('lock_pos')
            elif ent == "monster": key = state.get('monster_pos')
            if not key or key[0] != x or key[1] != y: return False
            continue
        if p.lower().startswith("has("):
            if "has(agent,key)" in p.lower():
                if not state.get('has_key'): return False
                else: continue
            if "has(agent,sword)" in p.lower():
                if not state.get('has_sword'): return False
                else: continue
        m = re.match(r"MonsterAlive\(\s*(True|False)\s*\)", p, flags=re.IGNORECASE)
        if m:
            val = m.group(1).lower() == "true"
            if bool(state.get('monster_alive')) != val: return False
            continue
        if p.lower().startswith("adjacent("):
            a = state.get('agent_pos'); mpos = state.get('monster_pos')
            if not a or not mpos: return False
            if abs(a[0]-mpos[0]) + abs(a[1]-mpos[1]) != 1: return False
            continue
        return False
    return True

def check_effect_between_states(effect: str, s_before: Dict[str,Any], s_after: Dict[str,Any]) -> bool:
    effs = [e.strip() for e in re.split(r";", effect)]
    for e in effs:
        if not e: continue
        if e.lower().startswith("has(agent,key)"):
            if not s_after.get('has_key'): return False
            else: continue
        if e.lower().startswith("has(agent,sword)"):
            if not s_after.get('has_sword'): return False
            else: continue
        m = re.match(r"At\(\s*key\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*None", e)
        if m:
            if s_after.get('key_pos') is not None: return False
            else: continue
        m = re.match(r"MonsterAlive\s*=\s*(True|False)", e, flags=re.IGNORECASE)
        if m:
            val = m.group(1).lower() == "true"
            if bool(s_after.get('monster_alive')) != val: return False
            else: continue
        m = re.match(r"At\(\s*([a-zA-Z_]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", e)
        if m:
            ent = m.group(1); x = int(m.group(2)); y = int(m.group(3))
            pos = None
            if ent == "agent": pos = s_after.get('agent_pos')
            elif ent == "key": pos = s_after.get('key_pos')
            elif ent == "sword": pos = s_after.get('sword_pos')
            elif ent == "lock": pos = s_after.get('lock_pos')
            elif ent == "monster": pos = s_after.get('monster_pos')
            if not pos or pos[0] != x or pos[1] != y: return False
            else: continue
        return False
    return True

def evaluate_rule_on_episodes(rule: Dict, episodes: List[Dict], max_examples: int=100) -> Dict:
    cond = rule.get("condition") or rule.get("cond") or ""
    effect = rule.get("effect") or ""
    support = 0; matches = 0; examples=[]
    for ep_idx, ep in enumerate(episodes, start=1):
        frames = ep.get("frames", [])
        states = []
        for f in frames:
            st = ascii_to_state_dict(f.get("ascii",""), f.get("parsed_meta",{}))
            states.append({"state": st, "action": f.get("parsed_meta",{}).get("action")})
        for i in range(len(states)-1):
            s_before = states[i]["state"]; s_after = states[i+1]["state"]; action = states[i]["action"]
            if check_condition_on_state(cond, s_before, action):
                support += 1
                if check_effect_between_states(effect, s_before, s_after):
                    matches += 1
                    if len(examples) < max_examples:
                        examples.append({"ep": ep_idx, "step": i, "desc": "cond_true -> effect_true"})
                else:
                    if len(examples) < max_examples:
                        examples.append({"ep": ep_idx, "step": i, "desc": "cond_true -> effect_false"})
    precision = matches / support if support>0 else None
    return {"support": support, "matches": matches, "precision": precision, "examples": examples[:10]}

# -------------------------
# Clustering
# -------------------------
def cluster_rules(rule_texts: List[str], threshold: float=0.8) -> List[List[int]]:
    n = len(rule_texts); assigned = [False]*n; clusters=[]
    for i in range(n):
        if assigned[i]: continue
        group=[i]; assigned[i]=True
        for j in range(i+1,n):
            if assigned[j]: continue
            ratio = difflib.SequenceMatcher(None, rule_texts[i], rule_texts[j]).ratio()
            if ratio >= threshold:
                group.append(j); assigned[j]=True
        clusters.append(group)
    return clusters

# -------------------------
# Image rendering for ascii + meta
# -------------------------
def render_ascii_to_image(ascii_text: str, meta_line: str = "", img_w: int = 320, img_h: int = 320, bg=(255,255,255), fg=(0,0,0)):
    if not HAS_PIL:
        raise RuntimeError("Pillow is required for rendering images")
    try: font = ImageFont.load_default()
    except Exception: font = None
    lines = (ascii_text + "\n" + meta_line).splitlines()
    img = Image.new("RGB", (img_w, img_h), color=bg)
    draw = ImageDraw.Draw(img)
    # robust char size
    try:
        if font and hasattr(draw, "textsize"):
            char_w, char_h = draw.textsize("M", font=font)
        elif font and hasattr(font, "getsize"):
            char_w, char_h = font.getsize("M")
        elif hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0,0), "M", font=font); char_w = bbox[2]-bbox[0]; char_h = bbox[3]-bbox[1]
        else: char_w, char_h = (6,11)
    except Exception:
        char_w, char_h = (6,11)
    margin = 4
    max_line_len = max((len(l) for l in lines), default=0)
    req_w = max_line_len * char_w + 2*margin
    req_h = len(lines) * char_h + 2*margin
    if req_w > img_w or req_h > img_h:
        img = Image.new("RGB", (max(req_w, img_w), max(req_h, img_h)), color=bg)
        draw = ImageDraw.Draw(img)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill=fg, font=font)
        y += char_h
    return img

# -------------------------
# Visualizations
# -------------------------
def plot_rule_support_precision(verification_results: List[Dict], out_dir: Path):
    if not HAS_PLOTTING: 
        logger.warning("matplotlib not available; skipping rule support/precision plots")
        return
    names = [ (r.get("id") or r.get("rule_text")[:30]) for r in verification_results ]
    supports = [ r.get("support",0) for r in verification_results ]
    precisions = [ r.get("precision") if r.get("precision") is not None else float('nan') for r in verification_results ]

    fig, ax1 = plt.subplots(figsize=(max(6, len(names)*0.5), 4))
    x = range(len(names))
    ax1.bar(x, supports, alpha=0.6, label="support (times cond true)")
    ax1.set_xlabel("Discovered rule"); ax1.set_ylabel("support (count)")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, precisions, color='C1', marker='o', label="precision")
    ax2.set_ylabel("precision (match / support)")
    ax1.set_title("Rule support & precision")
    fig.tight_layout()
    out = out_dir / "rule_support_precision.png"
    fig.savefig(out); plt.close(fig)
    logger.info("Saved plot: %s", out)

def plot_cluster_summary(clustered: List[Dict], verification_results: List[Dict], out_dir: Path):
    if not HAS_PLOTTING:
        logger.warning("matplotlib not available; skipping cluster plots")
        return
    cluster_sizes = [ len(c["members"]) for c in clustered ]
    cluster_ids = [ c["cluster_id"] for c in clustered ]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(cluster_ids, cluster_sizes, color='C2')
    ax.set_xlabel("Cluster ID"); ax.set_ylabel("members")
    ax.set_title("Rule cluster sizes")
    fig.tight_layout(); p = out_dir / "cluster_sizes.png"; fig.savefig(p); plt.close(fig)
    logger.info("Saved cluster sizes plot: %s", p)

    # pairwise similarity heatmap for canonical representatives
    texts = [ c["rep"]["rule_text"] for c in clustered ]
    n = len(texts)
    if n >= 2:
        sim = [[difflib.SequenceMatcher(None, texts[i], texts[j]).ratio() for j in range(n)] for i in range(n)]
        fig, ax = plt.subplots(figsize=(max(4,n), max(4,n)))
        im = ax.imshow(sim, vmin=0, vmax=1, cmap=cm.viridis)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([f"c{c['cluster_id']}" for c in clustered], rotation=45)
        ax.set_yticklabels([f"c{c['cluster_id']}" for c in clustered])
        fig.colorbar(im, ax=ax, label="text similarity")
        ax.set_title("Pairwise similarity of cluster representatives")
        fig.tight_layout(); p = out_dir / "cluster_similarity.png"; fig.savefig(p); plt.close(fig)
        logger.info("Saved cluster similarity heatmap: %s", p)

def save_rule_examples_images(verification_results: List[Dict], episode_entries: List[Dict], out_dir: Path, max_examples_per_rule: int = 3):
    if not HAS_PIL:
        logger.warning("Pillow is not available -> skipping per-rule example images")
        return
    examples_dir = out_dir / "rule_examples"; examples_dir.mkdir(exist_ok=True)
    # Build flat list of episodes with frames for quick access
    # episode_entries is list of dicts with 'frames'
    for ridx, r in enumerate(verification_results):
        rid = r.get("id", f"r{ridx}")
        examples = r.get("examples", []) or []
        saved = 0
        for ex in examples:
            if saved >= max_examples_per_rule: break
            ep_idx = ex.get("ep", None)
            step = ex.get("step", None)
            if ep_idx is None or step is None: continue
            # episode_entries indexes start at 0 for ep_idx=1
            if ep_idx-1 < 0 or ep_idx-1 >= len(episode_entries): continue
            ep = episode_entries[ep_idx-1]
            frames = ep.get("frames", [])
            if step < 0 or step >= len(frames): continue
            # render the ascii + meta_line for the frame and the following frame (to show effect)
            f_before = frames[step]
            f_after = frames[step+1] if (step+1) < len(frames) else None
            img_before = render_ascii_to_image(f_before.get("ascii",""), f_before.get("meta_line",""))
            if f_after:
                img_after = render_ascii_to_image(f_after.get("ascii",""), f_after.get("meta_line",""))
                # combine side-by-side
                w1,h1 = img_before.size; w2,h2 = img_after.size
                h = max(h1,h2); w = w1 + w2
                new = Image.new("RGB", (w,h), (255,255,255))
                new.paste(img_before, (0,0)); new.paste(img_after, (w1,0))
            else:
                new = img_before
            fname = examples_dir / f"{rid}_ep{ep_idx}_step{step}.png"
            new.save(str(fname))
            saved += 1
        if saved == 0:
            # write a small text file to record lack of examples
            (examples_dir / f"{rid}_no_examples.txt").write_text("No examples saved for this rule.")
    logger.info("Saved per-rule example images (if available) to %s", examples_dir)

# -------------------------
# Refinement prompt builder
# -------------------------
def build_refinement_prompt(rule: Dict, counterexamples: List[Dict], few_shot: List[Dict]=PDDL_FEWSHOT) -> str:
    header = "Refine or disambiguate the following rule using the counterexamples below. Return a JSON object with fields: id, rule_text, type, condition, effect, probability (optional), confidence (0..1)\n"
    examples = "\n\n".join([f"INPUT: {e['input']}\nOUTPUT: {json.dumps(e['output'])}" for e in few_shot])
    ce_text = "\n".join([f"- ep{c['ep']} step{c['step']}: {c['desc']}" for c in counterexamples[:20]])
    prompt = header + "Original rule:\n" + json.dumps(rule, indent=2) + "\n\nCounterexamples (where rule condition was true but effect failed):\n" + ce_text + "\n\nExamples:\n" + examples + "\n\nPlease output a single JSON object with a refined rule (no commentary)."
    return prompt[:12000]

# -------------------------
# Pipeline core (main)
# -------------------------
def run_pipeline(run_dir: str,
                 vlm: str = "dry",
                 vlm_url: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 max_episodes: Optional[int] = 200,
                 max_frames_per_ep: int = 60,
                 sample_successful_only: bool = True,
                 out_rules_file: str = "rules_vlm.jsonl",
                 out_summary_file: str = "rules_summary.json",
                 refine_low_precision: bool = True,
                 precision_threshold: float = 0.85,
                 cluster_threshold: float = 0.82,
                 random_seed: int = 0,
                 visualize: bool = True):
    random.seed(random_seed)
    run_p = Path(run_dir)
    assert run_p.exists(), f"run_dir {run_dir} not found"
    ep_files = find_episode_files(run_dir)
    episode_entries=[]
    if not ep_files:
        print("No episode files found.")
        return
    # trajectories.jsonl case
    if ep_files and any("trajectories" in str(x.name) and x.suffix==".jsonl" for x in ep_files):
        for f in ep_files:
            if "trajectories" in f.name and f.suffix==".jsonl":
                episode_entries.extend(parse_trajectories_jsonl(f))
    else:
        for f in ep_files:
            try:
                ep_info = parse_episode_file(f)
                success=False
                for ev in ep_info.get("rule_events", []):
                    if ev.get("event","") in ("used_key_on_lock","used_key_and_sword_on_lock"):
                        success=True
                if sample_successful_only and not success:
                    continue
                episode_entries.append(ep_info)
            except Exception as e:
                logger.exception("Parse failed for %s: %s", f, e)
    if not episode_entries:
        print("No (successful) episodes found. Try --sample_successful_only False")
        return
    if max_episodes:
        episode_entries = episode_entries[:max_episodes]
    logger.info("Processing %d episodes from %s", len(episode_entries), run_dir)

    out_rules_path = Path(out_rules_file)
    out_summary_path = Path(out_summary_file)
    wf = out_rules_path.open("w", encoding="utf-8")

    discovered_rules = []
    # Step 1: query VLM per-episode
    iterator = range(1, len(episode_entries)+1)
    if HAS_TQDM:
        iterator = trange(1, len(episode_entries)+1, desc="VLM queries")
    for idx in iterator:
        ep = episode_entries[idx-1]
        prompt = build_prompt_for_episode_symbolic(ep, max_frames=max_frames_per_ep)
        response_text = ""
        parsed = []
        try:
            if vlm == "dry":
                pfile = out_rules_path.with_suffix(f".ep{idx}.prompt.txt")
                pfile.write_text(prompt, encoding="utf-8")
                logger.info("Wrote prompt: %s", pfile)
                response_text = ""
            elif vlm == "openai":
                response_text = call_vlm_openai(prompt, model=model, temp=0.0, max_tokens=1024)
            elif vlm == "http":
                if not vlm_url: raise RuntimeError("Must pass --vlm_url for http")
                response_text = call_vlm_http(prompt, vlm_url)
            else:
                raise RuntimeError("Unknown vlm option: "+str(vlm))
            parsed = extract_json_from_text(response_text) or []
            if isinstance(parsed, dict): parsed = [parsed]
            if not parsed:
                # fallback: parse DSL-like lines from text
                parsed = []
                for line in (response_text or "").splitlines():
                    d = parse_rule_dsl(line)
                    if d: parsed.append(d)
        except Exception as e:
            logger.exception("VLM call/parse failed for ep %d: %s", idx, e)
            parsed = []
        # write raw responses (if parsed empty, write whatever response_text)
        if parsed:
            for r in parsed:
                discovered_rules.append({"episode_idx": idx, "raw": r if isinstance(r, (str, dict)) else str(r), "parsed": r if isinstance(r, dict) else None})
                wf.write(json.dumps({"episode": idx, "response": r}) + "\n")
        else:
            wf.write(json.dumps({"episode": idx, "response_raw_text": response_text}) + "\n")
        time.sleep(0.12)
    wf.close()
    logger.info("Wrote raw VLM outputs to %s", out_rules_path)

    if not discovered_rules:
        print("No rules discovered by VLM. Check prompts or run with --vlm dry to inspect prompts.")
        return

    # Normalize parsed rules
    structured_rules=[]
    for dr in discovered_rules:
        parsed = dr.get("parsed"); raw = dr.get("raw")
        if isinstance(parsed, dict):
            rule = {"id": parsed.get("id") or f"r{len(structured_rules)}",
                    "rule_text": parsed.get("rule_text") or str(parsed),
                    "type": parsed.get("type") or "deterministic",
                    "condition": parsed.get("condition") or parsed.get("cond") or "",
                    "effect": parsed.get("effect") or "",
                    "probability": parsed.get("probability"),
                    "confidence": parsed.get("confidence") or 0.5,
                    "examples": parsed.get("examples") or []}
            structured_rules.append(rule)
        else:
            if isinstance(raw, str):
                d = parse_rule_dsl(raw)
                if d:
                    rule = {"id": f"r{len(structured_rules)}", "rule_text": raw, "type":"deterministic", "condition": d.get("condition"), "effect": d.get("effect"), "probability": d.get("probability"), "confidence": 0.4}
                    structured_rules.append(rule)
                else:
                    structured_rules.append({"id": f"r{len(structured_rules)}", "rule_text": str(raw), "type":"unknown", "condition":"","effect":"","probability":None,"confidence":0.2})

    # Step 2: verification across episodes
    logger.info("Verifying discovered rules across %d episodes...", len(episode_entries))
    verification_results = []
    it = range(len(structured_rules))
    if HAS_TQDM: it = trange(len(structured_rules), desc="Verify rules")
    for i in it:
        r = structured_rules[i]
        vr = evaluate_rule_on_episodes(r, episode_entries)
        r_copy = dict(r); r_copy.update(vr)
        verification_results.append(r_copy)

    # Step 3: clustering
    rule_texts = [r.get("rule_text","") for r in verification_results]
    clusters_idx = cluster_rules(rule_texts, threshold=cluster_threshold)
    clustered = []
    for ci, cluster in enumerate(clusters_idx):
        members = [verification_results[i] for i in cluster]
        rep = max(members, key=lambda x: (x.get("precision") or 0.0, x.get("confidence") or 0.0))
        clustered.append({"cluster_id": ci, "rep": rep, "members": members})

    # Step 4: refinement (optional)
    refinements=[]
    if refine_low_precision:
        to_refine = [r for r in verification_results if (r.get("precision") is None or (r.get("precision") < precision_threshold and r.get("support",0)>0))]
        logger.info("Refinement candidates: %d", len(to_refine))
        for r in to_refine:
            ce = [m for m in r.get("examples",[]) if isinstance(m, dict) and m.get("desc","").endswith("effect_false")]
            if not ce:
                eps = evaluate_rule_on_episodes(r, episode_entries, max_examples=50)
                ce = [e for e in eps.get("examples",[]) if "effect_false" in e.get("desc","")]
            if not ce: continue
            prompt = build_refinement_prompt(r, ce)
            response_text=""; parsed=None
            try:
                if vlm == "dry":
                    pf = out_summary_path.with_suffix(f".refine_{r.get('id')}.prompt.txt")
                    pf.write_text(prompt, encoding="utf-8"); logger.info("Wrote refinement prompt: %s", pf); continue
                elif vlm == "openai":
                    response_text = call_vlm_openai(prompt, model=model, temp=0.0, max_tokens=512)
                elif vlm == "http":
                    response_text = call_vlm_http(prompt, vlm_url)
                parsed = extract_json_from_text(response_text)
            except Exception:
                logger.exception("Refinement VLM call failed for rule %s", r.get("id"))
            if parsed:
                refinements.append({"original": r, "refined": parsed, "response_snippet": (response_text or "")[:400]})
            time.sleep(0.15)

    # Save outputs + canonical rules
    out_summary = {
        "run_dir": run_dir,
        "num_input_episodes": len(episode_entries),
        "discovered_rules_count": len(structured_rules),
        "verification_results": verification_results,
        "clusters": [{"cluster_id": c["cluster_id"], "representative": c["rep"]["rule_text"], "members_count": len(c["members"])} for c in clustered],
        "refinements": refinements
    }
    out_summary_path.write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    logger.info("Saved summary to %s", out_summary_path)

    canon_path = out_rules_path.with_name(out_rules_path.stem + "_canonical.jsonl")
    with canon_path.open("w", encoding="utf-8") as cf:
        for c in clustered:
            cf.write(json.dumps(c["rep"], indent=None) + "\n")
    logger.info("Saved canonical rules to %s", canon_path)

    # Visualizations & per-rule images
    if visualize:
        viz_dir = run_p / "vlm_viz"
        viz_dir.mkdir(exist_ok=True)
        try:
            plot_rule_support_precision(verification_results, viz_dir)
            plot_cluster_summary(clustered, verification_results, viz_dir)
            save_rule_examples_images(verification_results, episode_entries, viz_dir)
        except Exception:
            logger.exception("Failed to generate visualizations")
    return out_summary_path

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--vlm", choices=("dry","openai","http"), default="dry")
    p.add_argument("--vlm_url", type=str, default="")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--max_episodes", type=int, default=200)
    p.add_argument("--max_frames_per_ep", type=int, default=60)
    p.add_argument("--sample_successful_only", action="store_true", default=True)
    p.add_argument("--out", type=str, default="rules_vlm.jsonl")
    p.add_argument("--out_summary", type=str, default="rules_summary.json")
    p.add_argument("--refine_low_precision", action="store_true", default=False)
    p.add_argument("--precision_threshold", type=float, default=0.85)
    p.add_argument("--cluster_threshold", type=float, default=0.82)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_visualize", action="store_true", default=False)
    args = p.parse_args()

    run_pipeline(run_dir=args.run_dir, vlm=args.vlm, vlm_url=(args.vlm_url or None),
                 model=args.model, max_episodes=args.max_episodes, max_frames_per_ep=args.max_frames_per_ep,
                 sample_successful_only=args.sample_successful_only, out_rules_file=args.out, out_summary_file=args.out_summary,
                 refine_low_precision=args.refine_low_precision, precision_threshold=args.precision_threshold,
                 cluster_threshold=args.cluster_threshold, random_seed=args.seed, visualize=not args.no_visualize)

if __name__ == "__main__":
    main()
