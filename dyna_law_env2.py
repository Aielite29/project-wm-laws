#!/usr/bin/env python3
"""
dyna_law_env_v2_opt_fixed.py

Final corrected and hardened script with analysis, plotting and imagination episode generation.
Keep behavior identical to previous version; this file adds:
 - saving per-run training rewards
 - saving run results json
 - plotting training curves and comparison across law subsets
 - model-quality plots (NLL / top-1)
 - "imagination" episode generation (GIFs) created from the learned model

Usage: same as before. Outputs (in out_dir/run...):
 - trajectories_run{run}.jsonl
 - rule_archive_run{run}.jsonl
 - q_table_run{run}.npz (.json inside)
 - train_rewards_run{run}.npy
 - results_run{run}.json
 - summary_run{run}.json
 - plots/*.png
 - imagination/*.gif
"""
from __future__ import annotations

import argparse
import json
import os
import random
import csv
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

# plotting / gif libs (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("dyna_law")

# Utilities
def _pos_tuple(p):
    return None if p is None else (p[0], p[1])

class StateEncoder:
    """Assign integer ids to immutable observation tuples. Fast encode/lookup."""
    def __init__(self):
        self._map: Dict[Tuple, int] = {}
        self._rev: List[Tuple] = []

    def encode(self, obs: Tuple) -> int:
        if obs in self._map:
            return self._map[obs]
        idx = len(self._rev)
        self._map[obs] = idx
        self._rev.append(obs)
        return idx

    def decode(self, idx: Optional[int]):
        if idx is None:
            return None
        if not (0 <= idx < len(self._rev)):
            raise IndexError(f"StateEncoder.decode: index out of range: {idx}")
        return self._rev[idx]

    def __len__(self):
        return len(self._rev)


# -----------------------
# Laws (unchanged)
# -----------------------
class Law:
    name = "base-law"
    def apply(self, state, action, env):
        raise NotImplementedError

class SimpleAdventureLaw(Law):
    name = "simple"
    def apply(self, state, action, env):
        s = dict(state)
        ax = action
        reward = -0.01; done = False; info = {}

        agent_pos = tuple(s.get('agent_pos', (0,0)))
        key_pos = _pos_tuple(s.get('key_pos'))
        sword_pos = _pos_tuple(s.get('sword_pos'))
        lock_pos = tuple(s.get('lock_pos', (0,0)))
        monster_pos = _pos_tuple(s.get('monster_pos')) if s.get('monster_alive', False) else None

        if ax in (0,1,2,3):
            dx = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[ax]
            new_pos = (agent_pos[0]+dx[0], agent_pos[1]+dx[1])
            if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
                if s.get('monster_alive', False) and new_pos == monster_pos:
                    if s.get('has_sword'):
                        s['agent_pos'] = list(new_pos)
                        env.rule_events.append({"event":"blocked_but_sword_contact","desc":"Stepped into monster tile while monster alive and has sword. Monster remains alive until attacked."})
                        reward = -0.05
                    else:
                        s['agent_pos'] = list(new_pos)
                        reward = env.death_penalty; done=True
                        env.rule_events.append({"event":"death_by_monster","desc":"Agent moved into monster tile without sword => death."})
                else:
                    s['agent_pos'] = list(new_pos)
            else:
                reward = -0.02
                env.rule_events.append({"event":"bumped_wall","desc":"Agent tried to move out of bounds."})
        elif ax == 4:  # pick
            picked = False
            if key_pos is not None and agent_pos == key_pos:
                s['has_key'] = 1; s['key_pos'] = None; picked=True
                env.rule_events.append({"event":"picked_key","desc":"Agent picked up the key at its tile."}); reward = 0.1
            if sword_pos is not None and agent_pos == sword_pos:
                s['has_sword'] = 1; s['sword_pos'] = None; picked=True
                env.rule_events.append({"event":"picked_sword","desc":"Agent picked up the sword at its tile."}); reward = 0.1 if reward < 0.1 else reward
            if not picked:
                reward = -0.02; env.rule_events.append({"event":"pick_nothing","desc":"Agent picked but nothing present."})
        elif ax == 5:  # attack
            if s.get('monster_alive'):
                axp = tuple(s['agent_pos']); mx,my = s['monster_pos']
                if abs(axp[0]-mx)+abs(axp[1]-my)==1 and s.get('has_sword'):
                    s['monster_alive'] = False; reward = 0.5
                    env.rule_events.append({"event":"killed_monster","desc":"Agent attacked adjacent monster while having sword -> monster dies."})
                else:
                    reward = -0.05; env.rule_events.append({"event":"attack_failed","desc":"Attack had no effect (monster not adjacent or no sword)."})
            else:
                reward = -0.02; env.rule_events.append({"event":"attack_no_monster","desc":"Attack when monster already dead."})
        elif ax == 6:  # use key on lock
            if agent_pos == lock_pos and s.get('has_key'):
                reward = env.success_reward; done = True
                env.rule_events.append({"event":"used_key_on_lock","desc":"Agent used key on lock -> level cleared."})
            else:
                reward = -0.02; env.rule_events.append({"event":"use_failed","desc":"Use failed (not at lock or no key)."})
        else:
            reward = -0.05; env.rule_events.append({"event":"invalid_action","desc":"Invalid action."})
        return s, reward, done, info

class LockRequiresSwordLaw(Law):
    name = "lock_requires_sword"
    def __init__(self):
        self._base = SimpleAdventureLaw()
    def apply(self, state, action, env):
        s2, r, done, info = self._base.apply(state, action, env)
        if action == 6:
            agent_pos = tuple(s2.get('agent_pos', (0,0))); lock_pos = tuple(s2.get('lock_pos', (0,0)))
            if agent_pos == lock_pos and s2.get('has_key') and s2.get('has_sword'):
                r = env.success_reward; done = True
                env.rule_events.append({"event":"used_key_and_sword_on_lock","desc":"Agent used key and sword on lock -> level cleared."})
            else:
                r = -0.02; env.rule_events.append({"event":"use_failed","desc":"Use failed (need both key and sword at lock)."})
        return s2, r, done, info

class AggressiveMonsterLaw(Law):
    name = "aggressive"
    def apply(self, state, action, env):
        base = SimpleAdventureLaw()
        s_after, r_action, done, info = base.apply(state, action, env)
        s = dict(s_after)
        r = r_action
        if not done and s.get('monster_alive'):
            axp = tuple(s['agent_pos']); mx,my = s['monster_pos']
            dx,dy = 0,0
            if axp[0] > mx: dx=1
            elif axp[0] < mx: dx=-1
            if axp[1] > my: dy=1
            elif axp[1] < my: dy=-1
            if dx != 0:
                new_mpos = (mx+dx, my)
            else:
                new_mpos = (mx, my+dy)
            if 0 <= new_mpos[0] < env.width and 0 <= new_mpos[1] < env.height:
                s['monster_pos'] = list(new_mpos)
                if tuple(s['monster_pos']) == tuple(s['agent_pos']):
                    if s.get('has_sword'):
                        s['monster_alive'] = False
                        env.rule_events.append({"event":"monster_collided_killed","desc":"Monster moved into agent while agent has sword -> monster dies."})
                        r = r_action + 0.3
                    else:
                        done = True
                        r = r_action + env.death_penalty
                        env.rule_events.append({"event":"monster_collision_death","desc":"Monster moved into agent without sword -> death."})
        return s, r, done, info

class ProbabilisticMonsterLaw(Law):
    name = "probabilistic"
    def __init__(self,p_death=0.9,p_kill=0.85):
        self.p_death=p_death; self.p_kill=p_kill
    def apply(self, state, action, env):
        s = dict(state); ax = action; reward=-0.01; done=False; info={}; rng = env.rng
        agent_pos = tuple(s.get('agent_pos', (0,0))); lock_pos = tuple(s.get('lock_pos', (0,0)))
        key_pos=_pos_tuple(s.get('key_pos')); sword_pos=_pos_tuple(s.get('sword_pos'))
        monster_pos = _pos_tuple(s.get('monster_pos')) if s.get('monster_alive') else None
        if ax in (0,1,2,3):
            dx = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[ax]
            new_pos=(agent_pos[0]+dx[0], agent_pos[1]+dx[1])
            if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
                if s.get('monster_alive') and new_pos == monster_pos:
                    if s.get('has_sword'):
                        s['agent_pos']=list(new_pos)
                        if rng.random() < self.p_kill:
                            s['monster_alive']=False; env.rule_events.append({"event":"stepped_and_killed_prob","desc":"Stepped into monster tile with sword and probabilistically killed it."}); reward=0.3
                        else:
                            env.rule_events.append({"event":"stepped_and_failed_kill","desc":"Stepped into monster tile with sword but failed to kill it."}); reward=-0.05
                    else:
                        if rng.random() < self.p_death:
                            s['agent_pos']=list(new_pos); reward = env.death_penalty; done=True; env.rule_events.append({"event":"prob_death","desc":"Agent stepped into monster tile and died probabilistically."})
                        else:
                            s['agent_pos']=list(new_pos); reward=-0.2; env.rule_events.append({"event":"prob_survive","desc":"Agent stepped into monster tile and survived probabilistically."})
                else:
                    s['agent_pos']=list(new_pos)
            else:
                reward=-0.02; env.rule_events.append({"event":"bumped_wall","desc":"Agent tried to move out of bounds."})
        elif ax == 4:
            picked=False
            if key_pos is not None and agent_pos==key_pos:
                s['has_key']=1; s['key_pos']=None; picked=True; env.rule_events.append({"event":"picked_key","desc":"Agent picked up the key at its tile."}); reward=0.1
            if sword_pos is not None and agent_pos==sword_pos:
                s['has_sword']=1; s['sword_pos']=None; picked=True; env.rule_events.append({"event":"picked_sword","desc":"Agent picked up the sword at its tile."}); reward=0.1 if reward<0.1 else reward
            if not picked:
                reward=-0.02; env.rule_events.append({"event":"pick_nothing","desc":"Agent picked but nothing present."})
        elif ax == 5:
            if s.get('monster_alive'):
                axp = tuple(s['agent_pos']); mx,my = s['monster_pos']
                if abs(axp[0]-mx)+abs(axp[1]-my)==1 and s.get('has_sword'):
                    if rng.random() < self.p_kill:
                        s['monster_alive']=False; env.rule_events.append({"event":"killed_monster_prob","desc":"Agent attacked adjacent monster with sword and probabilistically killed it."}); reward=0.5
                    else:
                        env.rule_events.append({"event":"attack_failed_prob","desc":"Agent attacked but failed probabilistically."}); reward=-0.05
                else:
                    reward=-0.05; env.rule_events.append({"event":"attack_failed","desc":"Attack had no effect (monster not adjacent or no sword)."})
            else:
                reward=-0.02; env.rule_events.append({"event":"attack_no_monster","desc":"Attack when monster already dead."})
        elif ax == 6:
            if agent_pos == lock_pos and s.get('has_key'):
                reward = env.success_reward; done=True; env.rule_events.append({"event":"used_key_on_lock","desc":"Agent used key on lock -> level cleared."})
            else:
                reward=-0.02; env.rule_events.append({"event":"use_failed","desc":"Use failed (not at lock or no key)."})
        else:
            reward=-0.05; env.rule_events.append({"event":"invalid_action","desc":"Invalid action."})
        return s, reward, done, info

class FleeingMonsterLaw(Law):
    name = "fleeing"
    def apply(self, state, action, env):
        base = SimpleAdventureLaw()
        s_after, r_action, done, info = base.apply(state, action, env)
        s = dict(s_after)
        if not done and s.get('monster_alive'):
            axp = tuple(s['agent_pos']); mx,my = s['monster_pos']
            candidates = []
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(0,0)]:
                n = (mx+dx, my+dy)
                if 0 <= n[0] < env.width and 0 <= n[1] < env.height:
                    if abs(n[0]-axp[0])+abs(n[1]-axp[1]) > abs(mx-axp[0])+abs(my-axp[1]):
                        candidates.append(n)
            if candidates:
                new_mpos = env.rng.choice(candidates); s['monster_pos'] = list(new_mpos)
        return s, r_action, done, info

ALL_LAW_CLASSES = [SimpleAdventureLaw, LockRequiresSwordLaw, AggressiveMonsterLaw, ProbabilisticMonsterLaw, FleeingMonsterLaw]


# -----------------------
# Environment (unchanged)
# -----------------------
class KeyLockEnv:
    def __init__(self, width=8, height=8, law: Optional[Law]=None, seed: int=0, success_reward=10.0, death_penalty=-10.0):
        self.width = int(width); self.height = int(height)
        if self.width * self.height < 5:
            raise ValueError("KeyLockEnv requires at least 5 cells (width*height >= 5)")
        self.rng = random.Random(int(seed))
        self.action_meanings = {0:'up',1:'down',2:'left',3:'right',4:'pick',5:'attack',6:'use'}
        self.law = law if law is not None else SimpleAdventureLaw()
        self.success_reward = float(success_reward); self.death_penalty = float(death_penalty)
        self.reset(randomize=True)

    def set_law(self, law: Law):
        self.law = law

    def reset(self, randomize=True):
        cells = [(x,y) for x in range(self.width) for y in range(self.height)]
        if randomize:
            self.rng.shuffle(cells)
        if len(cells) < 5:
            raise RuntimeError("Not enough cells to place all objects")
        agent=cells.pop(); key=cells.pop(); sword=cells.pop(); lock=cells.pop(); monster=cells.pop()
        self.state = {'agent_pos':list(agent),'key_pos':list(key),'sword_pos':list(sword),'lock_pos':list(lock),'monster_pos':list(monster),'monster_alive':True,'has_key':0,'has_sword':0}
        self.rule_events = []
        self.trajectory = []
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        s=self.state
        mpos = s.get('monster_pos')
        if mpos is None:
            mpos = [-1,-1]
        return (s['agent_pos'][0],s['agent_pos'][1],
                None if s['key_pos'] is None else s['key_pos'][0], None if s['key_pos'] is None else s['key_pos'][1],
                None if s['sword_pos'] is None else s['sword_pos'][0], None if s['sword_pos'] is None else s['sword_pos'][1],
                s['lock_pos'][0], s['lock_pos'][1],
                int(bool(s.get('monster_alive'))), mpos[0], mpos[1],
                int(bool(s.get('has_key'))), int(bool(s.get('has_sword'))))

    def step(self, action):
        self.steps += 1
        next_state, reward, done, info = self.law.apply(self.state, action, self)
        for k in ('agent_pos','key_pos','sword_pos','lock_pos','monster_pos'):
            if k in next_state and next_state[k] is not None and not isinstance(next_state[k], list):
                next_state[k] = list(next_state[k])
        self.state = next_state
        desc = self._describe_step(action, reward, done); self.trajectory.append(desc)
        if self.steps >= 500:
            done = True; self.rule_events.append({"event":"timeout","desc":"Episode timed out after max steps."})
        return self._get_obs(), reward, done, {"rule_events": list(self.rule_events)}

    def _describe_step(self, action, reward, done):
        s=self.state; agent=tuple(s['agent_pos']); inv=[]
        if s.get('has_key'): inv.append('key')
        if s.get('has_sword'): inv.append('sword')
        inv = ", ".join(inv) if inv else "none"
        return {"step":self.steps,"agent_pos":agent,"inventory":inv,"action":action,"action_meaning":self.action_meanings.get(action,str(action)),"reward":float(reward),"monster_alive":bool(s.get('monster_alive')),"key_pos": None if s.get('key_pos') is None else tuple(s.get('key_pos')) ,"sword_pos": None if s.get('sword_pos') is None else tuple(s.get('sword_pos')) ,"lock_pos": tuple(s.get('lock_pos')) ,"done":bool(done)}

    def ascii_snapshot(self):
        s=self.state; grid=[[" ." for _ in range(self.width)] for _ in range(self.height)]
        lx,ly = s['lock_pos']; grid[ly][lx]=" L"
        mx,my = s['monster_pos']; grid[my][mx] = " M" if s.get('monster_alive') else " m"
        if s.get('key_pos') is not None: kx,ky=s['key_pos']; grid[ky][kx]=" K"
        if s.get('sword_pos') is not None: sx,sy=s['sword_pos']; grid[sy][sx]=" S"
        ax,ay = s['agent_pos']; inv=""
        if s.get('has_key'): inv+="k"
        if s.get('has_sword'): inv+="s"
        rep="A"+(inv if inv else " "); rep=f"{rep[:2]:>2}"; grid[ay][ax]=rep
        lines=["".join(grid[y]) for y in range(self.height)]
        meta = f"step={self.steps} | agent={tuple(s['agent_pos'])} | inv={'key' if s.get('has_key') else 'none'}/{ 'sword' if s.get('has_sword') else 'none'} | monster_alive={s.get('monster_alive')} | law={self.law.name}"
        return "\n".join(lines)+"\n"+meta


# -----------------------
# DynaQ Agent (unchanged, with model_counts)
# -----------------------
class DynaQAgent:
    def __init__(self, actions=7, alpha=0.4, gamma=0.98, epsilon_start=1.0, epsilon_end=0.05, planning_steps=20, seed=0, use_prioritized=False):
        self.actions = list(range(actions))
        self.n_actions = int(actions)
        self.alpha = float(alpha); self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start); self.epsilon_end = float(epsilon_end)
        self.planning_steps = int(planning_steps)
        self.q: Dict[int, np.ndarray] = {}
        self.model_counts: Dict[Tuple[int,int], Dict[Optional[int], int]] = {}
        self.model_reward_sums: Dict[Tuple[int,int], float] = {}
        self.model_reward_counts: Dict[Tuple[int,int], int] = {}
        self.keys_list: List[Tuple[int,int]] = []
        self.key_index_map: Dict[Tuple[int,int], int] = {}
        self.priority: Dict[Tuple[int,int], float] = {}
        self.priority_array: Optional[np.ndarray] = None
        self.priority_dirty = True
        self.rng = random.Random(int(seed))
        self.use_prioritized = bool(use_prioritized)

    def _ensure_q(self, s_id: int) -> np.ndarray:
        if s_id not in self.q:
            self.q[s_id] = np.zeros(self.n_actions, dtype=float)
        return self.q[s_id]

    def choose_action(self, s_id: int, epsilon: float) -> int:
        if self.rng.random() < float(epsilon):
            return self.rng.randrange(self.n_actions)
        qs = self.q.get(s_id)
        if qs is None:
            return 0
        maxv = qs.max()
        candidates = np.flatnonzero(qs == maxv)
        return int(self.rng.choice(candidates.tolist()))

    def _register_model_key(self, key: Tuple[int,int]):
        if key not in self.key_index_map:
            self.key_index_map[key] = len(self.keys_list)
            self.keys_list.append(key)
            self.priority[key] = 0.0
            self.priority_dirty = True

    def learn(self, s_id: int, a: int, r: float, s2_id: Optional[int]):
        q_s = self._ensure_q(s_id)
        q_next = 0.0 if s2_id is None else self._ensure_q(s2_id).max()
        td_target = r + self.gamma * q_next
        td_error = td_target - q_s[a]
        q_s[a] += self.alpha * td_error

        key = (s_id, a)
        self._register_model_key(key)
        counts = self.model_counts.setdefault(key, {})
        counts[s2_id] = counts.get(s2_id, 0) + 1
        self.model_reward_sums[key] = self.model_reward_sums.get(key, 0.0) + float(r)
        self.model_reward_counts[key] = self.model_reward_counts.get(key, 0) + 1

        if self.use_prioritized:
            self.priority[key] = abs(td_error)
            self.priority_dirty = True

    def planning(self):
        if not self.keys_list:
            return
        if self.use_prioritized and self.priority_dirty:
            arr = np.array([self.priority.get(k, 0.0) + 1e-12 for k in self.keys_list], dtype=float)
            total = arr.sum()
            if total > 0:
                self.priority_array = arr / total
            else:
                self.priority_array = None
            self.priority_dirty = False

        for _ in range(self.planning_steps):
            if self.use_prioritized and self.priority_array is not None:
                weights = self.priority_array.tolist()
                idx = self.rng.choices(range(len(self.keys_list)), weights=weights, k=1)[0]
            else:
                idx = self.rng.randrange(len(self.keys_list))
            k = self.keys_list[idx]
            s_id, a = k
            counts = self.model_counts.get(k, {})
            total = sum(counts.values())
            if total == 0:
                continue
            r_pick = self.rng.randint(1, total); cum=0; s2_choice=None
            for s2k, c in counts.items():
                cum += c
                if r_pick <= cum:
                    s2_choice = s2k; break
            reward_est = (self.model_reward_sums.get(k, 0.0) / self.model_reward_counts.get(k, 1))
            q_s = self._ensure_q(s_id)
            q_next = 0.0 if s2_choice is None else self._ensure_q(s2_choice).max()
            q_s[a] += self.alpha * (reward_est + self.gamma * q_next - q_s[a])
            if self.use_prioritized:
                self.priority[k] = max(0.0, self.priority.get(k,0.0) * 0.99)
                self.priority_dirty = True

    def _epsilon_for_episode(self, ep_idx: int, total_eps: int) -> float:
        if total_eps <= 1:
            return self.epsilon_end
        frac = min(1.0, (ep_idx) / float(total_eps - 1))
        return float(self.epsilon_start * (1 - frac) + self.epsilon_end * frac)

    def save_q(self, path: str):
        try:
            q_dict = {str(k): v.tolist() for k, v in self.q.items()}
            p = Path(path)
            p_json = p.with_suffix('.json')
            with p_json.open('w', encoding='utf-8') as jf:
                json.dump(q_dict, jf)
            q_json = json.dumps(q_dict)
            np.savez_compressed(str(p), q_json=q_json)
        except Exception as e:
            logger.exception(f"Failed to save Q-table to {path}: {e}")


# -----------------------
# Training harness (extended with saving + plotting + imagination)
# -----------------------
def _obs_tuple_to_state_dict(obs):
    """Convert encoder-decoded obs tuple to env.state dict structure."""
    # obs format: (ax,ay, kx,ky, sx,sy, lx,ly, monster_alive, mx, my, has_key, has_sword)
    ax, ay, kx, ky, sx, sy, lx, ly, m_alive, mx, my, has_key, has_sword = obs
    state = {
        'agent_pos': [int(ax), int(ay)],
        'key_pos': None if kx is None else [int(kx), int(ky)],
        'sword_pos': None if sx is None else [int(sx), int(sy)],
        'lock_pos': [int(lx), int(ly)],
        'monster_pos': [int(mx), int(my)],
        'monster_alive': bool(int(m_alive)),
        'has_key': int(has_key),
        'has_sword': int(has_sword)
    }
    return state

def _render_ascii_to_image(ascii_text: str, img_w: int = 400, img_h: int = 400, bg=(255,255,255), fg=(0,0,0)):
    """Render ascii grid+meta text into a PIL Image (monospace)."""
    # Use default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    lines = ascii_text.splitlines()
    # compute font size
    # create image wide enough
    img = Image.new("RGB", (img_w, img_h), color=bg)
    draw = ImageDraw.Draw(img)
    # compute text block size: approximate char size
    if font:
        char_w, char_h = draw.textsize("M", font=font)
    else:
        char_w, char_h = (6, 11)
    margin = 4
    max_line_len = max(len(l) for l in lines) if lines else 0
    # compute required size
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

def _generate_imagination_gif(agent: DynaQAgent, encoder: StateEncoder, env: KeyLockEnv, out_path: Path, prefix="imagine", num_episodes=3, max_steps=200):
    """Generate imagination episodes by sampling the learned model transitions.
    For each episode:
      - pick a start state from encoder._rev (observed states) or env.reset()
      - run greedy policy but transition using model_counts sampled next-state
      - render frames (ascii -> image) and write GIF
    """
    imag_dir = out_path / "imagination"; imag_dir.mkdir(exist_ok=True)
    # choose a set of start states
    observed_states = encoder._rev if hasattr(encoder, "_rev") else []
    if not observed_states:
        # fallback: one env reset
        obs_real = env.reset(randomize=True)
        observed_states = [obs_real]
    rng = random.Random(0)
    for i in range(num_episodes):
        start_obs = rng.choice(observed_states)
        s_id = encoder.encode(start_obs)
        frames = []
        # set environment to decoded state for rendering
        current_obs = encoder.decode(s_id)
        env.state = _obs_tuple_to_state_dict(current_obs)
        # render initial
        frames.append(_render_ascii_to_image(env.ascii_snapshot()))
        done = False
        steps = 0
        while not done and steps < max_steps:
            # greedy action according to learned Q
            qs = agent.q.get(s_id, np.zeros(agent.n_actions))
            a = int(np.argmax(qs))
            key = (s_id, a)
            # sample predicted s2 from model_counts if available, else break
            if key not in agent.model_counts or not agent.model_counts[key]:
                break
            counts = agent.model_counts[key]
            total = sum(counts.values())
            r_pick = rng.randint(1, total); cum = 0; chosen_s2 = None
            for s2k, c in counts.items():
                cum += c
                if r_pick <= cum:
                    chosen_s2 = s2k; break
            # estimate reward as average
            reward_est = agent.model_reward_sums.get(key, 0.0) / max(1, agent.model_reward_counts.get(key, 1))
            # transition
            s2_id = chosen_s2
            # decode and set env state for rendering
            decoded = encoder.decode(s2_id) if s2_id is not None else None
            if decoded is None:
                break
            env.state = _obs_tuple_to_state_dict(decoded)
            frames.append(_render_ascii_to_image(env.ascii_snapshot()))
            # check terminal by seeing if any success rule event would apply (we can't compute rule_events easily here)
            # we use heuristic: if last frame had 'used_key_on_lock' in rules? Not available.
            # Instead, stop if agent at lock and has_key (simple heuristic)
            sdict = env.state
            if sdict.get('has_key') and sdict.get('agent_pos') == sdict.get('lock_pos'):
                break
            s_id = s2_id
            steps += 1
        # write gif
        gif_path = imag_dir / f"{prefix}_run_episode{i+1}.gif"
        try:
            # convert frames to images via imageio
            images = [np.array(im) for im in frames]
            imageio.mimsave(str(gif_path), images, duration=0.25)
            logger.info(f"Saved imagination GIF: {gif_path}")
        except Exception:
            # fallback: try PIL save
            try:
                frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], format="GIF", duration=250, loop=0)
                logger.info(f"Saved imagination GIF (PIL fallback): {gif_path}")
            except Exception:
                logger.exception(f"Failed to save imagination GIF to {gif_path}")

# -----------------------
# run_single_training (extended)
# -----------------------
def run_single_training(args, law_class_subset: List[type], out_dir: str, run_id: int=0):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    episodes_dir = out_path / "episodes"; episodes_dir.mkdir(exist_ok=True)
    trajectories_file = out_path / f"trajectories_run{run_id}.jsonl"
    rules_file = out_path / f"rule_archive_run{run_id}.jsonl"
    q_file = out_path / f"q_table_run{run_id}.npz"
    train_rewards_file = out_path / f"train_rewards_run{run_id}.npy"
    results_file = out_path / f"results_run{run_id}.json"

    for p in (trajectories_file, rules_file):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            logger.warning(f"Could not remove previous file {p}, continuing")

    env = KeyLockEnv(width=args.width, height=args.height, seed=args.seed, success_reward=args.success_reward, death_penalty=args.death_penalty)
    encoder = StateEncoder()
    law_instances = [cls() for cls in law_class_subset]
    agent = DynaQAgent(actions=7, alpha=args.alpha, gamma=args.gamma, epsilon_start=args.eps_start, epsilon_end=args.eps_end, planning_steps=args.planning, seed=args.seed, use_prioritized=args.use_prioritized)

    seen_rules = set()
    results = {"episode_rewards":[], "successes":0, "steps":[], "law_sequence":[]}

    episode_logs_buffer: List[Tuple[Path, str]] = []
    trajectories_buffer: List[str] = []
    rule_lines_buffer: List[str] = []
    flush_every = max(1, int(args.flush_every))

    mode = args.mode.lower()
    fixed_law_idx = int(args.fixed_law) if hasattr(args, "fixed_law") else 0
    if args.law_indices:
        try:
            fixed_law_idx = int(args.law_indices.split(",")[0])
        except Exception:
            fixed_law_idx = fixed_law_idx

    for ep in range(1, args.episodes + 1):
        try:
            if mode == "mixed":
                law_choice = random.choice(law_instances)
            else:
                law_choice = law_instances[fixed_law_idx % len(law_instances)]
            env.set_law(law_choice); results["law_sequence"].append(law_choice.name)

            obs = env.reset(randomize=True)
            s_id = encoder.encode(obs)
            total_r = 0.0; done=False; steps=0
            env.rule_events = []; env.trajectory = []
            epsilon = agent._epsilon_for_episode(ep-1, args.episodes)

            episode_lines: List[str] = []
            episode_lines.append(f"Episode {ep}\nLaw: {law_choice.name}\n" + "="*60 + "\n")
            episode_lines.append("Initial:\n" + env.ascii_snapshot() + "\n\n")

            while not done and steps < args.max_steps:
                a = agent.choose_action(s_id, epsilon)
                obs2, r, done, info = env.step(a)
                s2_id = encoder.encode(obs2)
                intrinsic = 0.0
                for ev in env.rule_events:
                    desc = ev.get("desc", "")
                    if desc not in seen_rules:
                        seen_rules.add(desc)
                        intrinsic += float(args.intrinsic_bonus)
                        rule_lines_buffer.append(json.dumps({"law": law_choice.name, **ev}))
                r_aug = float(r) + intrinsic
                total_r += r_aug; steps += 1

                episode_lines.append(f"Step {steps} | action={env.action_meanings.get(a)} | base_reward={r:.3f} | intrinsic={intrinsic:.3f}\n")
                episode_lines.append(env.ascii_snapshot() + "\n")
                for ev in env.rule_events:
                    episode_lines.append(f"RULE_EVENT: {ev.get('event')} - {ev.get('desc')}\n")
                episode_lines.append("-"*60 + "\n")

                agent.learn(s_id, a, r_aug, None if done else s2_id)
                agent.planning()
                s_id = s2_id

            results["episode_rewards"].append(total_r); results["steps"].append(steps)
            if done and any(ev.get("event","") in ("used_key_on_lock","used_key_and_sword_on_lock") for ev in env.rule_events):
                results["successes"] += 1
            traj_record = {"episode":ep,"law":law_choice.name,"total_reward":total_r,"steps":steps,"trajectory":env.trajectory,"rule_events":env.rule_events}
            trajectories_buffer.append(json.dumps(traj_record))

            ep_filename = episodes_dir / f"episode_{ep:05d}_run{run_id}.txt"
            episode_logs_buffer.append((ep_filename, "".join(episode_lines)))

            if ep % flush_every == 0:
                for fname, content in episode_logs_buffer:
                    try:
                        with fname.open('w', encoding='utf-8') as f:
                            f.write(content)
                    except Exception:
                        logger.exception(f"Failed to write episode log {fname}")
                episode_logs_buffer = []
                if trajectories_buffer:
                    try:
                        with trajectories_file.open('a', encoding='utf-8') as f:
                            f.write("\n".join(trajectories_buffer) + ("\n" if trajectories_buffer else ""))
                    except Exception:
                        logger.exception(f"Failed to append to trajectories file {trajectories_file}")
                    trajectories_buffer = []
                if rule_lines_buffer:
                    try:
                        with rules_file.open('a', encoding='utf-8') as rf:
                            rf.write("\n".join(rule_lines_buffer) + ("\n" if rule_lines_buffer else ""))
                    except Exception:
                        logger.exception(f"Failed to append to rules file {rules_file}")
                    rule_lines_buffer = []

            if args.verbose and (ep % max(1, args.episodes//10) == 0 or ep <= 5):
                avg_reward = np.mean(results["episode_rewards"][-min(50,len(results["episode_rewards"])):])
                print(f"Run {run_id} Ep {ep}/{args.episodes} | avg_reward(last50) {avg_reward:.3f} | successes {results['successes']}/{ep} | eps {epsilon:.3f}")

        except Exception as e:
            logger.exception(f"Exception during episode {ep}: {e}")
            if ep % flush_every == 0:
                try:
                    for fname, content in episode_logs_buffer:
                        with fname.open('w', encoding='utf-8') as f:
                            f.write(content)
                except Exception:
                    logger.exception("Failed to flush episode buffer after an exception")
                episode_logs_buffer = []

    # final flush
    try:
        for fname, content in episode_logs_buffer:
            with fname.open('w', encoding='utf-8') as f:
                f.write(content)
    except Exception:
        logger.exception("Failed to write remaining episode logs")

    try:
        if trajectories_buffer:
            with trajectories_file.open('a', encoding='utf-8') as f:
                f.write("\n".join(trajectories_buffer) + ("\n" if trajectories_buffer else ""))
    except Exception:
        logger.exception("Failed to write remaining trajectories buffer")

    try:
        if rule_lines_buffer:
            with rules_file.open('a', encoding='utf-8') as rf:
                rf.write("\n".join(rule_lines_buffer) + ("\n" if rule_lines_buffer else ""))
    except Exception:
        logger.exception("Failed to write remaining rule lines buffer")

    try:
        agent.save_q(str(q_file))
    except Exception:
        logger.exception("Failed to save Q table")

    # Save training rewards & results for plotting later
    try:
        np.save(train_rewards_file, np.array(results["episode_rewards"], dtype=float))
        with open(results_file, "w", encoding="utf-8") as rf:
            json.dump({"train_stats": results, "args": vars(args)}, rf, indent=2)
    except Exception:
        logger.exception("Failed to save train rewards/results")

    # EVALUATION and model-quality metrics (unchanged)
    eval_results = {}; model_quality = {}
    for law in law_instances:
        try:
            env.set_law(law)
            ep_rewards = []; ep_successes = 0; ep_steps = []
            nlls = []; top1 = 0; total_obs = 0
            for eidx in range(args.eval_episodes):
                obs = env.reset(randomize=True)
                s_id = encoder.encode(obs)
                done=False; total_r=0.0; steps=0; env.rule_events=[]; env.trajectory=[]
                while not done and steps < args.max_steps:
                    qs = agent.q.get(s_id, np.zeros(agent.n_actions))
                    a = int(np.argmax(qs))
                    obs2, r, done, info = env.step(a)
                    s2_id = encoder.encode(obs2)
                    key = (s_id, a)
                    if key in agent.model_counts:
                        counts = agent.model_counts[key]; total = sum(counts.values())
                        if total>0:
                            prob = counts.get(s2_id, 0) / total
                            nlls.append(-np.log(prob + 1e-12))
                            pred_s2 = max(counts.items(), key=lambda kv: kv[1])[0]
                            if pred_s2 == s2_id: top1 += 1
                            total_obs += 1
                    total_r += r; steps += 1; s_id = s2_id
                ep_rewards.append(total_r); ep_steps.append(steps)
                if any(ev.get("event","") in ("used_key_on_lock","used_key_and_sword_on_lock") for ev in env.rule_events):
                    ep_successes += 1
            eval_results[law.name] = {"avg_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0, "success_rate": float(ep_successes)/args.eval_episodes if args.eval_episodes>0 else 0.0, "avg_steps": float(np.mean(ep_steps)) if ep_steps else 0.0}
            model_quality[law.name] = {"avg_nll": float(np.mean(nlls)) if nlls else None, "top1_accuracy": float(top1/total_obs) if total_obs>0 else None, "obs_transitions": int(total_obs)}
        except Exception:
            logger.exception(f"Evaluation failed for law {law.name}")
            eval_results[law.name] = {"avg_reward": None, "success_rate": None, "avg_steps": None}
            model_quality[law.name] = {"avg_nll": None, "top1_accuracy": None, "obs_transitions": 0}

    summary = {"run_id": run_id, "mode": args.mode, "law_subset":[cls.__name__ for cls in law_class_subset], "train_stats":{"successes": results["successes"], "avg_reward_last50": float(np.mean(results["episode_rewards"][-50:])) if results["episode_rewards"] else 0.0}, "eval_results": eval_results, "model_quality": model_quality}
    try:
        with (out_path / f"summary_run{run_id}.json").open('w', encoding='utf-8') as sf: json.dump(summary, sf, indent=2)
    except Exception:
        logger.exception("Failed to write summary file")

    # PLOTTING & IMAGINATION (added)
    try:
        if HAS_PLOTTING:
            plots_dir = out_path / "plots"; plots_dir.mkdir(exist_ok=True)
            # training curve
            train_rewards = np.array(results["episode_rewards"], dtype=float)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(np.arange(1, len(train_rewards)+1), train_rewards, label="episode reward", alpha=0.6)
            if len(train_rewards) >= 5:
                window = min(50, max(5, len(train_rewards)//10))
                ma = np.convolve(train_rewards, np.ones(window)/window, mode='valid')
                ax.plot(np.arange(window, len(train_rewards)+1), ma, label=f"MA({window})", lw=2)
            ax.set_xlabel("Episode"); ax.set_ylabel("Total reward"); ax.set_title(f"Training curve run{run_id}")
            ax.legend(); fig.tight_layout()
            fig_path = plots_dir / f"train_curve_run{run_id}.png"; fig.savefig(str(fig_path)); plt.close(fig)

            # per-law eval bar plots
            law_names = list(eval_results.keys())
            avg_rewards = [eval_results[n]["avg_reward"] for n in law_names]
            succ_rates = [eval_results[n]["success_rate"] for n in law_names]
            fig, ax = plt.subplots(figsize=(10,4))
            x = np.arange(len(law_names))
            ax.bar(x - 0.15, avg_rewards, width=0.3, label="avg_reward")
            ax.bar(x + 0.15, succ_rates, width=0.3, label="success_rate")
            ax.set_xticks(x); ax.set_xticklabels(law_names, rotation=30, ha='right')
            ax.set_title("Per-law evaluation"); ax.legend(); fig.tight_layout()
            fig_path = plots_dir / f"per_law_eval_run{run_id}.png"; fig.savefig(str(fig_path)); plt.close(fig)

            # model quality plots (NLL & top1)
            nlls = [model_quality[n]["avg_nll"] if model_quality[n]["avg_nll"] is not None else np.nan for n in law_names]
            top1s = [model_quality[n]["top1_accuracy"] if model_quality[n]["top1_accuracy"] is not None else np.nan for n in law_names]
            fig, axes = plt.subplots(1,2, figsize=(12,4))
            axes[0].bar(x, nlls); axes[0].set_xticks(x); axes[0].set_xticklabels(law_names, rotation=30, ha='right'); axes[0].set_title("Model avg NLL")
            axes[1].bar(x, top1s); axes[1].set_xticks(x); axes[1].set_xticklabels(law_names, rotation=30, ha='right'); axes[1].set_title("Model top-1 acc")
            fig.tight_layout(); fig_path = plots_dir / f"model_quality_run{run_id}.png"; fig.savefig(str(fig_path)); plt.close(fig)
        else:
            logger.warning("Plotting libraries not available; skipping plots.")
    except Exception:
        logger.exception("Failed to generate plots")

    # Generate imagination episodes (GIFs) from the learned model
    try:
        if HAS_PLOTTING:
            _generate_imagination_gif(agent, encoder, env, out_path, prefix=f"imagine_run{run_id}", num_episodes=3, max_steps=min(200, args.max_steps))
        else:
            logger.warning("Skipping imagination GIFs because plotting libraries are not present")
    except Exception:
        logger.exception("Failed to generate imagination episodes")

    return summary

# -----------------------
# train_and_compare (unchanged, uses run_single_training)
# -----------------------
def train_and_compare(args):
    available = ALL_LAW_CLASSES.copy()
    max_num = min(len(available), args.max_num_laws if args.max_num_laws else len(available))
    base_out = args.out_dir
    if args.compare_num_laws:
        comparison_rows=[]
        for k in range(1, max_num+1):
            if args.law_indices:
                indices=[int(i) for i in args.law_indices.split(",") if i.strip()!='']
                subset=[available[i] for i in indices]
            else:
                subset = available[:k]
            out_dir_k = os.path.join(base_out, f"num_laws_{k}")
            print(f"Running experiment with {k} laws -> out_dir {out_dir_k}")
            summary = run_single_training(args, subset, out_dir_k, run_id=k)
            evals = summary.get("eval_results", {}); modelq = summary.get("model_quality", {})
            avg_reward = float(np.mean([v["avg_reward"] for v in evals.values() if v.get("avg_reward") is not None])) if evals else None
            avg_succ = float(np.mean([v["success_rate"] for v in evals.values() if v.get("success_rate") is not None])) if evals else None
            avg_nll = float(np.mean([v["avg_nll"] for v in modelq.values() if v.get("avg_nll") is not None])) if modelq else None
            avg_top1 = float(np.mean([v["top1_accuracy"] for v in modelq.values() if v.get("top1_accuracy") is not None])) if modelq else None
            comparison_rows.append({"num_laws":k, "avg_reward":avg_reward, "avg_success":avg_succ, "avg_model_nll":avg_nll, "avg_model_top1":avg_top1})
        csv_path = os.path.join(base_out, "comparison_num_laws.csv")
        try:
            with open(csv_path, "w", newline='', encoding='utf-8') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=comparison_rows[0].keys()); writer.writeheader();
                for r in comparison_rows: writer.writerow(r)
        except Exception:
            logger.exception("Failed to write comparison CSV")
        return {"comparison_csv": csv_path, "rows": comparison_rows}
    else:
        if args.law_indices:
            indices=[int(i) for i in args.law_indices.split(",") if i.strip()!='']
            subset=[ALL_LAW_CLASSES[i] for i in indices]
        else:
            n = args.num_laws if args.num_laws else len(ALL_LAW_CLASSES)
            subset = ALL_LAW_CLASSES[:n]
        out_dir = os.path.join(base_out, f"run_numlaws_{len(subset)}"); summary = run_single_training(args, subset, out_dir, run_id=0); return summary

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="mixed")
    parser.add_argument("--law_indices", type=str, default="")
    parser.add_argument("--num_laws", type=int, default=3)
    parser.add_argument("--compare_num_laws", action="store_true")
    parser.add_argument("--max_num_laws", type=int, default=5)
    parser.add_argument("--fixed_law", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--width", type=int, default=8); parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.4); parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eps_start", type=float, default=1.0); parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--planning", type=int, default=20); parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--intrinsic_bonus", type=float, default=0.6)
    parser.add_argument("--success_reward", type=float, default=10.0); parser.add_argument("--death_penalty", type=float, default=-10.0)
    parser.add_argument("--use_prioritized", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./runs/dyna_law_v2_opt")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--flush_every", type=int, default=10, help="How many episodes to buffer before writing files (increase for speed)")
    args = parser.parse_args()

    if args.flush_every <= 0:
        logger.warning("--flush_every must be positive; using 10")
        args.flush_every = 10
    if args.width * args.height < 5:
        raise SystemExit("Grid too small: width * height must be >= 5")

    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.compare_num_laws:
        logger.info("Running compare_num_laws ...")
        print(json.dumps(train_and_compare(args), indent=2))
    else:
        logger.info("Running single experiment ...")
        print(json.dumps(train_and_compare(args), indent=2))
