#!/usr/bin/env python3
"""
keylock_dyna_explore.py

Upgraded Key/Lock/Sword/Monster environment + Dyna-Q with:
 - Larger default grid (configurable)
 - Epsilon-linear decay schedule (explore -> exploit)
 - Intrinsic bonus on discovering NEW rule events (encourages exploration of rule-space)
 - Larger Dyna planning_steps by default
 - Per-episode ASCII frame logs for VLM "watching"
 - Trajectories and rule archive JSONL outputs

Usage:
  python keylock_dyna_explore.py --episodes 500 --width 8 --height 8 --out_dir ./runs/demo

Outputs:
 - <out_dir>/trajectories.jsonl
 - <out_dir>/rule_archive.jsonl
 - <out_dir>/episodes/episode_###.txt  (one file per episode; contains ASCII frames + rule events)
"""
import argparse, json, os, random, time
from collections import defaultdict, OrderedDict
import numpy as np

# -----------------------
# Law classes (same style)
# -----------------------
class Law:
    def apply(self, state, action, env):
        raise NotImplementedError
    def describe_rule_event(self, event):
        return json.dumps(event, sort_keys=True)

class SimpleAdventureLaw(Law):
    def apply(self, state, action, env):
        s = dict(state)
        ax = action
        reward = -0.01
        done = False
        info = {}

        agent_pos = tuple(s['agent_pos'])
        key_pos = tuple(s['key_pos']) if s['key_pos'] is not None else None
        sword_pos = tuple(s['sword_pos']) if s['sword_pos'] is not None else None
        lock_pos = tuple(s['lock_pos'])
        monster_pos = tuple(s['monster_pos']) if s['monster_alive'] else None

        # movement actions: 0:up,1:down,2:left,3:right
        if ax in (0,1,2,3):
            dx = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[ax]
            new_pos = (agent_pos[0]+dx[0], agent_pos[1]+dx[1])
            # bounds check
            if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
                if s['monster_alive'] and new_pos == monster_pos:
                    if s['has_sword']:
                        s['agent_pos'] = list(new_pos)
                        env.rule_events.append({
                            "event":"blocked_but_sword_contact",
                            "desc":"Stepped into monster tile while monster alive and has sword. Monster remains alive until attacked."
                        })
                        reward = -0.05
                    else:
                        s['agent_pos'] = list(new_pos)
                        reward = -1.0
                        done = True
                        env.rule_events.append({
                            "event":"death_by_monster",
                            "desc":"Agent moved into monster tile without sword => death."
                        })
                else:
                    s['agent_pos'] = list(new_pos)
            else:
                reward = -0.02
                env.rule_events.append({
                    "event":"bumped_wall",
                    "desc":"Agent tried to move out of bounds."
                })

        elif ax == 4:  # pick
            picked = False
            if key_pos is not None and agent_pos == key_pos:
                s['has_key'] = 1
                s['key_pos'] = None
                picked = True
                env.rule_events.append({"event":"picked_key","desc":"Agent picked up the key at its tile."})
                reward = 0.1
            if sword_pos is not None and agent_pos == sword_pos:
                s['has_sword'] = 1
                s['sword_pos'] = None
                picked = True
                env.rule_events.append({"event":"picked_sword","desc":"Agent picked up the sword at its tile."})
                reward = 0.1 if reward < 0.1 else reward
            if not picked:
                reward = -0.02
                env.rule_events.append({"event":"pick_nothing","desc":"Agent picked but nothing present."})

        elif ax == 5:  # attack
            if s['monster_alive']:
                axp = agent_pos
                mx,my = monster_pos
                if abs(axp[0]-mx) + abs(axp[1]-my) == 1 and s['has_sword']:
                    s['monster_alive'] = False
                    env.rule_events.append({"event":"killed_monster","desc":"Agent attacked adjacent monster while having sword -> monster dies."})
                    reward = 0.5
                else:
                    reward = -0.05
                    env.rule_events.append({"event":"attack_failed","desc":"Attack had no effect (monster not adjacent or no sword)."})
            else:
                reward = -0.02
                env.rule_events.append({"event":"attack_no_monster","desc":"Attack when monster already dead."})

        elif ax == 6:  # use (use key on lock)
            if agent_pos == lock_pos and s['has_key']:
                reward = 1.0
                done = True
                env.rule_events.append({"event":"used_key_on_lock","desc":"Agent used key on lock -> level cleared."})
            else:
                reward = -0.02
                env.rule_events.append({"event":"use_failed","desc":"Use failed (not at lock or no key)."})

        else:
            reward = -0.05
            env.rule_events.append({"event":"invalid_action","desc":"Agent took invalid action index."})

        return s, reward, done, info

# -----------------------
# Environment with ASCII renderer
# -----------------------
class KeyLockEnv:
    def __init__(self, width=8, height=8, law=None, seed=0):
        self.width = width
        self.height = height
        self.law = law if law is not None else SimpleAdventureLaw()
        self.rng = random.Random(seed)
        self.action_meanings = {0:'up',1:'down',2:'left',3:'right',4:'pick',5:'attack',6:'use'}
        self.reset()

    def reset(self, randomize=True):
        # place items in distinct cells to avoid trivial overlap
        all_cells = [(x,y) for x in range(self.width) for y in range(self.height)]
        self.rng.shuffle(all_cells)
        agent_pos = all_cells.pop()
        key_pos = all_cells.pop()
        sword_pos = all_cells.pop()
        lock_pos = all_cells.pop()
        monster_pos = all_cells.pop()
        self.state = {
            'agent_pos': list(agent_pos),
            'key_pos': list(key_pos),
            'sword_pos': list(sword_pos),
            'lock_pos': list(lock_pos),
            'monster_pos': list(monster_pos),
            'monster_alive': True,
            'has_key': 0,
            'has_sword': 0
        }
        self.rule_events = []
        self.trajectory = []
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        s = self.state
        tup = (
            s['agent_pos'][0], s['agent_pos'][1],
            None if s['key_pos'] is None else s['key_pos'][0], None if s['key_pos'] is None else s['key_pos'][1],
            None if s['sword_pos'] is None else s['sword_pos'][0], None if s['sword_pos'] is None else s['sword_pos'][1],
            s['lock_pos'][0], s['lock_pos'][1],
            int(s['monster_alive']), s['monster_pos'][0], s['monster_pos'][1],
            int(s['has_key']), int(s['has_sword'])
        )
        return tup

    def step(self, action):
        self.steps += 1
        next_state, reward, done, info = self.law.apply(self.state, action, self)
        # normalize positions to lists
        for k in ['agent_pos','key_pos','sword_pos','lock_pos','monster_pos']:
            if k in next_state and next_state[k] is not None and not isinstance(next_state[k], list):
                next_state[k] = list(next_state[k])
        self.state = next_state
        desc = self._describe_step(action, reward, done)
        self.trajectory.append(desc)
        if self.steps >= 500:  # larger timeout with larger grid
            done = True
            self.rule_events.append({"event":"timeout","desc":"Episode timed out after max steps."})
        return self._get_obs(), reward, done, {"rule_events": list(self.rule_events)}

    def _describe_step(self, action, reward, done):
        s = self.state
        agent = tuple(s['agent_pos'])
        inv = []
        if s['has_key']: inv.append('key')
        if s['has_sword']: inv.append('sword')
        inv = ", ".join(inv) if inv else "none"
        return {
            "step": self.steps,
            "agent_pos": agent,
            "inventory": inv,
            "action": action,
            "action_meaning": self.action_meanings.get(action, str(action)),
            "reward": float(reward),
            "monster_alive": bool(s['monster_alive']),
            "key_pos": None if s['key_pos'] is None else tuple(s['key_pos']),
            "sword_pos": None if s['sword_pos'] is None else tuple(s['sword_pos']),
            "lock_pos": tuple(s['lock_pos']),
            "done": bool(done)
        }

    def ascii_snapshot(self):
        s = self.state
        # create blank grid (string cells 2-chars wide)
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]
        # lock
        lx,ly = s['lock_pos']; grid[ly][lx] = " L"
        # monster (alive vs dead)
        mx,my = s['monster_pos']
        grid[my][mx] = " M" if s['monster_alive'] else " m"
        # items (key and sword)
        if s['key_pos'] is not None:
            kx,ky = s['key_pos']; grid[ky][kx] = " K"
        if s['sword_pos'] is not None:
            sx,sy = s['sword_pos']; grid[sy][sx] = " S"
        # agent (draw last)
        ax,ay = s['agent_pos']
        inv = ""
        if s['has_key']: inv += "k"
        if s['has_sword']: inv += "s"
        rep = "A" + (inv if inv else " ")
        rep = f"{rep[:2]:>2}"
        grid[ay][ax] = rep
        # build text (y increasing downward)
        lines = []
        for y in range(self.height):
            lines.append("".join(grid[y]))
        meta = f"step={self.steps} | agent={tuple(s['agent_pos'])} | inv={'key' if s['has_key'] else 'none'}/{ 'sword' if s['has_sword'] else 'none'} | monster_alive={s['monster_alive']}"
        return "\n".join(lines) + "\n" + meta

# -----------------------
# Dyna-Q with intrinsic bonus for new rule events
# -----------------------
class DynaQAgent:
    def __init__(self, actions, alpha=0.4, gamma=0.98, epsilon_start=1.0, epsilon_end=0.05, planning_steps=20, seed=0):
        self.actions = list(range(actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.planning_steps = planning_steps
        self.q = defaultdict(lambda: np.zeros(len(self.actions), dtype=float))
        self.model = {}  # (s,a) -> (r, s2)
        self.rng = random.Random(seed)

    def choose_action(self, s, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.choice(self.actions)
        qs = self.q[s]
        maxv = qs.max()
        candidates = [i for i,v in enumerate(qs) if v == maxv]
        return self.rng.choice(candidates)

    def learn(self, s, a, r, s2):
        q_sa = self.q[s][a]
        q_next = 0 if s2 is None else self.q[s2].max()
        self.q[s][a] += self.alpha * (r + self.gamma * q_next - q_sa)
        self.model[(s,a)] = (r, s2)

    def planning(self):
        if not self.model:
            return
        keys = list(self.model.keys())
        for _ in range(self.planning_steps):
            (s,a) = self.rng.choice(keys)
            r, s2 = self.model[(s,a)]
            q_sa = self.q[s][a]
            q_next = 0 if s2 is None else self.q[s2].max()
            self.q[s][a] += self.alpha * (r + self.gamma * q_next - q_sa)

    def _epsilon_for_episode(self, ep_idx, total_eps):
        # linear decay scheduled epsilon
        if total_eps <= 1:
            return self.epsilon_end
        frac = min(1.0, (ep_idx) / float(total_eps - 1))
        return self.epsilon_start * (1-frac) + self.epsilon_end * frac

    def train(self, env, episodes=500, max_steps_per_episode=500, out_dir=".", intrinsic_bonus=0.5, verbose=False):
        results = {"episode_rewards":[], "successes":0, "steps":[]}
        trajectories_file = os.path.join(out_dir, "trajectories.jsonl")
        rules_file = os.path.join(out_dir, "rule_archive.jsonl")
        episodes_dir = os.path.join(out_dir, "episodes")
        os.makedirs(episodes_dir, exist_ok=True)
        seen_rules_global = OrderedDict()   # store unique rule desc strings (archive)
        # clear old logs
        for p in (trajectories_file, rules_file):
            try: os.remove(p)
            except FileNotFoundError: pass

        for ep in range(1, episodes+1):
            s_tup = env.reset(randomize=True)
            s = s_tup
            total_r = 0.0
            done = False
            steps = 0
            env.rule_events = []
            env.trajectory = []
            # per-episode new rules tracker (so we can give bonus only when truly new globally)
            # NOTE: we give intrinsic bonus when a rule event description is not in seen_rules_global
            ep_file = os.path.join(episodes_dir, f"episode_{ep:04d}.txt")
            # compute epsilon for this episode
            epsilon = self._epsilon_for_episode(ep-1, episodes)
            # write initial snapshot and then step-by-step append
            with open(ep_file, "w") as ef:
                ef.write(f"Episode {ep}\n")
                ef.write("="*40 + "\n")
                ef.write("Initial:\n")
                ef.write(env.ascii_snapshot() + "\n\n")

                while not done and steps < max_steps_per_episode:
                    a = self.choose_action(s, epsilon)
                    s2, r, done, info = env.step(a)

                    # intrinsic bonus: check env.rule_events against seen_rules_global
                    intrinsic_added = 0.0
                    if env.rule_events:
                        # consider all events present so far; treat as new those not yet in global archive
                        for ev in env.rule_events:
                            desc = ev.get("desc", "")
                            if desc not in seen_rules_global:
                                seen_rules_global[desc] = ev
                                intrinsic_added += intrinsic_bonus
                                # record immediately in rules_file as new global rule found
                                with open(rules_file, "a") as rf:
                                    rf.write(json.dumps(ev) + "\n")

                    # add intrinsic bonus to the scalar reward used for learning & totals
                    r_with_intrinsic = r + intrinsic_added
                    total_r += r_with_intrinsic
                    steps += 1

                    # record ascii frame + rule events into episode log
                    ef.write(f"Step {steps} | action={env.action_meanings.get(a)} | base_reward={r:.3f} | intrinsic={intrinsic_added:.3f}\n")
                    ef.write(env.ascii_snapshot() + "\n")
                    if env.rule_events:
                        for e in env.rule_events:
                            ef.write(f"RULE_EVENT: {e.get('event')} - {e.get('desc')}\n")
                    ef.write("-"*40 + "\n")

                    # training and planning use the augmented reward r_with_intrinsic
                    self.learn(s, a, r_with_intrinsic, s2 if not done else None)
                    self.planning()
                    s = s2

            # episode end bookkeeping
            results["episode_rewards"].append(total_r)
            results["steps"].append(steps)
            if done and any(ev.get("event")=="used_key_on_lock" for ev in env.rule_events):
                results["successes"] += 1

            traj_record = {"episode":ep, "total_reward":total_r, "steps":steps, "trajectory":env.trajectory, "rule_events":env.rule_events}
            with open(trajectories_file, "a") as f:
                f.write(json.dumps(traj_record) + "\n")

            if verbose:
                print(f"[Ep {ep:04d}] eps={epsilon:.3f} reward={total_r:.3f} steps={steps} successes={results['successes']}")

            if ep % max(1, episodes//10) == 0 or ep <= 5:
                avg_reward = np.mean(results["episode_rewards"][-min(50, len(results["episode_rewards"])) :])
                print(f"Episode {ep:4d}/{episodes} | avg_reward(last50) {avg_reward:.3f} | successes {results['successes']}/{ep} | eps {epsilon:.3f}")

        # finally ensure rule_archive has all seen rules (some were written during training already)
        print("Training finished. Successes:", results["successes"], "Episodes:", episodes)
        return results, trajectories_file, rules_file, episodes_dir

# -----------------------
# Runner & CLI
# -----------------------
def main(args):
    law = SimpleAdventureLaw()
    env = KeyLockEnv(width=args.width, height=args.height, law=law, seed=args.seed)
    agent = DynaQAgent(actions=7, alpha=args.alpha, gamma=args.gamma,
                       epsilon_start=args.eps_start, epsilon_end=args.eps_end,
                       planning_steps=args.planning, seed=args.seed)
    t0 = time.time()
    results, traj_file, rules_file, episodes_dir = agent.train(
        env,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        out_dir=args.out_dir,
        intrinsic_bonus=args.intrinsic_bonus,
        verbose=args.verbose
    )
    t1 = time.time()
    print("Saved:", traj_file)
    print("Saved:", rules_file)
    print("Per-episode frames saved in:", episodes_dir)
    print("Time:", t1-t0)
    print("Average reward (last 50):", np.mean(results["episode_rewards"][-50:]))
    print("Success count:", results["successes"], "/", args.episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--width", type=int, default=8, help="Grid width (bigger -> more exploration)")
    parser.add_argument("--height", type=int, default=8, help="Grid height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.4, help="Q-learning alpha")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount gamma")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon (exploration)")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final epsilon (exploitation)")
    parser.add_argument("--planning", type=int, default=20, help="Dyna planning steps per real step")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--intrinsic_bonus", type=float, default=0.6, help="Intrinsic reward for each NEW rule-event discovered")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory for logs")
    parser.add_argument("--verbose", action="store_true", help="print per-episode concise logs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
