#!/usr/bin/env python3
"""
FIXED Dyna-Q Training with Law Discovery Capabilities
Senior Research Scientist Review - All Critical Issues Addressed
"""
from __future__ import annotations

import json
import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import deque, defaultdict
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except:
    HAS_PLOTTING = False

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger("dyna_fixed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# CONFIGURATION - FIXED
# ============================================================================

CONFIG = {
    'width': 8,
    'height': 8,
    'success_reward': 10.0,
    'death_penalty': -10.0,
    
    # Training
    'episodes': 2000,
    'eval_episodes': 50,
    'eval_interval': 200,  # NEW: Periodic evaluation
    'max_steps': 100,
    'mode': 'fixed',
    'num_laws': 1,
    
    # Agent - FIXED parameters
    'alpha': 5e-4,  # Lower for stability
    'gamma': 0.99,
    'eps_start': 1.0,
    'eps_end': 0.05,  # Lower final epsilon
    'epsilon_decay_episodes': 1500,
    'planning': 10,  # More conservative
    'planning_freq': 5,  # Less frequent but more stable
    
    # Network
    'batch_size': 64,  # Smaller for stability
    'buffer_capacity': 50000,
    'target_update_freq': 500,  # Less frequent hard updates
    
    # ICM - FIXED
    'use_icm': True,
    'icm_weight': 0.1,  # Much lower to prevent explosion
    'icm_reward_clip': 0.5,  # NEW: Clip intrinsic rewards
    'icm_beta': 0.2,  # NEW: Forward model weight
    
    # Optimizations
    'use_amp': True,
    'gradient_clip': 0.5,  # Tighter clipping
    'polyak_tau': 0.001,  # Much slower for stability
    'warmup_steps': 2000,  # Longer warmup
    'planning_warmup': 5000,  # NEW: Separate planning warmup
    
    # NEW: Law Discovery
    'save_trajectories': True,
    'save_embeddings': True,
    'save_high_error_transitions': True,
    'error_threshold': 0.5,
    'max_embeddings_buffer': 10000,  # NEW: Prevent memory leak
    'max_trajectories_buffer': 500,  # NEW: Prevent memory leak
    'log_interval': 100,
    
    # NEW: Early stopping
    'early_stop_threshold': 0.9,  # Stop if success rate > 90%
    'early_stop_patience': 10,  # Number of eval intervals
    
    'seed': 42,
    'out_dir': './dyna_output_fixed',
    'verbose': True,
}

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

args = Config(CONFIG)
os.makedirs(args.out_dir, exist_ok=True)

# ============================================================================
# UTILITIES - FIXED
# ============================================================================

def _pos_tuple(p):
    return None if p is None else (int(p[0]), int(p[1]))

GRID_NORM = 1.0 / 8.0

def normalize_state(state_tuple, grid_size: int = 8):
    """FIXED: Proper normalization for all state components."""
    state_list = []
    for i, val in enumerate(state_tuple):
        if val is None:
            state_list.append(-1.0)
        else:
            if i < 11:  # All position values
                # Normalize positions to [0, 1]
                if val == -1:  # Dead monster position
                    state_list.append(-1.0)
                else:
                    state_list.append(float(val) * GRID_NORM)
            else:
                # Binary flags (has_key, has_sword)
                state_list.append(float(val))
    return torch.tensor(state_list, dtype=torch.float32, device='cpu')

def state_to_symbolic(obs: Tuple) -> str:
    """Convert observation to symbolic representation for VLM."""
    agent_x, agent_y = obs[0], obs[1]
    key_x, key_y = obs[2], obs[3]
    sword_x, sword_y = obs[4], obs[5]
    lock_x, lock_y = obs[6], obs[7]
    monster_alive, monster_x, monster_y = obs[8], obs[9], obs[10]
    has_key, has_sword = obs[11], obs[12]
    
    parts = [f"agent({agent_x},{agent_y})"]
    
    if has_key:
        parts.append("has_key")
    else:
        if key_x is not None:
            parts.append(f"key({key_x},{key_y})")
    
    if has_sword:
        parts.append("has_sword")
    else:
        if sword_x is not None:
            parts.append(f"sword({sword_x},{sword_y})")
    
    parts.append(f"lock({lock_x},{lock_y})")
    
    if monster_alive:
        parts.append(f"monster({monster_x},{monster_y},alive)")
    
    return " ".join(parts)

class StateEncoder:
    """FIXED: Added proper cache management."""
    __slots__ = ['_map', '_rev', 'grid_size', '_cache', '_max_cache_size']
    
    def __init__(self, grid_size: int = 8, max_cache_size: int = 100000):
        self._map: Dict[Tuple, int] = {}
        self._rev: List[Tuple] = []
        self.grid_size = grid_size
        self._cache = {}
        self._max_cache_size = max_cache_size
        
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
        return self._rev[idx]
    
    def obs_to_features(self, obs: Tuple) -> torch.Tensor:
        if obs in self._cache:
            return self._cache[obs].clone()  # FIXED: Clone to prevent modification
        
        features = normalize_state(obs, self.grid_size)
        
        # FIXED: Cache size management
        if len(self._cache) < self._max_cache_size:
            self._cache[obs] = features.clone()
        elif len(self._cache) >= self._max_cache_size * 1.5:
            # Clear half the cache when it grows too large
            keys_to_remove = list(self._cache.keys())[:len(self._cache) // 2]
            for k in keys_to_remove:
                del self._cache[k]
        
        return features
    
    def __len__(self):
        return len(self._rev)

# ============================================================================
# REPLAY BUFFER - FIXED
# ============================================================================

class PrioritizedReplayBuffer:
    """FIXED: Better priority management and sampling."""
    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        self.min_priority = 0.01  # NEW: Prevent zero priorities
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if batch_size == 0:
            return None
            
        priorities = np.array(self.priorities, dtype=np.float32)
        priorities = np.maximum(priorities, self.min_priority)  # FIXED: Avoid zero
        probs = np.power(priorities, self.alpha)
        probs /= probs.sum()
        
        # FIXED: Handle small buffer sizes
        replace = len(self.buffer) < batch_size
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=replace)
        
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = np.power(len(self.buffer) * probs[indices], -beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # FIXED: Handle None states properly
        valid_next_states = []
        for ns in next_states:
            if ns is not None:
                valid_next_states.append(ns)
            else:
                valid_next_states.append(torch.zeros_like(states[0]))
        
        states_batch = torch.stack(states).to(device, non_blocking=True)
        actions_batch = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_batch = torch.stack(valid_next_states).to(device, non_blocking=True)
        dones_batch = torch.tensor(dones, dtype=torch.float32, device=device)
        weights_batch = torch.tensor(weights, dtype=torch.float32, device=device)
        
        self.frame += 1
        
        return (indices, states_batch, actions_batch, rewards_batch, 
                next_states_batch, dones_batch, weights_batch)
    
    def update_priorities(self, indices, priorities):
        priorities = np.clip(priorities, self.min_priority, None)  # FIXED: Clip
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):  # FIXED: Bounds check
                self.priorities[idx] = float(priority)
                self.max_priority = max(self.max_priority, float(priority))
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# NETWORKS - IMPROVED
# ============================================================================

class QNetwork(nn.Module):
    """IMPROVED: Better architecture with residual connections."""
    def __init__(self, state_dim=13, action_dim=7, hidden_dim=256):
        super().__init__()
        
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Residual block
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        # Dueling architecture
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        # Input processing
        x = F.relu(self.norm1(self.input_layer(x)))
        
        # Residual block 1
        residual = x
        x = F.relu(self.norm2(self.hidden1(x)))
        x = self.dropout1(x)
        x = x + residual  # Residual connection
        
        # Residual block 2
        residual = x
        x = F.relu(self.norm3(self.hidden2(x)))
        x = self.dropout2(x)
        x = x + residual  # Residual connection
        
        # Dueling architecture
        value = self.value(x)
        advantage = self.advantage(x)
        
        # Combine value and advantage
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q


class WorldModel(nn.Module):
    """IMPROVED: Better world model with uncertainty estimation."""
    def __init__(self, state_dim=13, action_dim=7, hidden_dim=256, latent_dim=128):
        super().__init__()
        
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + action_dim
        
        # Encoder - Larger latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Prediction heads
        self.next_state = nn.Linear(hidden_dim, state_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.done = nn.Linear(hidden_dim, 1)
        
        # NEW: Uncertainty head
        self.uncertainty = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def encode(self, state, action):
        """Expose encoding for law discovery."""
        action_onehot = F.one_hot(action, self.action_dim).float()
        x = torch.cat([state, action_onehot], dim=-1)
        return self.encoder(x)
    
    def forward(self, state, action):
        latent = self.encode(state, action)
        features = self.decoder(latent)
        
        next_state = self.next_state(features)
        reward = self.reward(features).squeeze(-1)
        done_logits = self.done(features).squeeze(-1)
        uncertainty = torch.sigmoid(self.uncertainty(features)).squeeze(-1)
        
        return next_state, reward, done_logits, latent, uncertainty


class IntrinsicCuriosityModule(nn.Module):
    """FIXED: Better ICM with proper loss balancing."""
    def __init__(self, state_dim=13, action_dim=7, hidden_dim=128, feature_dim=64):
        super().__init__()
        
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Forward model (predict next state features from current features + action)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Inverse model (predict action from state features)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state, action, next_state):
        # Encode states
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Forward model
        action_onehot = F.one_hot(action, self.action_dim).float()
        pred_next_feat = self.forward_model(torch.cat([state_feat, action_onehot], dim=-1))
        
        # FIXED: Proper intrinsic reward calculation
        intrinsic_reward = 0.5 * F.mse_loss(
            pred_next_feat, next_state_feat.detach(), reduction='none'
        ).mean(dim=-1)
        
        # Inverse model
        pred_action = self.inverse_model(torch.cat([state_feat, next_state_feat], dim=-1))
        
        return intrinsic_reward, pred_action, pred_next_feat, next_state_feat.detach()

print("Creating networks...")
q_net = QNetwork().to(device)
world_model = WorldModel().to(device)
icm = IntrinsicCuriosityModule().to(device)

target_net = QNetwork().to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

print(f"  Q-Net: {sum(p.numel() for p in q_net.parameters()):,} params")
print(f"  World: {sum(p.numel() for p in world_model.parameters()):,} params")
print(f"  ICM: {sum(p.numel() for p in icm.parameters()):,} params")

# ============================================================================
# LAW DISCOVERY TRACKER - FIXED
# ============================================================================

class LawDiscoveryTracker:
    """FIXED: Proper memory management and data cleanup."""
    def __init__(self, save_dir: Path, max_trajectories=500, max_embeddings=10000):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_trajectories = max_trajectories
        self.max_embeddings = max_embeddings
        
        # Trajectory data with deque for automatic size management
        self.trajectories = deque(maxlen=max_trajectories)
        self.current_trajectory = []
        
        # High-error transitions
        self.high_error_transitions = deque(maxlen=1000)
        
        # Embeddings with automatic cleanup
        self.embeddings_buffer = deque(maxlen=max_embeddings)
        
        # Symbolic traces
        self.symbolic_traces = deque(maxlen=100)
        
    def start_episode(self):
        """Start tracking new episode."""
        self.current_trajectory = []
        
    def log_transition(self, obs, action, next_obs, reward, done, 
                      latent_embedding, prediction_error, info=None):
        """Log a single transition."""
        transition = {
            'state_symbolic': state_to_symbolic(obs),
            'action': int(action),
            'next_state_symbolic': state_to_symbolic(next_obs),
            'reward': float(reward),
            'done': bool(done),
            'prediction_error': float(prediction_error),
            'latent': latent_embedding.cpu().numpy().tolist() if latent_embedding is not None else None,
            'info': info or {}
        }
        
        self.current_trajectory.append(transition)
        
        # Save high-error transitions
        if prediction_error > args.error_threshold:
            self.high_error_transitions.append(transition)
        
        # Save embedding
        if latent_embedding is not None:
            self.embeddings_buffer.append({
                'state': state_to_symbolic(obs),
                'action': int(action),
                'embedding': latent_embedding.cpu().numpy()
            })
    
    def end_episode(self, episode_reward, success):
        """End episode and save trajectory."""
        if self.current_trajectory:
            self.trajectories.append({
                'transitions': self.current_trajectory,
                'total_reward': float(episode_reward),
                'success': bool(success),
                'length': len(self.current_trajectory)
            })
        self.current_trajectory = []
    
    def save_data(self, episode):
        """Periodically save collected data."""
        try:
            # Save high-error transitions
            if self.high_error_transitions:
                error_file = self.save_dir / f'high_error_transitions_ep{episode}.json'
                with open(error_file, 'w') as f:
                    json.dump(list(self.high_error_transitions), f, indent=2)
            
            # Save trajectories
            if self.trajectories:
                traj_file = self.save_dir / f'trajectories_ep{episode}.json'
                with open(traj_file, 'w') as f:
                    json.dump(list(self.trajectories), f, indent=2)
            
            # Save embeddings
            if self.embeddings_buffer:
                emb_file = self.save_dir / f'embeddings_ep{episode}.npy'
                embeddings_array = np.array([e['embedding'] for e in self.embeddings_buffer])
                np.save(emb_file, embeddings_array)
                
                # Save corresponding states
                states_file = self.save_dir / f'embedding_states_ep{episode}.json'
                with open(states_file, 'w') as f:
                    json.dump([{'state': e['state'], 'action': e['action']} 
                              for e in self.embeddings_buffer], f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving law discovery data: {e}")
    
    def get_summary(self):
        """Get summary statistics."""
        return {
            'total_trajectories': len(self.trajectories),
            'total_high_error': len(self.high_error_transitions),
            'total_embeddings': len(self.embeddings_buffer),
        }

# ============================================================================
# LAWS (unchanged)
# ============================================================================

class Law:
    name = "base-law"
    def apply(self, state, action, env):
        raise NotImplementedError

class SimpleAdventureLaw(Law):
    name = "simple"
    
    def apply(self, state, action, env):
        s = dict(state)
        reward = -0.01
        done = False
        info = {}

        agent_pos = tuple(s.get('agent_pos', (0, 0)))
        key_pos = _pos_tuple(s.get('key_pos'))
        sword_pos = _pos_tuple(s.get('sword_pos'))
        lock_pos = tuple(s.get('lock_pos', (0, 0)))
        monster_pos = _pos_tuple(s.get('monster_pos')) if s.get('monster_alive', False) else None

        if action in (0, 1, 2, 3):
            dx = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[action]
            new_pos = (agent_pos[0] + dx[0], agent_pos[1] + dx[1])
            
            if 0 <= new_pos[0] < env.width and 0 <= new_pos[1] < env.height:
                if s.get('monster_alive', False) and new_pos == monster_pos:
                    if s.get('has_sword'):
                        s['agent_pos'] = list(new_pos)
                        reward = -0.05
                    else:
                        s['agent_pos'] = list(new_pos)
                        reward = env.death_penalty
                        done = True
                else:
                    s['agent_pos'] = list(new_pos)
            else:
                reward = -0.05
                
        elif action == 4:  # pick
            picked = False
            if key_pos is not None and agent_pos == key_pos:
                s['has_key'] = 1
                s['key_pos'] = None
                picked = True
                reward = 1.0
            if sword_pos is not None and agent_pos == sword_pos:
                s['has_sword'] = 1
                s['sword_pos'] = None
                picked = True
                reward = 1.0 if reward < 1.0 else reward
            if not picked:
                reward = -0.05
                
        elif action == 5:  # attack
            if s.get('monster_alive'):
                axp = tuple(s['agent_pos'])
                mx, my = s['monster_pos']
                if abs(axp[0] - mx) + abs(axp[1] - my) == 1 and s.get('has_sword'):
                    s['monster_alive'] = False
                    reward = 2.0
                else:
                    reward = -0.05
            else:
                reward = -0.05
                
        elif action == 6:  # use
            if agent_pos == lock_pos and s.get('has_key'):
                reward = env.success_reward
                done = True
                info['event'] = 'success'
            else:
                reward = -0.05
        else:
            reward = -0.05
            
        return s, reward, done, info

ALL_LAW_CLASSES = [SimpleAdventureLaw]

# ============================================================================
# ENVIRONMENT (unchanged)
# ============================================================================

class KeyLockEnv:
    def __init__(self, width=8, height=8, law: Optional[Law]=None, seed: int=0, 
                 success_reward=10.0, death_penalty=-10.0):
        self.width = int(width)
        self.height = int(height)
        self.rng = random.Random(int(seed))
        self.action_meanings = {
            0: 'up', 1: 'down', 2: 'left', 3: 'right',
            4: 'pick', 5: 'attack', 6: 'use'
        }
        self.law = law if law is not None else SimpleAdventureLaw()
        self.success_reward = float(success_reward)
        self.death_penalty = float(death_penalty)
        self.state = None
        self.rule_events = []
        self.steps = 0
        self.reset(randomize=True)
    
    def set_law(self, law: Law):
        self.law = law
    
    def reset(self, randomize=True):
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        if randomize:
            self.rng.shuffle(cells)
        
        self.state = {
            'agent_pos': list(cells.pop()),
            'key_pos': list(cells.pop()),
            'sword_pos': list(cells.pop()),
            'lock_pos': list(cells.pop()),
            'monster_pos': list(cells.pop()),
            'monster_alive': True,
            'has_key': 0,
            'has_sword': 0
        }
        self.rule_events = []
        self.steps = 0
        return self._get_obs()
    
    def _get_obs(self):
        s = self.state
        key_x = s['key_pos'][0] if s['key_pos'] is not None else None
        key_y = s['key_pos'][1] if s['key_pos'] is not None else None
        sword_x = s['sword_pos'][0] if s['sword_pos'] is not None else None
        sword_y = s['sword_pos'][1] if s['sword_pos'] is not None else None
        
        monster_alive = int(bool(s.get('monster_alive', False)))
        monster_x = s['monster_pos'][0] if s.get('monster_pos') is not None else -1
        monster_y = s['monster_pos'][1] if s.get('monster_pos') is not None else -1
        
        return (
            s['agent_pos'][0], s['agent_pos'][1],
            key_x, key_y, sword_x, sword_y,
            s['lock_pos'][0], s['lock_pos'][1],
            monster_alive, monster_x, monster_y,
            int(bool(s.get('has_key', 0))),
            int(bool(s.get('has_sword', 0)))
        )
    
    def step(self, action):
        self.steps += 1
        self.rule_events = []
        
        next_state, reward, done, info = self.law.apply(self.state, action, self)
        
        for k in ('agent_pos', 'key_pos', 'sword_pos', 'lock_pos', 'monster_pos'):
            if k in next_state and next_state[k] is not None:
                if not isinstance(next_state[k], list):
                    next_state[k] = list(next_state[k])
        
        self.state = next_state
        
        if self.steps >= args.max_steps:
            done = True
        
        return self._get_obs(), reward, done, info

# ============================================================================
# AGENT - COMPLETELY FIXED
# ============================================================================

class NeuralDynaQAgent:
    """FIXED: All critical bugs addressed."""
    def __init__(self, q_network, target_network, world_model, icm, 
                 alpha=5e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_episodes=1500, planning_steps=10, batch_size=64,
                 buffer_capacity=50000, target_update_freq=500, use_icm=True,
                 icm_weight=0.1, icm_beta=0.2, seed=0, polyak_tau=0.001, 
                 gradient_clip=0.5, icm_reward_clip=0.5):
        
        self.q_network = q_network
        self.target_network = target_network
        self.world_model = world_model
        self.icm = icm
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.planning_steps = planning_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_icm = use_icm
        self.icm_weight = icm_weight
        self.icm_beta = icm_beta  # Forward model weight
        self.polyak_tau = polyak_tau
        self.gradient_clip = gradient_clip
        self.icm_reward_clip = icm_reward_clip
        
        # Optimizers
        self.q_optimizer = optim.AdamW(q_network.parameters(), lr=alpha, weight_decay=1e-5)
        self.model_optimizer = optim.AdamW(world_model.parameters(), lr=alpha, weight_decay=1e-5)
        if icm:
            self.icm_optimizer = optim.AdamW(icm.parameters(), lr=alpha, weight_decay=1e-5)
        
        # Buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        
        # Schedulers
        self.q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.q_optimizer, T_max=epsilon_decay_episodes, eta_min=alpha*0.1
        )
        self.model_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer, T_max=epsilon_decay_episodes, eta_min=alpha*0.1
        )
        
        # Stats with EMA
        self.update_count = 0
        self.q_losses = []
        self.model_losses = []
        self.icm_losses = []
        self.q_loss_ema = 0.0
        self.model_loss_ema = 0.0
        self.icm_loss_ema = 0.0
        self.ema_alpha = 0.05
        
    def choose_action(self, state_features, epsilon):
        if random.random() < epsilon:
            return random.randrange(7)
        
        with torch.no_grad():
            state_tensor = state_features.unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def learn(self):
        """FIXED: Proper learning step with all improvements."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        sample_result = self.replay_buffer.sample(self.batch_size)
        if sample_result is None:
            return None
            
        indices, states, actions, rewards, next_states, dones, weights = sample_result
        
        # Update Q-network
        q_loss, td_errors = self._update_q_network(states, actions, rewards, next_states, dones, weights)
        self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
        
        # Update world model (every step now, but with lower learning rate)
        model_loss = self._update_world_model(states, actions, rewards, next_states, dones)
        
        # Update ICM
        icm_loss = 0.0
        if self.icm:
            icm_loss = self._update_icm(states, actions, next_states)
        
        # Update EMA losses
        if self.q_loss_ema == 0.0:
            self.q_loss_ema = q_loss
            self.model_loss_ema = model_loss
            self.icm_loss_ema = icm_loss
        else:
            self.q_loss_ema = (1 - self.ema_alpha) * self.q_loss_ema + self.ema_alpha * q_loss
            self.model_loss_ema = (1 - self.ema_alpha) * self.model_loss_ema + self.ema_alpha * model_loss
            self.icm_loss_ema = (1 - self.ema_alpha) * self.icm_loss_ema + self.ema_alpha * icm_loss
        
        self.q_losses.append(q_loss)
        self.model_losses.append(model_loss)
        if self.icm:
            self.icm_losses.append(icm_loss)
        
        self.update_count += 1
        
        # FIXED: Separate hard and soft updates
        if self.update_count % self.target_update_freq == 0:
            self._hard_update_target()  # Periodic hard update
        else:
            self._soft_update_target()  # Continuous soft update
        
        return q_loss, model_loss, icm_loss
    
    def _hard_update_target(self):
        """FIXED: Hard update for stability."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _soft_update_target(self):
        """FIXED: Soft update between hard updates."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.polyak_tau * param.data + (1.0 - self.polyak_tau) * target_param.data)
    
    def _update_q_network(self, states, actions, rewards, next_states, dones, weights):
        """FIXED: Double DQN with proper error handling."""
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Double DQN: use online network to select action, target network to evaluate
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        td_errors = torch.abs(current_q - target_q)
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.q_optimizer.step()
        
        return loss.item(), td_errors.detach()
    
    def _update_world_model(self, states, actions, rewards, next_states, dones):
        """FIXED: Better loss balancing."""
        pred_next_states, pred_rewards, pred_done_logits, _, pred_uncertainty = self.world_model(states, actions)
        
        # State prediction loss
        state_loss = F.mse_loss(pred_next_states, next_states)
        
        # Reward prediction loss
        reward_loss = F.smooth_l1_loss(pred_rewards, rewards)
        
        # Done prediction loss
        done_loss = F.binary_cross_entropy_with_logits(pred_done_logits, dones)
        
        # FIXED: Balanced loss with uncertainty regularization
        total_loss = state_loss + reward_loss + done_loss + 0.01 * pred_uncertainty.mean()
        
        self.model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.gradient_clip)
        self.model_optimizer.step()
        
        return total_loss.item()
    
    def _update_icm(self, states, actions, next_states):
        """FIXED: Proper ICM update with balanced losses."""
        intrinsic_rewards, predicted_actions, pred_features, target_features = self.icm(states, actions, next_states)
        
        # Forward model loss (prediction error)
        forward_loss = F.mse_loss(pred_features, target_features)
        
        # Inverse model loss (action prediction)
        inverse_loss = F.cross_entropy(predicted_actions, actions)
        
        # FIXED: Balanced combination
        total_loss = self.icm_beta * forward_loss + (1 - self.icm_beta) * inverse_loss
        
        self.icm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), self.gradient_clip)
        self.icm_optimizer.step()
        
        return total_loss.item()
    
    def planning(self):
        """FIXED: Stable planning with proper gradient handling."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        planning_batch_size = min(32, self.batch_size // 2)  # Smaller batches
        
        for step in range(self.planning_steps):
            # Sample states from replay buffer
            sample_result = self.replay_buffer.sample(planning_batch_size)
            if sample_result is None:
                continue
                
            _, states, _, _, _, _, _ = sample_result
            
            with torch.no_grad():
                # FIXED: More exploration in planning
                if random.random() < 0.2:  # 20% random actions
                    actions = torch.randint(0, 7, (states.size(0),), device=device)
                else:
                    actions = self.q_network(states).argmax(1)
            
            # Predict next states using world model
            pred_next_states, pred_rewards, pred_done_logits, _, pred_uncertainty = self.world_model(states, actions)
            pred_dones = torch.sigmoid(pred_done_logits)
            
            # FIXED: Weight by uncertainty (trust certain predictions more)
            with torch.no_grad():
                confidence = 1.0 - pred_uncertainty
                confidence = torch.clamp(confidence, 0.1, 1.0)  # Don't completely ignore
            
            # Update Q-network with imagined transitions
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                # Use target network for stability
                next_q = self.target_network(pred_next_states).max(1)[0]
                target_q = pred_rewards + self.gamma * next_q * (1 - pred_dones)
            
            # FIXED: Weighted loss by prediction confidence
            loss = (confidence * F.smooth_l1_loss(current_q, target_q.detach(), reduction='none')).mean()
            
            # FIXED: Separate gradient step for each planning iteration
            self.q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip * 0.5)
            self.q_optimizer.step()
    
    def get_epsilon(self, episode):
        """FIXED: Proper epsilon calculation."""
        if episode >= self.epsilon_decay_episodes:
            return self.epsilon_end
        
        # Linear decay
        fraction = min(1.0, episode / self.epsilon_decay_episodes)
        epsilon = self.epsilon_start * (1 - fraction) + self.epsilon_end * fraction
        
        return max(self.epsilon_end, epsilon)  # FIXED: Ensure minimum
    
    def get_intrinsic_reward(self, state, action, next_state):
        """FIXED: Clipped intrinsic rewards."""
        if self.icm is None:
            return 0.0
        
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(device)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)
            next_state_tensor = next_state.unsqueeze(0).to(device)
            
            intrinsic_reward, _, _, _ = self.icm(state_tensor, action_tensor, next_state_tensor)
            
            # FIXED: Clip to prevent explosion
            clipped_reward = torch.clamp(intrinsic_reward, 0, self.icm_reward_clip).item()
            
            return clipped_reward * self.icm_weight
    
    def get_prediction_error(self, state, action, next_state, reward):
        """Get world model prediction error for law discovery."""
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(device)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)
            next_state_tensor = next_state.unsqueeze(0).to(device)
            
            pred_next, pred_reward, _, _, _ = self.world_model(state_tensor, action_tensor)
            
            state_error = F.mse_loss(pred_next, next_state_tensor).item()
            reward_error = abs(pred_reward.item() - reward)
            
            return state_error + reward_error
    
    def get_latent_embedding(self, state, action):
        """Get latent embedding for law discovery."""
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(device)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)
            
            latent = self.world_model.encode(state_tensor, action_tensor)
            return latent.squeeze(0)

# ============================================================================
# TRAINING - IMPROVED
# ============================================================================

def train():
    print("\n" + "="*80)
    print("FIXED DYNA TRAINING WITH LAW DISCOVERY")
    print("="*80)
    
    # Setup
    law_subset = ALL_LAW_CLASSES[:args.num_laws]
    print(f"Laws: {[cls.__name__ for cls in law_subset]}")
    
    env = KeyLockEnv(
        width=args.width,
        height=args.height,
        seed=args.seed,
        success_reward=args.success_reward,
        death_penalty=args.death_penalty
    )
    encoder = StateEncoder(grid_size=args.width)
    law_instances = [cls() for cls in law_subset]
    
    # Agent
    agent = NeuralDynaQAgent(
        q_network=q_net,
        target_network=target_net,
        world_model=world_model,
        icm=icm,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        planning_steps=args.planning,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
        use_icm=args.use_icm,
        icm_weight=args.icm_weight,
        icm_beta=args.icm_beta,
        seed=args.seed,
        polyak_tau=args.polyak_tau,
        gradient_clip=args.gradient_clip,
        icm_reward_clip=args.icm_reward_clip
    )
    
    # Law discovery tracker
    tracker = LawDiscoveryTracker(
        Path(args.out_dir) / "law_discovery",
        max_trajectories=args.max_trajectories_buffer,
        max_embeddings=args.max_embeddings_buffer
    )
    
    # Results
    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "successes": 0,
        "q_losses": [],
        "model_losses": [],
        "icm_losses": [],
        "law_sequence": [],
        "eval_results": []
    }
    
    # Early stopping
    best_success_rate = 0.0
    patience_counter = 0
    
    print(f"\nDevice: {device}")
    print(f"Planning steps: {args.planning}")
    print(f"Planning warmup: {args.planning_warmup}")
    print(f"ICM weight: {args.icm_weight}")
    print(f"Batch size: {args.batch_size}")
    print("="*80 + "\n")
    
    print("Training...")
    
    for ep in range(1, args.episodes + 1):
        # Choose law
        law_choice = law_instances[0]  # Fixed mode
        env.set_law(law_choice)
        results["law_sequence"].append(law_choice.name)
        
        # Reset
        obs = env.reset(randomize=True)
        state_features = encoder.obs_to_features(obs)
        
        total_reward = 0.0
        episode_length = 0
        done = False
        epsilon = agent.get_epsilon(ep - 1)
        
        # Start tracking
        if args.save_trajectories:
            tracker.start_episode()
        
        # Episode loop
        while not done and episode_length < args.max_steps:
            # Choose action
            action = agent.choose_action(state_features, epsilon)
            
            # Step
            next_obs, reward, done, info = env.step(action)
            next_state_features = encoder.obs_to_features(next_obs)
            
            # Get intrinsic reward (after warmup)
            intrinsic_reward = 0.0
            if agent.icm and len(agent.replay_buffer) > args.warmup_steps:
                intrinsic_reward = agent.get_intrinsic_reward(state_features, action, next_state_features)
            
            shaped_reward = reward + intrinsic_reward
            
            # Store transition
            agent.store_transition(
                state_features,
                action,
                shaped_reward,
                next_state_features if not done else None,
                done
            )
            
            # Learn (after warmup)
            if len(agent.replay_buffer) >= args.warmup_steps:
                losses = agent.learn()
            
            # Planning (after planning warmup)
            if (len(agent.replay_buffer) >= args.planning_warmup and 
                ep % args.planning_freq == 0 and 
                episode_length % 5 == 0):  # Less frequent planning within episode
                agent.planning()
            
            # Track for law discovery
            if args.save_trajectories and len(agent.replay_buffer) >= args.batch_size:
                prediction_error = agent.get_prediction_error(
                    state_features, action, next_state_features, reward
                )
                latent = agent.get_latent_embedding(state_features, action)
                
                tracker.log_transition(
                    obs, action, next_obs, reward, done,
                    latent, prediction_error, info
                )
            
            # Update
            obs = next_obs
            state_features = next_state_features
            total_reward += reward
            episode_length += 1
        
        # End episode
        results["episode_rewards"].append(total_reward)
        results["episode_lengths"].append(episode_length)
        
        success = info.get('event') == 'success' or total_reward > 5
        if success:
            results["successes"] += 1
        
        if args.save_trajectories:
            tracker.end_episode(total_reward, success)
        
        # Log losses
        if agent.q_losses:
            results["q_losses"].append(agent.q_loss_ema)
        if agent.model_losses:
            results["model_losses"].append(agent.model_loss_ema)
        if agent.icm_losses:
            results["icm_losses"].append(agent.icm_loss_ema)
        
        # Update schedulers
        if ep > args.warmup_steps:
            agent.q_scheduler.step()
            agent.model_scheduler.step()
        
        # Periodic evaluation
        if ep % args.eval_interval == 0:
            eval_results = evaluate_agent(agent, env, encoder, law_instances, args.eval_episodes, args.max_steps)
            results["eval_results"].append({
                'episode': ep,
                'results': eval_results
            })
            
            current_success_rate = eval_results[law_instances[0].name]['success_rate']
            
            print(f"\n{'='*60}")
            print(f"Evaluation at episode {ep}:")
            for law_name, metrics in eval_results.items():
                print(f"  {law_name}: Success={metrics['success_rate']:.1%}, Reward={metrics['avg_reward']:.2f}")
            print(f"{'='*60}\n")
            
            # Early stopping check
            if current_success_rate > best_success_rate:
                best_success_rate = current_success_rate
                patience_counter = 0
                
                # Save best model
                best_model_file = Path(args.out_dir) / "model_best.pt"
                torch.save({
                    'episode': ep,
                    'q_network': agent.q_network.state_dict(),
                    'target_network': agent.target_network.state_dict(),
                    'world_model': agent.world_model.state_dict(),
                    'icm': agent.icm.state_dict() if agent.icm else None,
                    'success_rate': best_success_rate
                }, best_model_file)
                print(f"  💾 Saved best model (success rate: {best_success_rate:.1%})")
            else:
                patience_counter += 1
            
            # Check for early stopping
            if best_success_rate >= args.early_stop_threshold and patience_counter >= args.early_stop_patience:
                print(f"\n🎉 Early stopping! Achieved {best_success_rate:.1%} success rate.")
                break
        
        # Logging
        if args.verbose and (ep % args.log_interval == 0 or ep <= 5):
            recent_rewards = results["episode_rewards"][-min(50, len(results["episode_rewards"])):]
            avg_reward = np.mean(recent_rewards)
            success_rate = results["successes"] / ep
            q_loss = agent.q_loss_ema
            model_loss = agent.model_loss_ema
            
            print(f"Ep {ep:4d}/{args.episodes} | "
                  f"R: {avg_reward:7.2f} | "
                  f"Success: {success_rate:5.1%} | "
                  f"ε: {epsilon:.3f} | "
                  f"QL: {q_loss:.4f} | "
                  f"ML: {model_loss:.4f} | "
                  f"Buf: {len(agent.replay_buffer):5d}")
        
        # Save law discovery data periodically
        if args.save_trajectories and ep % 500 == 0:
            tracker.save_data(ep)
            if ep % 500 == 0:
                print(f"  💾 Saved law discovery data: {tracker.get_summary()}")
        
        # Memory management
        if ep % 200 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    print("\n✓ Training complete!")
    
    # Final save
    if args.save_trajectories:
        tracker.save_data(args.episodes)
        print(f"\n📊 Final law discovery summary: {tracker.get_summary()}")
    
    return agent, env, encoder, law_instances, results, tracker

# ============================================================================
# EVALUATION (unchanged)
# ============================================================================

def evaluate_agent(agent, env, encoder, law_instances, num_episodes, max_steps):
    """Evaluate agent on all law types."""
    eval_results = {}
    
    agent.q_network.eval()
    
    with torch.no_grad():
        for law in law_instances:
            env.set_law(law)
            rewards = []
            successes = 0
            lengths = []
            
            for _ in range(num_episodes):
                obs = env.reset(randomize=True)
                state_features = encoder.obs_to_features(obs)
                total_reward = 0.0
                episode_length = 0
                done = False
                
                while not done and episode_length < max_steps:
                    action = agent.choose_action(state_features, epsilon=0.0)
                    next_obs, reward, done, info = env.step(action)
                    next_state_features = encoder.obs_to_features(next_obs)
                    
                    state_features = next_state_features
                    total_reward += reward
                    episode_length += 1
                
                rewards.append(total_reward)
                lengths.append(episode_length)
                
                if info.get('event') == 'success' or total_reward > 5:
                    successes += 1
            
            eval_results[law.name] = {
                "avg_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "success_rate": float(successes / num_episodes),
                "avg_length": float(np.mean(lengths))
            }
    
    agent.q_network.train()
    
    return eval_results

# ============================================================================
# PLOTTING - IMPROVED
# ============================================================================

def plot_training_results(results, out_path, run_id):
    """Plot training curves with improvements."""
    if not HAS_PLOTTING:
        print("⚠️  Plotting not available")
        return
    
    plots_dir = out_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
    
    # Rewards
    ax = axes[0, 0]
    rewards = np.array(results["episode_rewards"])
    ax.plot(rewards, alpha=0.2, color='blue', linewidth=0.5, label='Raw')
    if len(rewards) >= 10:
        window = min(50, max(10, len(rewards) // 10))
        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(rewards)), ma, color='red', linewidth=2, label=f'MA{window}')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-loss
    ax = axes[0, 1]
    if results["q_losses"]:
        ax.plot(results["q_losses"], color='green', linewidth=1, alpha=0.7)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("Q-Network Loss (EMA)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Model loss
    ax = axes[0, 2]
    if results["model_losses"]:
        ax.plot(results["model_losses"], color='orange', linewidth=1, alpha=0.7)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("World Model Loss (EMA)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Success rate
    ax = axes[1, 0]
    window = 50
    rewards_array = np.array(results["episode_rewards"])
    success_indicators = (rewards_array > 5).astype(int)
    if len(success_indicators) >= window:
        success_rate = np.convolve(success_indicators, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(success_indicators)), success_rate, color='purple', linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Success Rate (MA{window})")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target (90%)')
        ax.legend()
    
    # Episode lengths
    ax = axes[1, 1]
    if results["episode_lengths"]:
        lengths = np.array(results["episode_lengths"])
        ax.plot(lengths, alpha=0.2, color='teal', linewidth=0.5)
        if len(lengths) >= 10:
            window = min(50, max(10, len(lengths) // 10))
            ma = np.convolve(lengths, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(lengths)), ma, color='darkblue', linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Episode Length")
        ax.grid(True, alpha=0.3)
    
    # ICM loss
    ax = axes[1, 2]
    if results["icm_losses"]:
        ax.plot(results["icm_losses"], color='magenta', linewidth=1, alpha=0.7)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("ICM Loss (EMA)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"training_curves_run{run_id}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot evaluation results if available
    if results["eval_results"]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
        
        episodes = [er['episode'] for er in results["eval_results"]]
        success_rates = [er['results'][list(er['results'].keys())[0]]['success_rate'] 
                        for er in results["eval_results"]]
        
        ax.plot(episodes, success_rates, marker='o', linewidth=2, markersize=8, color='green')
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target (90%)')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success Rate")
        ax.set_title("Evaluation Success Rate")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"eval_success_rate_run{run_id}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Plots saved to {plots_dir}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Train
    agent, env, encoder, law_instances, results, tracker = train()
    
    # Save final model
    out_path = Path(args.out_dir)
    model_file = out_path / "model_final.pt"
    torch.save({
        'q_network': agent.q_network.state_dict(),
        'target_network': agent.target_network.state_dict(),
        'world_model': agent.world_model.state_dict(),
        'icm': agent.icm.state_dict() if agent.icm else None,
        'config': vars(args),
    }, model_file)
    print(f"\n✓ Model saved: {model_file}")
    
    # Save results
    results_file = out_path / "results.json"
    # Convert numpy types for JSON serialization
    json_results = {
        'episode_rewards': [float(r) for r in results['episode_rewards']],
        'episode_lengths': [int(l) for l in results['episode_lengths']],
        'successes': int(results['successes']),
        'q_losses': [float(l) for l in results['q_losses']],
        'model_losses': [float(l) for l in results['model_losses']],
        'icm_losses': [float(l) for l in results['icm_losses']],
        'law_sequence': results['law_sequence'],
        'eval_results': results['eval_results']
    }
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Results saved: {results_file}")
    
    # Final evaluation
    print("\nFinal Evaluation...")
    eval_results = evaluate_agent(agent, env, encoder, law_instances, args.eval_episodes, args.max_steps)
    
    # Summary
    summary = {
        "config": vars(args),
        "train_stats": {
            "total_episodes": len(results["episode_rewards"]),
            "successes": results["successes"],
            "success_rate": results["successes"] / len(results["episode_rewards"]),
            "avg_reward": float(np.mean(results["episode_rewards"])),
            "std_reward": float(np.std(results["episode_rewards"])),
            "avg_length": float(np.mean(results["episode_lengths"])),
            "final_avg_reward_50": float(np.mean(results["episode_rewards"][-50:])),
            "final_success_rate_50": float(np.mean([1 if r > 5 else 0 for r in results["episode_rewards"][-50:]])),
        },
        "eval_results": eval_results,
        "law_discovery": tracker.get_summary(),
        "training_completed": True
    }
    
    with open(out_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot
    plot_training_results(results, out_path, 0)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nTraining Stats:")
    print(f"  Total Episodes: {summary['train_stats']['total_episodes']}")
    print(f"  Successes: {results['successes']}")
    print(f"  Success Rate: {summary['train_stats']['success_rate']:.2%}")
    print(f"  Avg Reward: {summary['train_stats']['avg_reward']:.2f} ± {summary['train_stats']['std_reward']:.2f}")
    print(f"  Final Avg (50 ep): {summary['train_stats']['final_avg_reward_50']:.2f}")
    print(f"  Final Success Rate (50 ep): {summary['train_stats']['final_success_rate_50']:.2%}")
    print(f"\nFinal Evaluation Results:")
    for law_name, metrics in eval_results.items():
        print(f"  {law_name}:")
        print(f"    Success Rate: {metrics['success_rate']:.2%}")
        print(f"    Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"    Avg Length: {metrics['avg_length']:.1f}")
    print(f"\n📊 Law Discovery Data:")
    print(f"  Trajectories collected: {tracker.get_summary()['total_trajectories']}")
    print(f"  High-error transitions: {tracker.get_summary()['total_high_error']}")
    print(f"  Embeddings saved: {tracker.get_summary()['total_embeddings']}")
    print("\n" + "=" * 80)
    print("✅ All fixes applied successfully!")
    print("=" * 80)
    
    return summary

if __name__ == "__main__":
    try:
        summary = main()
        print("\n🎉 Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise