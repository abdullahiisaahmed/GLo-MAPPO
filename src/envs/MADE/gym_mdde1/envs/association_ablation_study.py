###########################################################################################################
# GLo-MAPPO: Multi-Flying LoRa UAVs Env -- Association-Scheme Ablation (Experiment B)

# Author by: A.I. Ahmed
# Date: 2025-02-08
# Description: Custom Gym environment for the Multi-Flying LoRa UAVs scenario.

###########################################################################################################

############## Import required libraries  ##################
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.sparse import lil_matrix
import gym
from gym import spaces
from gym.spaces import Discrete, Tuple, MultiDiscrete
############## Import required libraries  ##################


def generate_positions(area, num_points, cs_position, exclusion_radius, seed=None):
    if seed is not None:
        np.random.seed(seed)
    positions = []
    cs_x, cs_y = cs_position[0], cs_position[1]
    while len(positions) < num_points:
        x = np.random.uniform(0, area[0])
        y = np.random.uniform(0, area[1])
        dx = x - cs_x
        dy = y - cs_y
        if dx**2 + dy**2 > exclusion_radius**2:
            positions.append([x, y, 0])
    return np.array(positions)

def generate_uav_positions(num_uavs, altitude):
    predefined = {
        2: [[50, 50, altitude], [150, 50, altitude]],
        3: [[50, 50, altitude], [150, 50, altitude], [250, 50, altitude]],
        4: [[50, 50, altitude], [150, 50, altitude], [250, 50, altitude], [350, 50, altitude]],
        5: [[50, 50, altitude], [150, 50, altitude], [250, 50, altitude], [350, 50, altitude], [450, 50, altitude]]

    }
    if num_uavs in predefined:
        return np.array(predefined[num_uavs])

def compute_pathloss(n_pos, g_pos, f):
    al = 4.88
    lam = 0.43
    eta_los = 0.1
    eta_nlos = 21
    c = 3e8

    d = distance.euclidean(n_pos, g_pos)              # 3D slant range
    d_horiz = distance.euclidean(n_pos[:2], g_pos[:2])  # horizontal distance, for elevation angle

    z_diff = abs(g_pos[2] - n_pos[2])
    theta = math.degrees(math.atan2(z_diff, d_horiz))

    var_x = np.longdouble(-lam * (theta - al))
    p_los = 1 / (1 + al * math.exp(var_x))
    p_nlos = 1 - p_los
    pl_fs = 20 * np.log10(4 * math.pi * d * f / c)
    a2g = pl_fs + (p_los * eta_los) + (p_nlos * eta_nlos)
   
    return a2g

def compute_gain(Ed_pos, Gw_pos, f, fading_power=1.0):
    pathloss_db = compute_pathloss(Ed_pos, Gw_pos, f)
    gain_linear = 10 ** (-pathloss_db / 10)
    return gain_linear * fading_power     # fading_power = |h|^2 (1.0 = no fading)

def rician_fading_power(theta_deg, A1, A2, rng):
    """Unit-mean Rician |h|^2 with elevation-dependent K-factor kappa = A1*exp(A2*theta_rad).
    Higher elevation -> larger K -> stronger LoS dominance (less fading). E[|h|^2] = 1,
    so fading redistributes gain stochastically without changing mean power."""
    theta_rad = math.radians(theta_deg)
    kappa = A1 * math.exp(A2 * theta_rad)
    s = math.sqrt(kappa / (kappa + 1.0))            # specular (LoS) component
    sigma = math.sqrt(1.0 / (2.0 * (kappa + 1.0)))  # scatter component per dimension
    hr = s + sigma * rng.standard_normal()
    hi = sigma * rng.standard_normal()
    return hr * hr + hi * hi

def compute_snr(tp_dbm, Ed_pos, Gw_pos, f, gain_linear=None, sigma_noise_w=None):
    tp_w = 10 ** ((tp_dbm - 30) / 10)
    if sigma_noise_w is None:
        sigma_noise_w = 10 ** ((-117 - 30) / 10)  
    if gain_linear is None:
        gain_linear = compute_gain(Ed_pos, Gw_pos, f)
    snr_val = tp_w * gain_linear / sigma_noise_w
    return snr_val


def compute_sinr_components(tp_dbm, ed_idx, ed_positions, sf, uav_pos,
                            frequency, current_sfs_all_eds, comm_range,
                            tp_dbm_allocated_all_eds, fading_row=None,
                            sigma_noise_w=None):
    """Single-pass SINR decomposition for the three-condition LoRa capture model.

    Returns (sinr_co, sinr_inter, sinr_eff), all LINEAR:
      sinr_co    = desired / (same-SF interference + noise)       -> co-SF gate   (>= 6 dB)
      sinr_inter = desired / (different-SF interference + noise)  -> inter-SF gate (>= q_iSFm)
      sinr_eff   = desired / (ALL interference + noise)           -> Shannon achieved rate

    All interferers enter at FULL power. The ONLY thing separating co- from
    inter-SF is which sum each interferer lands in (no cross-SF rejection factor).
    """
    if sigma_noise_w is None:
        sigma_noise_w = 10 ** ((-117 - 30) / 10)

    Ed_pos = ed_positions[ed_idx]
    fade_desired = 1.0 if fading_row is None else fading_row[ed_idx]
    gain_desired = compute_gain(Ed_pos, uav_pos, frequency, fading_power=fade_desired)
    desired_signal_power_w = (10 ** ((tp_dbm - 30) / 10)) * gain_desired

    I_same = 0.0   # same-SF interference power (W)
    I_diff = 0.0   # different-SF interference power (W)
    n_co = 0       # #in-range co-SF interferers   (offered load for ALOHA temporal model)
    n_in = 0       # #in-range inter-SF interferers
    for i, interferer_pos in enumerate(ed_positions):
        if i == ed_idx:
            continue
        if current_sfs_all_eds[i] == 0:          # SF=0 -> ED not transmitting this step
            continue
        if distance.euclidean(interferer_pos[:2], uav_pos[:2]) > comm_range:
            continue
        interferer_tp_w = 10 ** ((tp_dbm_allocated_all_eds[i] - 30) / 10)
        fade_i = 1.0 if fading_row is None else fading_row[i]
        gain_i = compute_gain(interferer_pos, uav_pos, frequency, fading_power=fade_i)
        p_i = interferer_tp_w * gain_i
        if current_sfs_all_eds[i] == sf:
            I_same += p_i; n_co += 1
        else:
            I_diff += p_i; n_in += 1

    sinr_co    = desired_signal_power_w / (I_same + sigma_noise_w)
    sinr_inter = desired_signal_power_w / (I_diff + sigma_noise_w)
    sinr_eff   = desired_signal_power_w / (I_same + I_diff + sigma_noise_w)
    return sinr_co, sinr_inter, sinr_eff, n_co, n_in


def compute_datarate(bw, sinr):
    
    datarate1 = bw * np.log2(1 +  sinr)
    
    return datarate1

def lora_time_on_air(sf, payload_bytes=20, bw=125e3, coding_rate=1,
                     preamble_len=8, explicit_header=True, crc=True,
                     low_dr_opt=None):
    """LoRa Time-on-Air (airtime) in seconds, per Semtech SX1276 datasheet / AN1200.13.
    sf: 7..12; coding_rate: 1..4 (= coding rates 4/5..4/8); bw in Hz.
    ToA grows ~2x per SF step, so it is the physical link between an ED's SF choice
    and how much of its duty-cycle budget each transmission consumes."""
    sf = int(sf)
    t_sym = (2 ** sf) / bw                       # symbol duration

    if low_dr_opt is None:
        low_dr_opt = t_sym > 16e-3
    de    = 1 if low_dr_opt else 0
    ih    = 0 if explicit_header else 1
    crc_f = 1 if crc else 0

    t_preamble = (preamble_len + 4.25) * t_sym

    num = 8 * payload_bytes - 4 * sf + 28 + 16 * crc_f - 20 * ih
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(math.ceil(num / den) * (coding_rate + 4), 0)

    return t_preamble + n_payload * t_sym

def propulsion_power(V_array): 
    V_array = np.array(V_array) # Ensure it's an array
    delta = 0.011; rho = 1.168; A = 0.214; Omega = 400.0; R = 0.26
    W = 20.0; k = 0.11; rotor_solidity = 0.045; kappa_tilde = 1.0
    v0 = 6.325; S_fp = 0.009
    P0 = (delta / 8.0) * rho * rotor_solidity * A * (Omega**3) * (R**3)
    Pi_hover = (1.0 + k) * (W**1.5) / math.sqrt(2.0 * rho * A)
    
    # Ensure V_array is handled element-wise for calculations involving V^2, V^3, V^4
    V_squared = V_array**2
    V_cubed = V_array**3
    V_quad = V_array**4

    term_profile = P0 * (1.0 + 3.0 * V_squared / (Omega**2 * R**2))
    
    # Handle sqrt for arrays safely
    inside_sqrt_arg = (kappa_tilde**2) + V_quad / (4.0 * (v0**4))
    # Ensure non-negativity before sqrt, then handle potential complex numbers if needed, though physics implies real.
    sqrt_term_base = np.sqrt(np.maximum(0, inside_sqrt_arg)) 
    
    term_induced_factor = sqrt_term_base - (V_squared / (2.0 * (v0**2)))
    # Ensure non-negativity again before the outer sqrt
    term_induced = Pi_hover * kappa_tilde * np.sqrt(np.maximum(0, term_induced_factor))

    term_parasitic = 0.5 * S_fp * rho * V_cubed
    P_prop = term_profile + term_induced + term_parasitic
    return P_prop


# Configuration for the environment
env_config = {
    'sf_options': [7, 8, 9, 10, 11, 12],
    'tp_options': [2, 5, 8, 11, 14],
    'movement_directions_angle': 120,  # Angle in degrees for the cone-based movement
    'movement_directions': 15, 
    'speed_steps': [0, 0.2, 0.4, 0.6, 0.8, 1.0],      # 6 levels {0, delta, ..., Smax}, delta = Smax/M, M=5
    'snr_thresholds': {7: -6, 8: -9, 9: -12, 10: -15, 11: -17.5, 12: -20},          # (Waret et al. 2018)
    'cosf_capture_threshold_db': 6.0,   # q_coSF, fixed for ALL SFs (= 4 in linear)
    'intersf_capture_thresholds': {     # q_iSFm [dB], Table I (Waret et al. 2018)
        7: -7.5, 8: -9.0, 9: -13.5, 10: -15.0, 11: -18.0, 12: -22.5},
    'bandwidth': 125e3,
    'frequency': 868e6,
    'exclusion_radius': 100,  # minimum distance from CS for ED placement
    'seed_value': 41,
    'delta_t': 0.5,
    'comm_range': 300,
    'power_limit': 200, # in watts
    'lora_ed_circuit_power': 1e-3,   # per-ED circuit power P_c in W (1 mW ≈ 0 dBm),
    'duty_cycle': 0.01,   # LoRa ED duty cycle (EU868 1%); drives ALOHA collision load + airtime budget
    'payload_bytes': 20,   # LoRa application payload size (bytes) for Time-on-Air
    'coding_rate': 1,      # LoRa coding rate index 1..4 -> 4/5, 4/6, 4/7, 4/8
    'sense_range': 75,   # radar sensing range R_sense (m) for neighbor-UAV observation; R_sense >> D_safe

    # --- Channel impairments ---
    'rician_A1': 1.0,        # K-factor scale: kappa(theta) = A1 * exp(A2 * theta_rad)
    'rician_A2': 4.39,       # K-factor elevation exponent (theta in radians); calibrated for G2A links
    'noise_figure_db': 6,        # LoRa receiver noise figure
    'noise_margin_db': 0,        # extra margin for unmodeled interference/impairments

    'propulsion_params': { 
        'P_bld': 79.86, 'Omega_tip': 300, 'C_d': 0.5, 'rho': 1.225,
        'A_ref': 0.3, 'P_ind': 88.63, 'alpha_0': 0.6, 'P_max': 200.0
    },

    # ---------- reward weights----------
    'reward_weights': {
        'omega1_association_indicator': 10.0,        # ω1  association indicator â_u
        'omega2_coverage_bonus': 400.0,              # (B) coverage bonus per NEW unique ED -- ATTACHED to association
        'omega3_system_ee_step':  4e-7,              # ω2  EE  
        'omega5_collision_penalty': 200.0,           # ω3  collision indicator D_safe
        'omega6_distance_to_cs_penalty': 2.0,        # ω4  CS distance (applied to d_cs/d_diag)
        'omega7_cs_arrival_failure_penalty': 400.0,  # ω5  terminal CS-arrival failure
        'omega4_spread_incentive': 5.0,              # (D) inter-UAV spread reward per step (0 disables)
        'spread_cap': 250.0,                         # (D) separation [m] at which the spread reward saturates
    },

}


class MultiFlyingLoRaEnv(gym.Env):
    def __init__(self, num_uavs, num_eds, area_size, uav_altitude, 
                 max_episode_steps, max_speed, comm_range, 
                 safe_distance, config=None, **kwargs):
        super().__init__()
        self.config = env_config.copy()
        if config: self.config.update(config)
        self.config.update(kwargs)

        self._setup_assoc_mode(kwargs.get('assoc_mode', None))

        self.n_agents = num_uavs
        self.num_eds = num_eds
        self.area_size = np.array(area_size)
        self.uav_altitude = uav_altitude
        self.max_episode_steps = max_episode_steps
        self.max_speed = max_speed
        self.comm_range = comm_range
        self.safe_distance = safe_distance
        self.max_ed_per_uav = max(1, num_eds // num_uavs)  
        self.episode_duration = max_episode_steps * self.config['delta_t']
        self.max_dc_time = self.episode_duration * self.config['duty_cycle']

        self.sigma_noise_dBm = (-174
                                + 10 * math.log10(self.config['bandwidth'])
                                + self.config['noise_figure_db']
                                + self.config['noise_margin_db'])
        self.sigma_noise_w = 10 ** ((self.sigma_noise_dBm - 30) / 10)

        # --- Global state for the centralized critic  ---
        # s[t] = { {q_u}, {kappa_u}, {q_cs}, {g_v}, {G_{u,v}} for all u,v }
        # Components:  UAV xy (2/UAV) + UAV vel xy (2/UAV) + kappa_u (1/UAV)
        #            + ED xy (2/ED)  + full gain matrix (U*V)  + q_cs xy (2)
        self.state_size = (
            self.n_agents * 2          # UAV positions (x, y)
            + self.n_agents * 2        # UAV velocities (vx, vy)  -> kinematic part of kappa_u
            + self.n_agents * 1        # kappa_u (propulsion power state)
            + self.num_eds * 2         # ED positions (x, y)
            + self.n_agents * self.num_eds   # full channel-gain matrix G_{u,v}
            + 2                        # charging-station position (x, y)
        )

        self.sense_range = self.config.get('sense_range', 50)        # R_sense for neighbor sensing
        self.num_neighbors = max(0, self.n_agents - 1)               # K = U - 1
        self.num_neighbor_features = 3                               # per neighbor: (rel_x, rel_y, dist)

        _achievable_speeds = np.array([0.0] + [c * self.max_speed for c in self.config['speed_steps']])
        self.kappa_max = float(np.max(propulsion_power(_achievable_speeds)) * 1.2)

        # ED-slot features: [x, y, dist_to_uav, faded_gain]. Per-ED SF/TP are now
        # chosen by the policy (one action head per ED slot), so the UAV-scalar SF/TP
        # features were dropped from the observation.
        self.num_ed_features = 4
        self.num_uav_features = 8 + 1 + self.num_neighbor_features * self.num_neighbors
        self.total_features_per_ed_slot = self.num_ed_features + self.num_uav_features

        # --- Factored per-ED action space ---
        # Mobility heads (direction, speed) are shared per UAV. SF and TP become a
        # per-ED-slot decision: one (SF, TP) head pair per observation slot k = 0..M-1.
        # SF/TP heads carry an extra index 0 = "null / do-not-serve this ED", which (a)
        # lets the agent explicitly decline a weak ED, and (b) is the only available action
        # on padded (empty) slots, so a whole head is never fully masked.
        self._n_sf = len(self.config['sf_options']) + 1   # +1 for null at index 0
        self._n_tp = len(self.config['tp_options']) + 1   # +1 for null at index 0
        self.action_nvec = [
            self.config['movement_directions'],
            len(self.config['speed_steps']),
        ] + [self._n_sf, self._n_tp] * self.max_ed_per_uav
        self.n_actions = sum(self.action_nvec)  # total logit width
        _per_agent_space = MultiDiscrete(self.action_nvec)
        self.action_space = gym.spaces.Tuple(tuple(self.n_agents * [_per_agent_space]))

        base_low = [
                    0, 0, 0, 0,
                    0, 0, -self.max_speed, -self.max_speed, -self.area_size[0], -self.area_size[1], 0, 0,
                    0.0,
                ]
        base_high = [
            self.area_size[0], self.area_size[1], self.comm_range * 1.1, 1.0,
            self.area_size[0], self.area_size[1], self.max_speed, self.max_speed,
            self.area_size[0], self.area_size[1],
            self.config['movement_directions'] - 1, len(self.config['speed_steps']) - 1,
            self.kappa_max,
        ]
        neigh_low = [-self.sense_range, -self.sense_range, 0.0] * self.num_neighbors
        neigh_high = [self.sense_range, self.sense_range, self.sense_range] * self.num_neighbors

        low_bounds = np.array(base_low + neigh_low, dtype=np.float32)
        high_bounds = np.array(base_high + neigh_high, dtype=np.float32)

        assert len(low_bounds) == self.total_features_per_ed_slot, "Bounds length mismatch!"
        assert len(high_bounds) == self.total_features_per_ed_slot, "Bounds length mismatch!"



        obs_space_list = []
        for _ in range(self.n_agents):
             agent_low_bounds = np.tile(low_bounds, (self.max_ed_per_uav, 1))
             agent_high_bounds = np.tile(high_bounds, (self.max_ed_per_uav, 1))

             obs_space_list.append(
                 spaces.Box(low=agent_low_bounds, high=agent_high_bounds,
                            shape=(self.max_ed_per_uav, self.total_features_per_ed_slot),
                            dtype=np.float32)
             )
        self.observation_space = spaces.Tuple(tuple(obs_space_list))

        self.cs_position = np.array([self.area_size[0] - 100, self.area_size[1] - 100, self.uav_altitude])
        self.arrival_threshold = self.config['exclusion_radius'] - 20
        
        self.uav_positions = np.zeros((self.n_agents, 3))
        self.ed_positions = np.zeros((self.num_eds, 3))

        self.episode_count = 0 
        self.initialize_tracking_variables()

    # ===== ASSOCIATION-SCHEME ABLATION (Experiment B, reviewer R5) =====
    # The policy still sets UAV trajectories + per-ED SF/TP; we only change WHICH UAV
    # serves each ED among the eligible (observed + non-null SF/TP + in-range + under-
    # quota) candidates. mode: gain (ours) | distance | random | fixed.
    _VALID_ASSOC = ('gain', 'distance', 'random', 'fixed')
    _assoc_banner_shown = False   # print once per worker process
    # SF/TP for benchmark-served EDs whose chosen UAV did NOT observe them (no policy head).
    _BASELINE_DEFAULT_SF = 9
    _BASELINE_DEFAULT_TP_DBM = 14

    def _setup_assoc_mode(self, mode):
        self.assoc_mode = (str(mode).lower() if mode is not None else 'gain')
        if self.assoc_mode not in self._VALID_ASSOC:
            raise ValueError(
                f"unknown assoc_mode='{mode}'. Valid: {', '.join(self._VALID_ASSOC)}")
        self._assoc_rng = np.random.default_rng(self.config.get('seed_value', None))
        self.fixed_ed_to_uav_map = {}
        if not MultiFlyingLoRaEnv._assoc_banner_shown:
            desc = {'gain':     'max channel gain  -- OURS (NEW policy-coupled scheme)',
                    'distance': 'nearest UAV       -- benchmark (OLD free scheme)',
                    'random':   'random in-range   -- benchmark (OLD free scheme)',
                    'fixed':    'pre-assigned      -- benchmark (OLD free scheme)'}[self.assoc_mode]
            print("=" * 64 + f"\n  [ASSOC ABLATION] association scheme = '{self.assoc_mode}'"
                  f"  ->  {desc}\n" + "=" * 64, flush=True)
            MultiFlyingLoRaEnv._assoc_banner_shown = True

    def _select_server(self, cands, ed_idx):
        """Pick ONE serving UAV among eligible candidates per self.assoc_mode.
        cands: list of dicts {i_active,u,gain,dist,sf_raw,tp_raw}. Returns one or None."""
        if not cands:
            return None
        if self.assoc_mode == 'gain':
            return max(cands, key=lambda c: c['gain'])
        if self.assoc_mode == 'distance':
            return min(cands, key=lambda c: c['dist'])
        if self.assoc_mode == 'random':
            return cands[int(self._assoc_rng.integers(len(cands)))]
        if self.assoc_mode == 'fixed':
            tgt = self.fixed_ed_to_uav_map.get(ed_idx)
            for c in cands:
                if c['u'] == tgt:
                    return c
            return None   # fixed UAV not eligible this step -> ED not served
        return max(cands, key=lambda c: c['gain'])

    def _initialize_fixed_associations(self):
        """Fixed ED->UAV map from INITIAL positions: greedy nearest, respecting quota.
        (Ported from the 1st-submission assoc_mode.py.)"""
        fixed_map = {}
        counts = np.zeros(self.n_agents, dtype=int)
        pairs = [(distance.euclidean(self.ed_positions[e][:2], self.uav_positions[u][:2]), e, u)
                 for e in range(self.num_eds) for u in range(self.n_agents)]
        pairs.sort(key=lambda x: x[0])
        for _d, e, u in pairs:
            if e in fixed_map or counts[u] >= self.max_ed_per_uav:
                continue
            fixed_map[e] = u
            counts[u] += 1
            if len(fixed_map) == self.num_eds:
                break
        return fixed_map

    def initialize_tracking_variables(self):
        self.timestep = 0
        self.episode_data = []
        self.dones = np.zeros(self.n_agents, dtype=bool)

        self.uav_unique_eds_ever = [set() for _ in range(self.n_agents)] 
        self.system_unique_eds_ever = set() 
        
        self.cumulative_data = 0.0 
        self.cumulative_data_this_step = 0.0 
        self.ed_tx_power_this_step = 0.0   

        self.ed_snr_this_step = {}   # ed_idx -> SNR (linear) for active/transmitting EDs this step

        # --- Per-ED action binding: slot <-> ED maps captured at get_obs() time ---
        # _obs_slot_ed[u][k]  = ed_idx occupying slot k of UAV u's observation (-1 if padded)
        # _obs_ed_slot[u][e]  = slot index of ED e in UAV u's observation
        # active_slot_counts[u] = number of real (non-padded) ED slots -> drives action masking
        self._obs_slot_ed = [[-1] * self.max_ed_per_uav for _ in range(self.n_agents)]
        self._obs_ed_slot = [dict() for _ in range(self.n_agents)]
        self.active_slot_counts = np.zeros(self.n_agents, dtype=int)

        self.uav_propulsion_power_this_step = np.zeros(self.n_agents)
        self.total_propulsion_power_episode = np.zeros(self.n_agents)
        self.total_ed_tx_power_episode = 0.0

        self.ed_transmission_history = {}
        self.ed_sf_history = {}   # ed_idx -> list of SF used on each successful tx (parallel to ed_transmission_history)
        # Per-SF accounting for Time-on-Air duty-cycle diagnostics
        self.tx_count_by_sf = {sf: 0 for sf in self.config['sf_options']}
        self.airtime_by_sf  = {sf: 0.0 for sf in self.config['sf_options']}
        self.unique_eds = set() 
        self.collision_counts = np.zeros(self.n_agents, dtype=int)
        self.has_reached_cs = np.zeros(self.n_agents, dtype=bool)
        self.uav_velocities = np.zeros((self.n_agents, 2))

        self._fade_rng = np.random.default_rng(self.config.get('seed_value', None))
        self.fading_matrix = np.ones((self.n_agents, self.num_eds), dtype=np.float64)

        self._validate_action_space() 

        if self.config['sf_options']:
            self.current_sf = np.full(self.n_agents, self.config['sf_options'][0], dtype=int)
        else:
            self.current_sf = np.zeros(self.n_agents, dtype=int) 
        
        if self.config['tp_options']:
            self.current_tp = np.full(self.n_agents, self.config['tp_options'][0], dtype=int)
        else:
            self.current_tp = np.zeros(self.n_agents, dtype=int) 

        self.action_history = {
            'direction': np.zeros((self.n_agents, self.max_episode_steps), dtype=int),
            'speed': np.zeros((self.n_agents, self.max_episode_steps), dtype=int)
        }
        self.last_distances_to_cs = np.zeros(self.n_agents)

        self.episode_sum_weighted_r_association = 0.0
        self.episode_sum_weighted_r_system_ee = 0.0
        self.episode_sum_weighted_r_collision = 0.0
        self.episode_sum_weighted_r_distance_cs = 0.0
        self.episode_sum_weighted_r_coverage = 0.0   
        self.episode_sum_weighted_r_spread = 0.0     

        self.episode_sum_raw_a_u_t = 0.0
        self.episode_sum_raw_EE_sys_at_each_step = 0.0
        self.episode_sum_norm_EE_sys_at_each_step = 0.0  # per-served-ED normalized EE accumulator
        self.episode_sum_raw_D_safe = 0.0
        self.episode_sum_raw_d_cs = 0.0

        self.episode_sum_weighted_r_cs_arrival_failure = 0.0
        self.episode_sum_raw_did_not_reach_cs_at_end_count = 0.0 # Counts agents failing to reach CS

        self.episode_num_agent_reward_calculations = 0


    def reset(self):
        self.uav_positions = generate_uav_positions(self.n_agents, self.uav_altitude)
        self.ed_positions = generate_positions(
            self.area_size, self.num_eds,
            self.cs_position, self.config['exclusion_radius'], self.config['seed_value']
        )
        # (assoc ablation) build the fixed ED->UAV map from this episode's initial positions
        if self.assoc_mode == 'fixed':
            self.fixed_ed_to_uav_map = self._initialize_fixed_associations()
        self.initialize_tracking_variables()

        self.associations = lil_matrix((self.n_agents, self.num_eds), dtype=np.float32)
        self.sf_allocations = lil_matrix((self.n_agents, self.num_eds), dtype=int)
        self.tp_allocations = lil_matrix((self.n_agents, self.num_eds), dtype=int)
        
        self.sf_allocations_all_eds = np.zeros(self.num_eds, dtype=int)
        self.tp_allocations_all_eds = np.zeros(self.num_eds)

        self.capture_stats = {
            'attempts': 0, 'delivered': 0,
            'fail_reception': 0, 'fail_cosf': 0, 'fail_intersf': 0,
        }

        self._measured_gain = None
        self._gain_fresh = None
        self._last_gain = None


        self.initial_distances_to_cs = np.array([
            distance.euclidean(pos[:2], self.cs_position[:2])
            for pos in self.uav_positions
        ])
        self.last_distances_to_cs = self.initial_distances_to_cs.copy()
        self._init_dist_to_cs = self.initial_distances_to_cs.copy()   # <--- NEW LINE

        
        self.episode_count += 1
        return self.get_obs()


    def get_state(self):
            """global state for the CTDE critic.
            State Space s[t]; contains information no single agent observes
            (notably kappa_u for all UAVs and the full U x V gain matrix)."""
            parts = []

            # All UAV positions (x, y)
            for i in range(self.n_agents):
                parts.append(self.uav_positions[i][:2].astype(np.float32))

            # All UAV velocities (vx, vy)
            for i in range(self.n_agents):
                parts.append(self.uav_velocities[i].astype(np.float32))

            # All UAV propulsion-power states kappa_u
            parts.append(self.uav_propulsion_power_this_step.astype(np.float32))

            # All ED positions (x, y)
            for j in range(self.num_eds):
                parts.append(self.ed_positions[j][:2].astype(np.float32))

            gain_matrix = np.empty(self.n_agents * self.num_eds, dtype=np.float32)
            idx = 0
            for i in range(self.n_agents):
                for j in range(self.num_eds):
                    g_lin = compute_gain(
                        self.ed_positions[j], self.uav_positions[i],
                        self.config['frequency'],
                        fading_power=self.fading_matrix[i, j]
                    )
                    gain_matrix[idx] = 10.0 * math.log10(max(g_lin, 1e-20))
                    idx += 1
            parts.append(gain_matrix)

            # Charging-station position (x, y)
            parts.append(self.cs_position[:2].astype(np.float32))

            state = np.concatenate(parts).astype(np.float32)
            assert state.shape[0] == self.state_size, \
                f"get_state() length {state.shape[0]} != state_size {self.state_size}"
            return state


    def set_measured_gain(self, gain_matrix, fresh_mask):
        """Closed-loop hook: inject externally MEASURED channel gains (linear, ED->UAV)
        for the observation, e.g. derived from FLoRa RSSI/SNR. Shapes (n_agents, num_eds).
        Fresh entries update a per-episode last-known cache; get_obs then uses measured
        (fresh) -> last-known -> analytic model, in that order. No-op unless called, so
        the open-loop path and training are unaffected."""
        g = np.asarray(gain_matrix, dtype=np.float64).reshape(self.n_agents, self.num_eds)
        f = np.asarray(fresh_mask, dtype=bool).reshape(self.n_agents, self.num_eds)
        if getattr(self, '_last_gain', None) is None:
            self._last_gain = np.zeros((self.n_agents, self.num_eds), dtype=np.float64)
        self._last_gain[f] = g[f]
        self._measured_gain = g
        self._gain_fresh = f

    def _gain_for_obs(self, uav_idx, ed_idx, ed_pos, uav_pos):
        """Channel gain ED->UAV used in get_obs. Returns the injected measured gain
        (closed-loop) when available/fresh, else last-known measured, else the analytic
        A2G+fading model (default/open-loop)."""
        mg = getattr(self, '_measured_gain', None)
        if mg is not None:
            if self._gain_fresh[uav_idx, ed_idx]:
                return float(mg[uav_idx, ed_idx])
            lk = self._last_gain[uav_idx, ed_idx]
            if lk > 0.0:
                return float(lk)
        return compute_gain(ed_pos, uav_pos, self.config['frequency'],
                            fading_power=self.fading_matrix[uav_idx, ed_idx])

    def get_obs(self):
        obs_list = []
        for uav_idx in range(self.n_agents):
            uav_pos = self.uav_positions[uav_idx]
            uav_vel = self.uav_velocities[uav_idx] 

            raw_uav_x = uav_pos[0]
            raw_uav_y = uav_pos[1]
            raw_vel_x = uav_vel[0]
            raw_vel_y = uav_vel[1]

            dx_cs = self.cs_position[0] - uav_pos[0] 
            dy_cs = self.cs_position[1] - uav_pos[1] 

            prev_dir_idx = self.action_history['direction'][uav_idx, max(0, self.timestep - 1)] 
            prev_speed_idx = self.action_history['speed'][uav_idx, max(0, self.timestep - 1)] 

            kappa_u = self.uav_propulsion_power_this_step[uav_idx]

            neighbor_features = []
            if self.num_neighbors > 0:
                neigh = []
                for other_idx in range(self.n_agents):
                    if other_idx == uav_idx:
                        continue
                    d_uu = distance.euclidean(uav_pos[:2], self.uav_positions[other_idx][:2])
                    neigh.append((d_uu, other_idx))
                neigh.sort(key=lambda x: x[0])
                for d_uu, other_idx in neigh:
                    if len(neighbor_features) >= self.num_neighbor_features * self.num_neighbors:
                        break
                    if d_uu <= self.sense_range:
                        other_pos = self.uav_positions[other_idx]
                        neighbor_features.extend([
                            other_pos[0] - uav_pos[0],
                            other_pos[1] - uav_pos[1],
                            d_uu
                        ])
            while len(neighbor_features) < self.num_neighbor_features * self.num_neighbors:
                neighbor_features.append(0.0)

            uav_features_raw = [
                raw_uav_x, raw_uav_y, raw_vel_x, raw_vel_y,
                dx_cs, dy_cs, prev_dir_idx, prev_speed_idx, kappa_u
            ] + neighbor_features

            nearby_eds_data = []
            slot_ed = []   # ed_idx occupying each filled slot, in slot (nearest-first) order
            if self.num_eds > 0:
                ed_distances_to_uav = [distance.euclidean(ed_pos[:2], uav_pos[:2]) for ed_pos in self.ed_positions]
                sorted_ed_indices = np.argsort(ed_distances_to_uav)

                for ed_idx in sorted_ed_indices:
                    if len(nearby_eds_data) >= self.max_ed_per_uav:
                        break
                    ed_pos = self.ed_positions[ed_idx]
                    dist_to_uav = ed_distances_to_uav[ed_idx]

                    if dist_to_uav <= self.comm_range:
                        gain_to_uav = self._gain_for_obs(uav_idx, ed_idx, ed_pos, uav_pos)

                        raw_ed_x = ed_pos[0]
                        raw_ed_y = ed_pos[1]

                        ed_features_raw = [raw_ed_x, raw_ed_y, dist_to_uav, gain_to_uav]
                        nearby_eds_data.append(ed_features_raw)
                        slot_ed.append(int(ed_idx))

            self.active_slot_counts[uav_idx] = len(slot_ed)
            self._obs_slot_ed[uav_idx] = slot_ed + [-1] * (self.max_ed_per_uav - len(slot_ed))
            self._obs_ed_slot[uav_idx] = {e: k for k, e in enumerate(slot_ed)}

            while len(nearby_eds_data) < self.max_ed_per_uav:
                nearby_eds_data.append([0.0] * self.num_ed_features)

            obs_matrix_agent = [ed_feat_raw + uav_features_raw for ed_feat_raw in nearby_eds_data]
            obs_list.append(np.array(obs_matrix_agent, dtype=np.float32))

        return tuple(obs_list)



    def _validate_action_space(self):
        if not hasattr(self, 'action_space') or not self.action_space:
            return
        space = self.action_space[0]
        if not isinstance(space, MultiDiscrete):
            raise ValueError("Action space must be MultiDiscrete after refactor.")
        expected = np.array(self.action_nvec)
        if not np.array_equal(space.nvec, expected):
            raise ValueError(f"Action space nvec mismatch! Expected {expected}, got {space.nvec}")


    def _decode_action(self, action):
        # action layout: [direction, speed, (sf_k, tp_k) for k in 0..M-1]
        # SF/TP head index 0 == null ("do-not-serve this ED slot"); real options are
        # index-1 into sf_options / tp_options.
        M = self.max_ed_per_uav
        sf_raw = np.empty(M, dtype=int)
        tp_raw = np.empty(M, dtype=int)
        for k in range(M):
            sf_raw[k] = int(action[2 + 2 * k])
            tp_raw[k] = int(action[2 + 2 * k + 1])
        return {
            'direction_idx': int(action[0]),
            'speed_idx':     int(action[1]),
            'sf_raw':        sf_raw,   # 0 = null, else sf_options[idx-1]
            'tp_raw':        tp_raw,   # 0 = null, else tp_options[idx-1]
        }

    def get_avail_agent_actions(self, uav_idx):
        """Per-logit availability mask (concatenated-head layout) for action masking.
        Mobility heads are always available. For each ED slot: active slots expose all
        SF/TP options (including null at index 0); padded (empty) slots expose ONLY the
        null action, so an empty slot can never drive a phantom transmission and no head
        is ever fully masked (a Categorical needs >= 1 available action)."""
        avail = [1] * self.config['movement_directions']
        avail += [1] * len(self.config['speed_steps'])
        n_active = int(self.active_slot_counts[uav_idx])
        for k in range(self.max_ed_per_uav):
            if k < n_active:
                avail += [1] * self._n_sf
                avail += [1] * self._n_tp
            else:
                avail += [1] + [0] * (self._n_sf - 1)   # only null SF
                avail += [1] + [0] * (self._n_tp - 1)   # only null TP
        return avail

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(u) for u in range(self.n_agents)]

    def step(self, actions):
        current_step_rewards = np.zeros(self.n_agents) 
        info = {'individual_rewards_components': [{} for _ in range(self.n_agents)]} 

        self.cumulative_data_this_step = 0.0
        self.ed_tx_power_this_step = 0.0
        self.uav_propulsion_power_this_step.fill(0.0)

        uav_current_distances_to_cs = np.zeros(self.n_agents)

        M = self.max_ed_per_uav
        sf_raw_all = np.zeros((self.n_agents, M), dtype=int)
        tp_raw_all = np.zeros((self.n_agents, M), dtype=int)

        for uav_idx, action in enumerate(actions):
            if self.dones[uav_idx]:
                uav_current_distances_to_cs[uav_idx] = distance.euclidean(self.uav_positions[uav_idx][:2], self.cs_position[:2])
                continue

            decoded_action = self._decode_action(action)

            sf_raw_all[uav_idx] = decoded_action['sf_raw']
            tp_raw_all[uav_idx] = decoded_action['tp_raw']

            self.action_history['direction'][uav_idx, self.timestep] = decoded_action['direction_idx']
            self.action_history['speed'][uav_idx, self.timestep] = decoded_action['speed_idx']

            uav_pos_before_move = self.uav_positions[uav_idx][:2].copy()
            target_pos_cs = self.cs_position[:2]
            h_vec = target_pos_cs - uav_pos_before_move
            norm_h = np.linalg.norm(h_vec)

           #############################################################################################################################
           ##### Cone-based movement direction calculation
           #############################################################################################################################

            direction_unit_to_cs = h_vec / norm_h                  
            
            beta = math.radians(self.config['movement_directions_angle'] / 2)
        

            theta_relative_to_cs = -beta + 2 * beta * (decoded_action['direction_idx'] / (self.config['movement_directions'] - 1))

            
            rot_matrix = np.array([[math.cos(theta_relative_to_cs), -math.sin(theta_relative_to_cs)],
                                   [math.sin(theta_relative_to_cs), math.cos(theta_relative_to_cs)]])
            actual_direction_vec = rot_matrix @ direction_unit_to_cs
            
            s = 0.0 # Default speed
            if self.config['speed_steps']: # Ensure speed_steps is not empty
                 s = self.config['speed_steps'][decoded_action['speed_idx']] * self.max_speed

            # Enforce propulsion-power budget P_max
            # P_prop(s) <= P_max. 
            p_max = self.config['power_limit']
            if propulsion_power(np.array([s]))[0] > p_max:
                admissible = [c * self.max_speed for c in self.config['speed_steps']
                              if propulsion_power(np.array([c * self.max_speed]))[0] <= p_max]
                s = max(admissible) if admissible else 0.0
                info['individual_rewards_components'][uav_idx]['p_max_clamped'] = True
            
            move_vec = s * actual_direction_vec * self.config['delta_t']
           
            dx, dy = move_vec[0], move_vec[1]

            self.uav_velocities[uav_idx] = [dx / self.config['delta_t'] if self.config['delta_t'] > 0 else 0, 
                                            dy / self.config['delta_t'] if self.config['delta_t'] > 0 else 0]
            
            new_x = self.uav_positions[uav_idx][0] + dx
            new_y = self.uav_positions[uav_idx][1] + dy


           #############################################################################################################################

            clipped_x = np.clip(new_x, 0, self.area_size[0])
            clipped_y = np.clip(new_y, 0, self.area_size[1])
            self.uav_positions[uav_idx] = [clipped_x, clipped_y, self.uav_altitude]

            prop_power_val = propulsion_power(np.array([s]))[0] # Pass speed as array
            self.uav_propulsion_power_this_step[uav_idx] = prop_power_val 
            self.total_propulsion_power_episode[uav_idx] += self.uav_propulsion_power_this_step[uav_idx]

            if new_x != clipped_x or new_y != clipped_y:
                info['individual_rewards_components'][uav_idx]['boundary_violation'] = True
                self.dones[uav_idx] = True

            uav_current_distances_to_cs[uav_idx] = distance.euclidean(self.uav_positions[uav_idx][:2], self.cs_position[:2])
            self.last_distances_to_cs[uav_idx] = uav_current_distances_to_cs[uav_idx]

            if uav_current_distances_to_cs[uav_idx] <= self.arrival_threshold and not self.has_reached_cs[uav_idx]:
                info['individual_rewards_components'][uav_idx]['arrived_at_cs_step'] = True
                self.has_reached_cs[uav_idx] = True

        uav_collided_this_step = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            if self.dones[i]: 
                continue
            for j in range(i + 1, self.n_agents):
                if self.dones[j]: 
                    continue
                if distance.euclidean(self.uav_positions[i][:2], self.uav_positions[j][:2]) <= self.safe_distance:
                    uav_collided_this_step[i] = True
                    uav_collided_this_step[j] = True
                    self.dones[i] = True
                    self.dones[j] = True
                    self.collision_counts[i] += 1
                    self.collision_counts[j] += 1
        
        self.associations = lil_matrix((self.n_agents, self.num_eds), dtype=np.float32)
        self.sf_allocations = lil_matrix((self.n_agents, self.num_eds), dtype=int)
        self.tp_allocations = lil_matrix((self.n_agents, self.num_eds), dtype=int)
        self.sf_allocations_all_eds.fill(0)
        self.tp_allocations_all_eds.fill(0.0)

        uav_had_successful_tx_this_step = np.zeros(self.n_agents, dtype=bool) 

        per_uav_rate_this_step = np.zeros(self.n_agents)
        per_uav_power_this_step = np.zeros(self.n_agents)
        per_uav_served_eds_this_step = np.zeros(self.n_agents, dtype=int)  # #EDs each UAV served this step (for per-served-ED EE normalization)
        per_uav_new_eds_this_step = np.zeros(self.n_agents)  # (B) #FIRST-EVER-served (system-unique) EDs each UAV bagged this step -> coverage bonus

        # --- Per-step Rician fading realization, H[t] in R^{U x V}, one draw per (u,v) ---
        # Block-fading per time slot; reused consistently across association, SNR, SINR,
        # and observed gain so each link has a single channel realization at time t.
        A1, A2 = self.config['rician_A1'], self.config['rician_A2']
        self.fading_matrix = np.ones((self.n_agents, self.num_eds), dtype=np.float64)
        for u in range(self.n_agents):
            for v in range(self.num_eds):
                d_h = distance.euclidean(self.uav_positions[u][:2], self.ed_positions[v][:2])
                z = abs(self.ed_positions[v][2] - self.uav_positions[u][2])
                theta_deg = math.degrees(math.atan2(z, d_h))
                self.fading_matrix[u, v] = rician_fading_power(theta_deg, A1, A2, self._fade_rng)

        active_uav_indices = [i for i, d in enumerate(self.dones) if not d]
        if active_uav_indices and self.num_eds > 0:
            _uav_positions_active = self.uav_positions[active_uav_indices]
            active_to_original_uav_idx = {i_active: i_original for i_active, i_original in enumerate(active_uav_indices)}
            _current_assignment_counts_active = np.zeros(len(active_uav_indices), dtype=int)
            ed_candidates = {}

            for ed_idx in range(self.num_eds):
                ed_pos = self.ed_positions[ed_idx]
                cands = []
                for i_active in range(len(active_uav_indices)):
                    if _current_assignment_counts_active[i_active] >= self.max_ed_per_uav:
                        continue                                  # both schemes respect the quota
                    u = active_to_original_uav_idx[i_active]
                    uav_pos = _uav_positions_active[i_active]
                    d_uv = distance.euclidean(ed_pos[:2], uav_pos[:2])
                    if d_uv > self.comm_range:
                        continue                                  # both schemes range-gate (confirmed)
                    k = self._obs_ed_slot[u].get(ed_idx)
                    served_by_policy = (k is not None and sf_raw_all[u][k] != 0 and tp_raw_all[u][k] != 0)

                    if self.assoc_mode == 'gain':
                        
                        if not served_by_policy:
                            continue
                        sf = self.config['sf_options'][sf_raw_all[u][k] - 1]
                        tp = self.config['tp_options'][tp_raw_all[u][k] - 1]
                    else:
                       
                        if served_by_policy:
                            sf = self.config['sf_options'][sf_raw_all[u][k] - 1]
                            tp = self.config['tp_options'][tp_raw_all[u][k] - 1]
                        else:
                            sf, tp = self._BASELINE_DEFAULT_SF, self._BASELINE_DEFAULT_TP_DBM

                    gain = compute_gain(ed_pos, uav_pos, self.config['frequency'],
                                        fading_power=self.fading_matrix[u, ed_idx])
                    cands.append({'i_active': i_active, 'u': u, 'gain': gain, 'dist': d_uv,
                                  'sf': sf, 'tp': tp})

                chosen = self._select_server(cands, ed_idx)
                if chosen is not None:
                    ed_candidates[ed_idx] = {
                        'uav_idx': chosen['u'], 'gain': chosen['gain'],
                        'sf': chosen['sf'], 'tp': chosen['tp'],
                    }
                    _current_assignment_counts_active[chosen['i_active']] += 1

            for ed_idx, C_info in ed_candidates.items():
                self.sf_allocations_all_eds[ed_idx] = C_info['sf']
                self.tp_allocations_all_eds[ed_idx] = C_info['tp']
            
            for ed_idx, C_info in ed_candidates.items():
                uav_idx_assoc, ed_pos = C_info['uav_idx'], self.ed_positions[ed_idx]
                uav_pos_assoc = self.uav_positions[uav_idx_assoc]
                chosen_sf, chosen_tp_dbm = C_info['sf'], C_info['tp']

                # SF-dependent Time-on-Air: each transmission consumes its actual airtime
                # from the per-ED duty-cycle budget (high SF -> longer ToA -> fewer tx allowed).
                toa = lora_time_on_air(chosen_sf,
                                       payload_bytes=self.config['payload_bytes'],
                                       bw=self.config['bandwidth'],
                                       coding_rate=self.config['coding_rate'])

                if sum(self.ed_transmission_history.get(ed_idx, [])) + toa > self.max_dc_time:
                    continue
                
                snr_val = compute_snr(chosen_tp_dbm, ed_pos, uav_pos_assoc, self.config['frequency'], 
                                        gain_linear=C_info['gain'], sigma_noise_w=self.sigma_noise_w)


#############################################################################################################################
                snr_db = 10 * np.log10(snr_val) if snr_val > 0 else -np.inf
                rx_thr_db = self.config['snr_thresholds'].get(chosen_sf, -np.inf)
                self.ed_snr_this_step[ed_idx] = {
                    'uav': uav_idx_assoc, 'sf': chosen_sf, 'tp': chosen_tp_dbm,
                    'snr_db': snr_db, 'threshold_db': rx_thr_db,
                    'passed': snr_db >= rx_thr_db,
                }

                # --- LoRa capture (power) + per-SF pure-ALOHA temporal collision ---
                # sinr_* gates = power capture; n_co/n_in = #in-range co-/inter-SF
                # interferers for the ALOHA overlap term.
                sinr_co, sinr_inter, sinr_eff, n_co, n_in = compute_sinr_components(
                    chosen_tp_dbm, ed_idx, self.ed_positions, chosen_sf, uav_pos_assoc,
                    self.config['frequency'], self.sf_allocations_all_eds,
                    self.comm_range, self.tp_allocations_all_eds,
                    fading_row=self.fading_matrix[uav_idx_assoc],
                    sigma_noise_w=self.sigma_noise_w)
                sinr_co_db    = 10 * np.log10(sinr_co)    if sinr_co    > 0 else -np.inf
                sinr_inter_db = 10 * np.log10(sinr_inter) if sinr_inter > 0 else -np.inf

                self.capture_stats['attempts'] += 1
                rx_ok = snr_db >= rx_thr_db                          # Gate 1: reception (coverage)
                if not rx_ok:
                    self.capture_stats['fail_reception'] += 1
                    continue

                # Power capture: the tagged packet survives an overlap if it is strong
                # enough vs co-/inter-SF interferers (existing SINR thresholds).
                co_captures = sinr_co_db    >= self.config['cosf_capture_threshold_db']
                in_captures = sinr_inter_db >= self.config['intersf_capture_thresholds'].get(chosen_sf, -np.inf)
                # Temporal collision: pure-ALOHA per SF. rho = duty cycle (saturated
                # assumption); P_overlap = 1 - exp(-2 * N_interferers * rho).
                delta = self.config['duty_cycle']
                p_co_loss = 0.0 if co_captures else (1.0 - math.exp(-2.0 * n_co * delta))
                p_in_loss = 0.0 if in_captures else (1.0 - math.exp(-2.0 * n_in * delta))
                p_deliver = (1.0 - p_co_loss) * (1.0 - p_in_loss)   # expected delivery prob

                # Expected-throughput accounting (fractional). delivered + fail_cosf +
                # fail_intersf == 1 per received attempt (split collision by its cause).
                collided = 1.0 - p_deliver
                denom = p_co_loss + p_in_loss
                if denom > 0.0:
                    self.capture_stats['fail_cosf']    += collided * (p_co_loss / denom)
                    self.capture_stats['fail_intersf'] += collided * (p_in_loss / denom)
                self.capture_stats['delivered'] += p_deliver

                # A delivered (non-colliding) packet sees only noise -> rate from SNR;
                # interference is already captured by the collision probability above.
                dr = compute_datarate(self.config['bandwidth'], snr_val)
                eff_dr = dr * p_deliver                             # expected delivered datarate
                tp_w = 10 ** ((chosen_tp_dbm - 30) / 10)
                p_c = self.config['lora_ed_circuit_power']          # per-ED circuit power (W)

                # Energy is spent on the transmission regardless of collision outcome.
                self.cumulative_data_this_step += eff_dr
                self.ed_tx_power_this_step += (tp_w + p_c)
                per_uav_rate_this_step[uav_idx_assoc] += eff_dr
                per_uav_power_this_step[uav_idx_assoc] += (tp_w + p_c)
                per_uav_served_eds_this_step[uav_idx_assoc] += 1

                self.associations[uav_idx_assoc, ed_idx] = eff_dr
                self.sf_allocations[uav_idx_assoc, ed_idx] = chosen_sf
                self.tp_allocations[uav_idx_assoc, ed_idx] = chosen_tp_dbm
                self.ed_transmission_history.setdefault(ed_idx, []).append(toa)
                self.ed_sf_history.setdefault(ed_idx, []).append(int(chosen_sf))
                self.tx_count_by_sf[int(chosen_sf)] += 1
                self.airtime_by_sf[int(chosen_sf)] += toa

                uav_had_successful_tx_this_step[uav_idx_assoc] = True

                if ed_idx not in self.system_unique_eds_ever:
                    self.system_unique_eds_ever.add(ed_idx)
                    per_uav_new_eds_this_step[uav_idx_assoc] += 1   # (B) first-ever service of this ED -> coverage bonus to the serving UAV
                if ed_idx not in self.uav_unique_eds_ever[uav_idx_assoc]:
                    self.uav_unique_eds_ever[uav_idx_assoc].add(ed_idx)
        

        self.cumulative_data += self.cumulative_data_this_step
        self.total_ed_tx_power_episode += self.ed_tx_power_this_step

        # Per-UAV instantaneous EE: zeta_u[t] = R_u / (P_T,u + P_c,u)
        zeta_u = np.zeros(self.n_agents)
        served = per_uav_power_this_step > 0.0
        zeta_u[served] = per_uav_rate_this_step[served] / per_uav_power_this_step[served]

        zeta_u_norm = np.zeros(self.n_agents)
        _nz = per_uav_served_eds_this_step > 0
        zeta_u_norm[_nz] = zeta_u[_nz] / per_uav_served_eds_this_step[_nz]

        # Weighted system EE, equal weights lambda_u = 1/U (sum_u lambda_u = 1)
        current_system_ee = np.sum(zeta_u) / self.n_agents              # raw aggregate EE (reported)
        current_system_ee_norm = np.sum(zeta_u_norm) / self.n_agents    # per-served-ED EE (used in reward)

        info['current_system_cumulative_EE'] = current_system_ee
        info['current_system_ee_norm'] = current_system_ee_norm

        omega1 = self.config['reward_weights']['omega1_association_indicator']
        omega3 = self.config['reward_weights']['omega3_system_ee_step']
        omega5 = self.config['reward_weights']['omega5_collision_penalty']
        omega6 = self.config['reward_weights']['omega6_distance_to_cs_penalty']
        omega7 = self.config['reward_weights']['omega7_cs_arrival_failure_penalty']
        omega2 = self.config['reward_weights']['omega2_coverage_bonus']   # (B)
        omega4 = self.config['reward_weights']['omega4_spread_incentive']       # (D)
        spread_cap = self.config['reward_weights']['spread_cap']           # (D)

        self.episode_sum_raw_EE_sys_at_each_step += current_system_ee
        self.episode_sum_norm_EE_sys_at_each_step += current_system_ee_norm

        nearest_other_dist = np.full(self.n_agents, spread_cap)
        if self.n_agents > 1 and omega4 != 0.0:
            for i in range(self.n_agents):
                nearest_other_dist[i] = min(
                    distance.euclidean(self.uav_positions[i][:2], self.uav_positions[j][:2])
                    for j in range(self.n_agents) if j != i)

        # --- per-agent reward ---
        for uav_idx in range(self.n_agents):
            if self.dones[uav_idx] and not uav_collided_this_step[uav_idx]:
                pass  # Agent was already done and didn't just collide

            # (B) Coverage-aware association: base indicator omega1*a_u PLUS a bonus for each
            # ED served for the FIRST time ever (system-unique) -> pulls UAVs toward unserved EDs.
            a_u_t = 1.0 if uav_had_successful_tx_this_step[uav_idx] else 0.0
            r_coverage = omega2 * per_uav_new_eds_this_step[uav_idx]
            r_association = omega1 * a_u_t + r_coverage
            current_step_rewards[uav_idx] += r_association
            info['individual_rewards_components'][uav_idx]['r_association (omega1*a_u + beta*new_eds)'] = r_association
            info['individual_rewards_components'][uav_idx]['r_coverage (beta*new_eds)'] = r_coverage

            # (D) Spread reward: rewards separation from the nearest other UAV, saturating at spread_cap.
            r_spread = omega4 * min(nearest_other_dist[uav_idx] / spread_cap, 1.0)
            current_step_rewards[uav_idx] += r_spread
            info['individual_rewards_components'][uav_idx]['r_spread (omega4*sep)'] = r_spread

            r_system_ee = omega3 * current_system_ee_norm   # reward uses per-served-ED normalized EE
            current_step_rewards[uav_idx] += r_system_ee
            info['individual_rewards_components'][uav_idx]['r_system_ee (omega3*EE_sys_norm)'] = r_system_ee

            d_safe_uav = 1.0 if uav_collided_this_step[uav_idx] else 0.0
            r_collision = -omega5 * d_safe_uav
            current_step_rewards[uav_idx] += r_collision
            info['individual_rewards_components'][uav_idx]['r_collision (-omega5*D_safe)'] = r_collision

            raw_d_cs_uav = uav_current_distances_to_cs[uav_idx]
            r_distance_cs = -omega6 * raw_d_cs_uav   # RAW d_cs (Table V omega6=2 -> matches Fig.4 scale)
            current_step_rewards[uav_idx] += r_distance_cs
            info['individual_rewards_components'][uav_idx]['r_distance_cs (-omega6*d_cs)'] = r_distance_cs

            # Accumulate weighted and raw components for get_stats
            self.episode_sum_weighted_r_association += r_association
            self.episode_sum_weighted_r_coverage += r_coverage   # (B)
            self.episode_sum_weighted_r_spread += r_spread        # (D)
            self.episode_sum_weighted_r_system_ee += r_system_ee
            self.episode_sum_weighted_r_collision += r_collision
            self.episode_sum_weighted_r_distance_cs += r_distance_cs

            self.episode_sum_raw_a_u_t += a_u_t
            self.episode_sum_raw_D_safe += d_safe_uav
            self.episode_sum_raw_d_cs += raw_d_cs_uav

            self.episode_num_agent_reward_calculations += 1
       
        # -------------------------------------

        self.timestep += 1
        for uav_idx in range(self.n_agents): 
            if self.has_reached_cs[uav_idx] and not self.dones[uav_idx]:
                 self.dones[uav_idx] = True
        
        episode_done_signal = (self.timestep >= self.max_episode_steps) or np.all(self.dones)

        if episode_done_signal:
            omega_cs_fail = self.config['reward_weights']['omega7_cs_arrival_failure_penalty']  # ω5 (Table V = 400)
            for uav_idx in range(self.n_agents):
                if not self.has_reached_cs[uav_idx]:
                    penalty_for_not_reaching_cs = -omega_cs_fail
                    current_step_rewards[uav_idx] += penalty_for_not_reaching_cs
                    # ... logging ...

                    # Log this component in info for the current (final) step
                    agent_info_comp = info['individual_rewards_components'][uav_idx]
                    agent_info_comp['r_cs_arrival_failure_penalty'] = \
                        agent_info_comp.get('r_cs_arrival_failure_penalty', 0.0) + penalty_for_not_reaching_cs

                    # Accumulate for episode-level stats (get_stats)
                    self.episode_sum_weighted_r_cs_arrival_failure += penalty_for_not_reaching_cs
                    self.episode_sum_raw_did_not_reach_cs_at_end_count += 1.0


        step_data = {
            'positions': [pos.copy() for pos in self.uav_positions],
            'speeds': [np.linalg.norm(self.uav_velocities[uav_idx]) for uav_idx in range(self.n_agents)],
            'angles': [math.degrees(math.atan2(self.uav_velocities[uav_idx][1], self.uav_velocities[uav_idx][0]))
                       if np.linalg.norm(self.uav_velocities[uav_idx]) > 1e-6 else 0 for uav_idx in range(self.n_agents)],
            'associations': [self.associations[uav_idx].nonzero()[1].tolist() for uav_idx in range(self.n_agents)], 
            'rewards_step': current_step_rewards.copy(), 
            'sf_allocations_step': self.sf_allocations_all_eds.copy(), 
            'tp_allocations_step': self.tp_allocations_all_eds.copy(), 
            'data_rate_step': {
                (uav_id, ed_idx): self.associations[uav_id, ed_idx]
                for uav_id in range(self.n_agents)
                for ed_idx in self.associations[uav_id].nonzero()[1]
            },
            # cumulative capture counts up to this step (last step == episode totals)
            'capture_stats': dict(self.capture_stats),
        }
        self.episode_data.append(step_data)

        info['capture_stats'] = dict(self.capture_stats) 

        
        return self.get_obs(), current_step_rewards.copy(), episode_done_signal, info


    # def calculate_global_ee(self):
    #     return self.episode_sum_raw_EE_sys_at_each_step / steps

    def calculate_global_ee(self):
        
        steps = max(1, self.timestep)
        return self.episode_sum_raw_EE_sys_at_each_step / steps


    def get_stats(self):
        stats = {
            'global_energy_efficiency': np.array([self.calculate_global_ee()], dtype=np.float32),
            'total_system_datarate': np.array([self.cumulative_data], dtype=np.float32),
            'total_ed_tp': np.array([self.total_ed_tx_power_episode], dtype=np.float32),
            'uavs_propulsion_power': np.array([np.sum(self.total_propulsion_power_episode)], dtype=np.float32),
            'uavs_reached_cs': np.sum(self.has_reached_cs).astype(np.float32),
            'unique_eds_served': len(self.system_unique_eds_ever),
            'uav_collisions': np.sum(self.collision_counts).astype(np.float32)   # UAV-UAV near-collisions (flight safety)
        }

        cs = self.capture_stats
        _att = max(1, cs['attempts'])
        stats['pdr']                     = np.array([cs['delivered'] / _att], dtype=np.float32)            # delivered / attempts
        stats['pkt_collision_rate']      = np.array([(cs['fail_cosf'] + cs['fail_intersf']) / _att], dtype=np.float32)
        stats['pkt_reception_fail_rate'] = np.array([cs['fail_reception'] / _att], dtype=np.float32)       # weak-link loss (not interference)
        stats['pkt_cosf_rate']           = np.array([cs['fail_cosf'] / _att], dtype=np.float32)
        stats['pkt_intersf_rate']        = np.array([cs['fail_intersf'] / _att], dtype=np.float32)
        stats['coverage_rate']           = np.array([len(self.system_unique_eds_ever) / max(1, self.num_eds)], dtype=np.float32)

        # --- Time-on-Air / duty-cycle diagnostics ---
        # Per-SF count of successful transmissions and total airtime consumed this episode.
        for sf in self.config['sf_options']:
            stats[f'tx_count_sf{sf}'] = np.array([self.tx_count_by_sf[sf]], dtype=np.float32)
            stats[f'airtime_sf{sf}']  = np.array([self.airtime_by_sf[sf]], dtype=np.float32)
        stats['tx_count_total'] = np.array([sum(self.tx_count_by_sf.values())], dtype=np.float32)

        # Per-ED duty-cycle usage = consumed airtime / max_dc_time, averaged/maxed over EDs
        # that transmitted at least once (0 if none). >1.0 in 'max' would indicate a violation.
        _usages = [sum(v) / self.max_dc_time for v in self.ed_transmission_history.values()] \
                  if self.max_dc_time > 0 else []
        stats['ed_dc_usage_mean'] = np.array([np.mean(_usages) if _usages else 0.0], dtype=np.float32)
        stats['ed_dc_usage_max']  = np.array([np.max(_usages) if _usages else 0.0], dtype=np.float32)

        num_agent_calcs = self.episode_num_agent_reward_calculations
        if num_agent_calcs == 0:
            num_agent_calcs = 1 # Avoid division by zero if episode had 0 agent steps

        sys_timesteps = self.timestep
        if sys_timesteps == 0:
            sys_timesteps = 1 # Avoid division by zero if episode had 0 timesteps

        # Average WEIGHTED reward components per agent calculation instance
        stats['avg_w_r_assoc_per_calc'] = np.array([self.episode_sum_weighted_r_association / num_agent_calcs], dtype=np.float32)
        stats['avg_w_r_ee_sys_per_calc'] = np.array([self.episode_sum_weighted_r_system_ee / num_agent_calcs], dtype=np.float32)
        stats['avg_w_r_coll_per_calc'] = np.array([self.episode_sum_weighted_r_collision / num_agent_calcs], dtype=np.float32)
        stats['avg_w_r_dist_cs_per_calc'] = np.array([self.episode_sum_weighted_r_distance_cs / num_agent_calcs], dtype=np.float32)

        # Average RAW (unweighted) component values
        stats['avg_raw_a_u_per_calc'] = np.array([self.episode_sum_raw_a_u_t / num_agent_calcs], dtype=np.float32)
        stats['avg_raw_EE_sys_per_system_step'] = np.array([self.episode_sum_raw_EE_sys_at_each_step / sys_timesteps], dtype=np.float32)
        stats['avg_norm_EE_sys_per_system_step'] = np.array([self.episode_sum_norm_EE_sys_at_each_step / sys_timesteps], dtype=np.float32)
        stats['avg_raw_D_safe_per_calc'] = np.array([self.episode_sum_raw_D_safe / num_agent_calcs], dtype=np.float32)
        stats['avg_raw_d_cs_per_calc'] = np.array([self.episode_sum_raw_d_cs / num_agent_calcs], dtype=np.float32)
        
        # Total weighted reward components for the episode (sum across all agents and steps)
        stats['total_episode_w_r_association'] = np.array([self.episode_sum_weighted_r_association], dtype=np.float32)
        stats['total_episode_w_r_coverage'] = np.array([self.episode_sum_weighted_r_coverage], dtype=np.float32)   # (B)
        stats['total_episode_w_r_spread'] = np.array([self.episode_sum_weighted_r_spread], dtype=np.float32)        # (D)
        stats['total_episode_w_r_system_ee'] = np.array([self.episode_sum_weighted_r_system_ee], dtype=np.float32)
        stats['total_episode_w_r_collision'] = np.array([self.episode_sum_weighted_r_collision], dtype=np.float32)
        stats['total_episode_w_r_distance_cs'] = np.array([self.episode_sum_weighted_r_distance_cs], dtype=np.float32)

        stats['total_episode_w_r_cs_arrival_failure'] = np.array([self.episode_sum_weighted_r_cs_arrival_failure], dtype=np.float32)
        stats['count_agents_not_reached_cs_at_end'] = np.array([self.episode_sum_raw_did_not_reach_cs_at_end_count], dtype=np.float32)
        avg_cs_failure_penalty_per_agent = self.episode_sum_weighted_r_cs_arrival_failure / self.n_agents if self.n_agents > 0 else 0.0
        stats['avg_w_r_cs_arrival_failure_per_agent_at_end'] = np.array([avg_cs_failure_penalty_per_agent], dtype=np.float32)
        
        
        return stats

    def save_trajectories(self, save_dir):
        # print(f"Saving trajectory data to: {save_dir}") 

        trajectory_data = {uav_id: [] for uav_id in range(self.n_agents)}
        for t, step_data in enumerate(self.episode_data):
            if not all(k in step_data for k in ['positions', 'speeds', 'angles', 'associations']):
                print(f"Warning: Incomplete step_data at timestep {t}. Skipping trajectory saving for this step.")
                continue
            for uav_id in range(self.n_agents):
                if not (uav_id < len(step_data['positions']) and
                        uav_id < len(step_data['speeds']) and
                        uav_id < len(step_data['angles']) and
                        uav_id < len(step_data['associations'])):
                    print(f"Warning: Missing data components for UAV {uav_id} at timestep {t}. Skipping entry.")
                    continue

                associated_eds_list = step_data['associations'][uav_id]
                entry = {
                    'timestep': t,
                    'x': step_data['positions'][uav_id][0],
                    'y': step_data['positions'][uav_id][1],
                    'speed': step_data['speeds'][uav_id],
                    'angle': step_data['angles'][uav_id],
                    'num_associated_eds': len(associated_eds_list)
                }
                trajectory_data[uav_id].append(entry)

        for uav_id, data in trajectory_data.items():
            if data: 
                file_path = os.path.join(save_dir, f"uav_{uav_id}_trajectory.csv")
                try:
                    pd.DataFrame(data).to_csv(file_path, index=False)
                except Exception as e:
                    print(f"Error saving {file_path}: {e}")
            else:
                print(f"Info: No trajectory data to save for UAV {uav_id}.")

        assoc_history_list = []
        for t, step_data in enumerate(self.episode_data):
            if not all(k in step_data for k in ['positions', 'associations']):
                 print(f"Warning: Incomplete step_data at timestep {t}. Skipping detailed assoc saving for this step.")
                 continue
            for uav_id in range(self.n_agents):
                if not (uav_id < len(step_data['positions']) and
                        uav_id < len(step_data['associations'])):
                    print(f"Warning: Missing data for UAV {uav_id} at timestep {t} in detailed assoc. Skipping entry.")
                    continue

                current_associations_list = step_data['associations'][uav_id]
                valid_ed_positions_str = ";".join([
                    f"{self.ed_positions[ed_idx][0]:.2f},{self.ed_positions[ed_idx][1]:.2f}"
                    for ed_idx in current_associations_list if ed_idx < self.num_eds 
                ])

                entry = {
                    'timestep': t,
                    'uav_id': uav_id,
                    'uav_x': step_data['positions'][uav_id][0],
                    'uav_y': step_data['positions'][uav_id][1],
                    'num_associated': len(current_associations_list),
                    'associated_eds_indices': str(current_associations_list), 
                    'associated_ed_positions': valid_ed_positions_str
                }
                assoc_history_list.append(entry)

        if assoc_history_list:
            file_path = os.path.join(save_dir, "detailed_association_history.csv")
            try:
                pd.DataFrame(assoc_history_list).to_csv(file_path, index=False)
            except Exception as e:
                print(f"Error saving {file_path}: {e}")
        else:
            print("Info: No detailed association history data to save.")

        pairwise_assoc_list = []
        for t, step_data in enumerate(self.episode_data):
            if not all(k in step_data for k in ['associations', 'sf_allocations_step', 'tp_allocations_step', 'data_rate_step']):
                 print(f"Warning: Incomplete step_data at timestep {t} for pairwise logging. Skipping.")
                 continue

            sf_state_at_t = step_data['sf_allocations_step']
            tp_state_at_t = step_data['tp_allocations_step']
            dr_state_at_t = step_data['data_rate_step'] 

            for uav_id in range(self.n_agents):
                 if uav_id >= len(step_data['associations']): continue 

                 for ed_idx in step_data['associations'][uav_id]:
                    if ed_idx >= self.num_eds:
                        print(f"Warning: Invalid ed_idx {ed_idx} found in associations for UAV {uav_id} at step {t}. Skipping pairwise entry.")
                        continue

                    sf_used = sf_state_at_t[ed_idx] if ed_idx < len(sf_state_at_t) else 'N/A'
                    tp_used = tp_state_at_t[ed_idx] if ed_idx < len(tp_state_at_t) else 'N/A'
                    dr_achieved = dr_state_at_t.get((uav_id, ed_idx), 0.0) 

                    pairwise_assoc_list.append({
                        'timestep': t,
                        'uav_id': uav_id,
                        'ed_idx': ed_idx,
                        'ed_x': self.ed_positions[ed_idx][0],
                        'ed_y': self.ed_positions[ed_idx][1],
                        'sf_used': sf_used, 
                        'tp_used_dbm': tp_used, 
                        'data_rate_bps': dr_achieved 
                    })

        if pairwise_assoc_list:
             file_path = os.path.join(save_dir, "pairwise_associations.csv")
             try:
                 pd.DataFrame(pairwise_assoc_list).to_csv(file_path, index=False)
             except Exception as e:
                 print(f"Error saving {file_path}: {e}")
        else:
             print("Info: No pairwise association data to save.")

        dc_data = []
        if not hasattr(self, 'ed_transmission_history') or not self.ed_transmission_history:
             print("Warning: ed_transmission_history not found or empty. Cannot save duty cycle data.")
        else:
            for ed_idx in range(self.num_eds):
                tx_list = self.ed_transmission_history.get(ed_idx, [])
                sf_list = self.ed_sf_history.get(ed_idx, []) if hasattr(self, 'ed_sf_history') else []
                total_tx = sum(tx_list)
                n_tx = len(tx_list)
                max_allowed_time = self.max_dc_time if hasattr(self, 'max_dc_time') else -1

                usage_ratio = 0.0
                if max_allowed_time > 0:
                    usage_ratio = total_tx / max_allowed_time
                elif total_tx > 0:
                    usage_ratio = float('inf')

                # Dominant SF = most-used SF over this ED's successful transmissions (0 if never served).
                if sf_list:
                    dominant_sf = max(set(sf_list), key=sf_list.count)
                else:
                    dominant_sf = 0

                dc_data.append({
                    'ed_id': ed_idx,
                    'total_tx_time': total_tx,
                    'max_allowed_time': max_allowed_time,
                    'usage_ratio': usage_ratio,
                    'n_tx': n_tx,
                    'dominant_sf': dominant_sf,
                    'violated': total_tx > max_allowed_time if max_allowed_time >= 0 else False
                })

        if dc_data:
            file_path = os.path.join(save_dir, "duty_cycle_summary.csv") 
            try:
                pd.DataFrame(dc_data).to_csv(file_path, index=False)
            except Exception as e:
                print(f"Error saving {file_path}: {e}")

        episode_summary_data = []
        final_stats = self.get_stats() 
        summary_entry = {
            'episode_id': self.episode_count, 
            'total_timesteps': self.timestep,
            'total_cumulative_data_bits': final_stats.get('total_system_datarate', np.array([np.nan]))[0], 
            'global_ee_bits_per_joule': final_stats.get('global_energy_efficiency', np.array([np.nan]))[0],
            'total_ed_tx_energy_joules': final_stats.get('total_ed_tp', np.array([np.nan]))[0],
            'total_uav_propulsion_energy_joules': final_stats.get('uavs_propulsion_power', np.array([np.nan]))[0],
            'num_uavs_reached_cs': final_stats.get('uavs_reached_cs', np.nan),
            'num_unique_eds_served': final_stats.get('unique_eds_served', np.nan),
            'total_collisions': final_stats.get('collisions', np.nan)
        }
        # Add the new reward component stats to the summary
        for key in [
            'avg_w_r_assoc_per_calc', 'avg_w_r_ee_sys_per_calc', 'avg_w_r_coll_per_calc', 'avg_w_r_dist_cs_per_calc',
            'avg_raw_a_u_per_calc', 'avg_raw_EE_sys_per_system_step', 'avg_raw_D_safe_per_calc', 'avg_raw_d_cs_per_calc',
            'total_episode_w_r_association', 'total_episode_w_r_system_ee', 'total_episode_w_r_collision', 'total_episode_w_r_distance_cs'
        ]:
            summary_entry[key] = final_stats.get(key, np.array([np.nan]))[0] # Extract scalar from np.array

        episode_summary_data.append(summary_entry)


        if episode_summary_data:
             file_path = os.path.join(save_dir, "episode_summary.csv")
             try:
                 pd.DataFrame(episode_summary_data).to_csv(file_path, index=False)
             except Exception as e:
                 print(f"Error saving {file_path}: {e}")
        else:
             print("Info: No episode summary data to save.")

        reward_log_list = []
        for t, step_data in enumerate(self.episode_data):
            if 'rewards_step' not in step_data:
                print(f"Warning: rewards_step missing at timestep {t}. Skipping reward logging for this step.")
                continue

            step_rewards = step_data['rewards_step'] 
            for agent_id in range(self.n_agents):
                 if agent_id < len(step_rewards):
                     reward_log_list.append({
                         'timestep': t,
                         'agent_id': agent_id,
                         'total_reward_step': step_rewards[agent_id]
                     })
                 else:
                     print(f"Warning: Missing reward data for agent {agent_id} at timestep {t}.")


        if reward_log_list:
             file_path = os.path.join(save_dir, "step_rewards.csv")
             try:
                 pd.DataFrame(reward_log_list).to_csv(file_path, index=False)
             except Exception as e:
                 print(f"Error saving {file_path}: {e}")
        else:
             print("Info: No step reward data to save.")


    def render(self, mode='human', save_path=None):
        plt.figure(figsize=(12, 12))
        plot_config = {
            'ed_color': '#2c7bb6', 'cs_color': '#d7191c',
            'trajectory_colors': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33'], 
            'trajectory_width': 2, 'start_marker': '^', 'end_marker': 's',
            'marker_size': 80, 'font_size': 12,
        }
        plt.scatter(self.ed_positions[:, 0], self.ed_positions[:, 1], c=plot_config['ed_color'], marker='x', label='End Devices', alpha=0.7)
        cs = self.cs_position
        plt.scatter(cs[0], cs[1], c=plot_config['cs_color'], marker='o', s=200, label='Charging Station')
        plt.gca().add_patch(plt.Circle((cs[0], cs[1]), self.config['exclusion_radius'], color=plot_config['cs_color'], alpha=0.1))

        for uav_id in range(self.n_agents):
            if not self.episode_data or not any(uav_id < len(step['positions']) for step in self.episode_data):
                continue 
            
            color = plot_config['trajectory_colors'][uav_id % len(plot_config['trajectory_colors'])]
            x = [step['positions'][uav_id][0] for step in self.episode_data if uav_id < len(step['positions'])]
            y = [step['positions'][uav_id][1] for step in self.episode_data if uav_id < len(step['positions'])]
            
            if not x or not y: continue 

            plt.plot(x, y, color=color, linewidth=plot_config['trajectory_width'], linestyle='-', label=f'UAV {uav_id} Path')
            plt.scatter(x[0], y[0], marker=plot_config['start_marker'], color=color, s=plot_config['marker_size'], edgecolor='k')
            plt.scatter(x[-1], y[-1], marker=plot_config['end_marker'], color=color, s=plot_config['marker_size'], edgecolor='k')

        plt.title(f"Episode {self.episode_count} - Trajectories\nUnique EDs Served: {len(self.system_unique_eds_ever)}/{self.num_eds}")
        plt.xlabel("X Coordinate (m)", fontsize=plot_config['font_size'])
        plt.ylabel("Y Coordinate (m)", fontsize=plot_config['font_size'])
        plt.legend(loc='best', framealpha=0.9); plt.grid(True, alpha=0.3)
        plt.xlim(0, self.area_size[0]); plt.ylim(0, self.area_size[1]) 
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else: plt.show()
        plt.close()

    def close(self):
        plt.close('all')
        self.episode_data.clear()
