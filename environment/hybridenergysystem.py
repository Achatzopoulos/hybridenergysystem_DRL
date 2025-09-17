from __future__ import annotations
import pdb
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eta_utility import get_logger, timeseries
from eta_utility.eta_x import ConfigOptRun
from eta_utility.eta_x.envs import BaseEnvSim, StateConfig, StateVar
try:
    from plotter.plotter import ETA_Plotter, Heatmap, Linegraph

    plotter_available = True
except ImportError as e:
    plotter_available = False


from eta_utility.type_hints import StepResult, TimeStep
from gymnasium import spaces
from scipy.special import expit

log = get_logger("eta_x.envs")
# import gym
# gym.logger.set_level(40)  # to prevent gym complaining about using just float32
# from gym import spaces
# from gym.utils import seeding
# from eta_utility.simulators import FMUSimulator
# from itertools import count
# import numpy as np
# import pandas as pd
# pd.options.mode.chained_assignment = None  # to desable SettingWithCopyWarning
# import matplotlib.pyplot as plt
# import os
# import time
# from datetime import datetime, timedelta
# import locale
# from scipy.special import expit
# import math'



class Hybridenergysystem(BaseEnvSim):
    """
    SupplysystemA environment class from BaseEnvSim.

    :param env_id: Identification for the environment, useful when creating multiple environments
    :param config_run: Configuration of the optimization run
    :param seed: Random seed to use for generating random numbers in this environment
        (default: None / create random seed)
    :param verbose: Verbosity to use for logging (default: 2)
    :param callback: Callback which should be called after each episode
    :param sampling_time: Length of a timestep in seconds
    :param episode_duration: Duration of one episode in seconds
    """
    # set info
    version = "v0.17"
    description = "(c) Heiko Ranzau, Niklas Panten and Benedikt Grosch"
    fmu_name = "hybridenergysystem"
  
    def __init__(
        self,
        render_mode: None,
        env_id: int,
        config_run: ConfigOptRun,  # config_run contains run_name, general_settings, path_settings, env_info
       #seed: int | None = None,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        sampling_time: float,  # from settings section of config file
        episode_duration: TimeStep | str,  # from settings section of config file
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        scenario_files: Sequence[Mapping[str, Any]],
        random_sampling,
        variant,
        discretize_action_space,
        reward_shaping,
        SOC_low,
        SOC_high,
        abort_costs,
        policyshaping_costs,
        power_cost_max,
        tax_el_per_kwh,
        tax_el_produced_per_kwh,
        peak_cost_per_kw,
        #energy_through_compensation, 
        #energy_after_compensation, 
        # activation_energy,
        **kwargs: Any,
        ):
        self.render_mode = render_mode
        super().__init__(
            env_id,
            config_run,
            #seed,
            verbose,
            callback,
            sampling_time=sampling_time,
            episode_duration=episode_duration,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            **kwargs,
        
    )
    
        # make variables readable class-wide
        self.random_sampling = random_sampling
        self.discretize_action_space = discretize_action_space
        self.reward_shaping = reward_shaping
        self.abort_costs = abort_costs
        self.policyshaping_costs = policyshaping_costs
        self.reward_shaping = reward_shaping
        self.SOC_low = SOC_low
        self.SOC_high = SOC_high
        self.power_cost_max = power_cost_max
        self.tax_el_per_kwh = tax_el_per_kwh
        self.tax_el_produced_per_kwh = tax_el_produced_per_kwh
        self.peak_cost_per_kw = peak_cost_per_kw
        self.variant = variant
        # self.energy_through_compensation = energy_through_compensation
        #self.energy_after_compensation = energy_after_compensation
        # self.activation_energy = activation_energy
        ###neue Init nach 21.07.2025###
        # --- Reward & cost parameters ---
        self.peak_running     = 0.0
        self.running_cost_avg = 1.0

        # Fester Arbeitspreis
        self.c_Arbeitspreis = 0.30   # €/kWh

        # Peak‐Leistungspreis (€/kW)
        self.c_power = peak_cost_per_kw

        # Degradation‐Parameter (Beispiele, anpassen falls nötig)
        self.c_BESS         = 144458.86
        self.c_KESS         = 144037.60
        self.E_maxBESS_Wh   = 122.4 * 1000   # Wh
        self.E_maxKESS_Wh   = 1.4   * 1000   # Wh
        self.n_cycles_BESS  = 6800
        self.n_cycles_KESS  = 1e6

        # SOC‐Band‐Strafen
        self.lambda_soc = 5.0

        # Optionaler Bias‐Term
        self.gamma_bias = 0.0

        # Gewichtungen (tune nach Bedarf)
        self.w_energy = 1.0
        self.w_losses = 1.0
        self.w_peak   = 1.0
        self.w_deg    = 0.5
        self.w_soc    = 0.2
        self.w_bias   = 0.0




        # check for different possible state-sizes
        extended_state = False if ("reduced_state" in variant) else True
        mpc_state = True if ("mpc_state" in variant) else False
        self.extended_predictions = False
        if "extended_predictions" in variant:
            extended_state = True
            self.extended_predictions = True

        # initialize episode statistics
        self.episode_statistics = {}
        self.episode_archive = []
        self.episode_df = pd.DataFrame()

        # set integer prediction steps (15m,1h,6h)
        self.n_steps_15m = int(900 // self.sampling_time)
        self.n_steps_1h = int(3600 // self.sampling_time)
        self.n_steps_6h = int(21600 // self.sampling_time)

        # initialize integrators and longtime stats
        self.P_el_total_15min_buffer = []
        self.P_gs_total_15min_buffer = []
        self.n_steps_longtime = 0
        self.reward_longtime_average = 0
        
         # define state variables
        state_var_tuple = (
            StateVar(
                name="u_powerbess",
                ext_id="uBESS",
                is_ext_input=True,
                is_agent_action=True,
                from_interact=True,
                interact_id="uBESS",
                low_value=-1.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_powerkess",
                ext_id="uKESS",
                is_ext_input=True,
                is_agent_action=True,
                from_interact=True,
                interact_id="uKESS",
                low_value=-1.0,
                high_value=1.0,
            ),
            # states
            StateVar(
                name="s_actual_BESS",
                ext_id="PowerActualBESS",
                is_ext_output=True,
                is_agent_observation=True,
                from_interact=True,
                interact_id="PowerActualBESS",
                low_value=-2e5,
                high_value=2e5,
            ),
                        
            StateVar(
                name="s_losses_BESS",
                ext_id="PowerLossesBESS",
                is_ext_output=True,
                is_agent_observation=True,
                from_interact=True,
                interact_id="PowerLossesBESS",
                low_value=0,
                high_value=2e6,
            ),
            StateVar(
                name="s_temp_BESS",
                ext_id="TempCell",
                is_ext_output=True,
                is_agent_observation=True,
                from_interact=True,
                interact_id="TempCell",
                low_value=-5e6,
                high_value=5e6,
            ),                       
            StateVar(
                name="s_actual_KESS",
                ext_id="PowerActualKESS",
                is_ext_output=True,
                is_agent_observation=True,
                from_interact=True,
                interact_id="PowerActualKESS",
                low_value=-5e6,
                high_value=5e6,
            ),
            StateVar(
                name="s_losses_load_KESS",
                ext_id="PowerLossesLoadKESS",
                is_ext_output=True,
                is_agent_observation=True,
                from_interact=True,
                interact_id="PowerLossesLoadKESS",
                low_value=0,
                high_value=2e6,
            ),
            StateVar(
                name="s_losses_idle_KESS",
                ext_id="PowerLossesIdleKESS",
                is_ext_output=True,
                is_agent_observation=True,
                from_interact=True,
                interact_id="PowerLossesIdleKESS",
                low_value=0,
                high_value=2e6,
            ),

            StateVar(
                name="s_SOC_BESS",
                ext_id="SoCBESS",
                is_ext_output=True,
                is_agent_observation=True,
                from_interact=True,
                interact_id="SoCBESS",
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="s_SOC_KESS",
                ext_id="SoCKESS",
                is_ext_output=True,
                is_agent_observation=True,
                from_interact=True,
                interact_id="SoCKESS",
                low_value=0,
                high_value=1,
            ),

              # to obtain €/kWh
            
            StateVar(
                name="d_usage",
                scenario_id="power_electricity",
                from_scenario=True,
                is_agent_observation=extended_state,
                low_value=-1e5,
                high_value=2e5,
            ),    
                        
        #    StateVar(
        #         name="energy_through_compensation",
        #         is_agent_observation=extended_state,
        #         low_value=-1e5,
        #         high_value=2e5,
        #     ),    
            # StateVar(
            #     name="energy_after_compensation",
            #     is_agent_observation=extended_state,
            #     low_value=-1e5,
            #     high_value=2e5,
            # ),  
            # StateVar(
            #     name="activation_energy",
            #     is_agent_observation=extended_state,
            #     low_value=-1e5,
            #     high_value=2e5,
            # ),  
                     
             # predictions

            StateVar(
                name="p_production_delta_electric_power_15m",
                is_agent_observation=extended_state,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_electric_power_1h",
                is_agent_observation=extended_state,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_electric_power_6h",
                is_agent_observation=extended_state,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_delta_price_electricity_15m",
                is_agent_observation=extended_state,
                low_value=-1e2,
                high_value=1e2,
            ),
            StateVar(name="p_delta_price_electricity_1h", is_agent_observation=True, low_value=-1e2, high_value=1e2),
            StateVar(name="p_delta_price_electricity_6h", is_agent_observation=True, low_value=-1e2, high_value=1e2),
            # virtual states
            StateVar(
                name="vs_electric_power_total_15min", is_agent_observation=True, low_value=-100000, high_value=500000
            ),

            StateVar(name="vs_time_daytime", is_agent_observation=True, low_value=0, high_value=24),
            # states
            StateVar(
                name="s_price_electricity",
                scenario_id="electrical_energy_price",
                from_scenario=True,
                is_agent_observation=True,
                low_value=-10,
                high_value=10,
            ),
            
        )
        # add predictions when extended predictions are required
        if self.extended_predictions:
            predictions_state_var_tuple = ()
            for i in range(self.n_steps_6h):
                predictions_state_var_tuple = (
                    *predictions_state_var_tuple,
                    StateVar(
                        name="p_production_delta_electric_power_t_plus_" + str(i + 1),
                        is_agent_observation=extended_state,
                        low_value=-2e6,
                        high_value=2e6,
                    ),
                )
            for i in range(self.n_steps_6h):
                predictions_state_var_tuple = (
                    *predictions_state_var_tuple,
                    StateVar(
                        name="p_delta_price_electricity_t_plus_" + str(i + 1),
                        is_agent_observation=extended_state,
                        low_value=-1e2,
                        high_value=1e2,
                    ),
                )

            state_var_tuple = (*state_var_tuple, *predictions_state_var_tuple)

        # build final state_config
        self.state_config = StateConfig(*state_var_tuple)
        #print("The selected observations are: ", self.state_config.observations)

        # Initiate the simulator
        self._init_simulator()

        # --- import all scenario files ---
        # ALT:
        # self.scenario_data = self.import_scenario(*scenario_files)

        # NEU: Prefixing sicherheitshalber aus
        self.scenario_data = self.import_scenario(*scenario_files, prefix_renamed=False)

        # --- Spalten-Aliasse in scenario_data sicherstellen ---
        def _ensure_col(df, canonical, candidates):
            if canonical in df.columns:
                return
            for c in candidates:
                if c in df.columns:
                    df[canonical] = df[c]
                    return
            # Suffix-Match (z. B. "prod_power_electricity")
            for c in df.columns:
                if isinstance(c, str) and c.endswith(canonical):
                    df[canonical] = df[c]
                    return
            # Wenn hier gelandet: später in reset() noch ein Versuch mit ts_current
            # (nicht hart failen, weil Subprozesse sonst wenig Logs zeigen)

        _ensure_col(self.scenario_data, "power_electricity",
                    ["P_el", "electric_power", "electricity_demand", "P_grid"])
        _ensure_col(self.scenario_data, "electrical_energy_price",
                    ["el_price", "price_el", "electricity_price"])


        # get action_space
        # TODO: implement this functionality into utility functions
        if self.discretize_action_space:
            # get number of actions agent has to give from state_config
            self.n_action_space = len(self.state_config.actions)
            # set 3 discrete actions (increase,decrease,equal) per control variable
            self.action_space = spaces.MultiDiscrete(np.full(self.n_action_space, 3))
            # customized for KESS, BESS
            self.action_disc_step = [[-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
            # initialize action
            self.action_disc_index = [0] * self.n_action_space
        else:
            self.action_space = self.state_config.continuous_action_space()

        # get observation_space (always continuous)
        self.observation_space = self.state_config.continuous_obs_space()

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.
        """
        if self.render_mode == "human":
            self.render()

        # initialize additional_state and create state backup
        self.state_backup = self.state.copy()
        self.additional_state = {}

        # convert discrete actions into continious space if discrete action space is chosen
        if self.discretize_action_space:
            _action = self.convert_disc_action(action)
        else:
            _action = action
             
 
        # overwrite actions if out of boundaries (policy shaping), values are explicitly written for logging purposes
        self.policy_shaping_active = False
        if self.state['d_usage'] > self.state['p_production_delta_electric_power_1h'] + 130000 :
            self.state['u_powerbess'] = -1
            self.state['u_powerkess'] = -1
            _action = np.array(
                [
                    self.additional_state.get("u_powerbess",0),
                    self.additional_state.get("u_powerkess",0),
                ]
            )
        self.policy_shaping_active = True                   
    
        # check actions for vilidity, perform simulation step and load new external values for the next time step
        self._actions_valid(_action)
        self.state["step_success"], _ = self._update_state(_action)

        # check if state is in valid boundaries
        self.state["step_abort"] = False if StateConfig.within_abort_conditions(self.state_config, self.state) else True

        # update predictions and virtual state for next time step
        self.state.update(self.update_predictions())
        self.state.update(self.update_virtual_state())

        # check if episode is over or not
        done = self._done() or not self.state["step_success"]
        done = done if not self.state["step_abort"] else True

        # calculate reward
        reward = self.calc_reward()

#         # update state_log
#         self.state_log.append(self.state)

# #         observations = self._observations()

# #         return observations, reward, done, {}
#         observations, _, terminated, truncated, info = super().step(action)
#         self.episode_reward -= abs(self.state["s"])
#         return observations, self.episode_reward, terminated, truncated, info
    
            # update state_log
        self.state_log.append(self.state)

        observations = self._observations()

        return observations, reward, done, False, {}
    
    def update_predictions(self):

        prediction_dict = {}

        # [W] electric power demand of production
        prediction_dict["p_production_delta_electric_power_15m"] = (
            self.ts_current["power_electricity"].iloc[self.n_steps : self.n_steps + self.n_steps_15m].mean()
            - self.ts_current["power_electricity"].iloc[self.n_steps]
        )
        prediction_dict["p_production_delta_electric_power_1h"] = (
            self.ts_current["power_electricity"].iloc[self.n_steps : self.n_steps + self.n_steps_1h].mean()
            - self.ts_current["power_electricity"].iloc[self.n_steps]
        )
        prediction_dict["p_production_delta_electric_power_6h"] = (
            self.ts_current["power_electricity"].iloc[self.n_steps : self.n_steps + self.n_steps_6h].mean()
            - self.ts_current["power_electricity"].iloc[self.n_steps]
        )

        # electricity price [€/kWh]
        prediction_dict["p_delta_price_electricity_15m"] = (
            self.ts_current["electrical_energy_price"].iloc[self.n_steps : self.n_steps + self.n_steps_15m].mean()
            - self.ts_current["electrical_energy_price"].iloc[self.n_steps]
        )
        prediction_dict["p_delta_price_electricity_1h"] = (
            self.ts_current["electrical_energy_price"].iloc[self.n_steps : self.n_steps + self.n_steps_1h].mean()
            - self.ts_current["electrical_energy_price"].iloc[self.n_steps]
        )
        prediction_dict["p_delta_price_electricity_6h"] = (
            self.ts_current["electrical_energy_price"].iloc[self.n_steps : self.n_steps + self.n_steps_6h].mean()
            - self.ts_current["electrical_energy_price"].iloc[self.n_steps]
        )

        # add  predictions when extended predictions are required
        if self.extended_predictions:

            for i in range(self.n_steps_6h):
                prediction_dict["p_production_delta_electric_power_t_plus_" + str(i + 1)] = (
                    self.ts_current["power_electricity"].iloc[self.n_steps + i + 1]
                    - self.ts_current["power_electricity"].iloc[self.n_steps]
                )
            for i in range(self.n_steps_6h):
                prediction_dict["p_delta_price_electricity_t_plus_" + str(i + 1)] = (
                    self.ts_current["electrical_energy_price"].iloc[self.n_steps + i + 1]
                    - self.ts_current["electrical_energy_price"].iloc[self.n_steps]
                )
        
       
    #print(prediction_dict)

        return prediction_dict
    
    def update_virtual_state(self):

        virtual_state = {}

        # daytime
        virtual_state["vs_time_daytime"] = (
            self.ts_current.index[self.n_steps].hour + self.ts_current.index[self.n_steps].minute / 60
        )

        # running 15min average electric power  # TODO: replace by using state_log!
        self.P_el_total_15min_buffer.append(self.state["d_usage"])
        if len(self.P_el_total_15min_buffer) > self.n_steps_15m:
            self.P_el_total_15min_buffer.pop(0)
        virtual_state["vs_electric_power_total_15min"] = sum(self.P_el_total_15min_buffer) / len(
            self.P_el_total_15min_buffer
        )

        return virtual_state

    def calc_reward(self):
        # Zeitschritt
        dt   = self.sampling_time
        dt_h = dt / 3600.0

        # Leistungen & Verluste [W]
        P_bess = self.state['s_actual_BESS']
        P_kess = self.state['s_actual_KESS']
        losses = self.state['s_losses_BESS'] + self.state['s_losses_idle_KESS']+ self.state['s_losses_load_KESS']
        
        # Netzbezug [W] (positiv=Bezug, negativ=Einspeisung)
        P_net = self.state['d_usage'] - (P_bess + P_kess) + losses
        self.state['energy_after_compensation'] = P_net
        self.state['energy_through_compensation'] = self.state['d_usage'] - P_net

        # 1) Arbeitspreis‐Kosten (nur bezogene Energie)
        energy_kWh = max(0.0, P_net) * dt_h
        cost_energy = energy_kWh * self.c_Arbeitspreis

        # 2) Verlustkosten separat (optional, da schon in P_net enthalten)
        cost_losses = losses * self.c_Arbeitspreis * dt_h

        # 3) Peak‐Kosten (15‑Min rollend)
        self.P_el_total_15min_buffer.append(P_net)
        if len(self.P_el_total_15min_buffer) > self.n_steps_15m:
            self.P_el_total_15min_buffer.pop(0)
        barP15    = sum(self.P_el_total_15min_buffer) / len(self.P_el_total_15min_buffer)
        old_peak  = self.peak_running
        self.peak_running = max(old_peak, barP15)
        peak_increment = max(0.0, self.peak_running - old_peak)
        cost_peak = self.c_power * peak_increment

        # 4) Degradation (Capex/Cycle)
        thr_bess_Wh = abs(P_bess) * dt_h
        thr_kess_Wh = abs(P_kess) * dt_h
        cost_deg = (
            self.c_BESS * thr_bess_Wh / (self.E_maxBESS_Wh * self.n_cycles_BESS)
        + self.c_KESS * thr_kess_Wh / (self.E_maxKESS_Wh * self.n_cycles_KESS)
        )

        # 5) SOC‐Band‐Strafe (quadratisch)
        def soc_pen(soc, low, high):
            if soc < low:  return (low - soc)**2
            if soc > high: return (soc - high)**2
            return 0.0
        cost_soc = self.lambda_soc * (
            soc_pen(self.state['s_SOC_BESS'], self.SOC_low, self.SOC_high)
        + soc_pen(self.state['s_SOC_KESS'], self.SOC_low, self.SOC_high)
        )

        # 6) Policy‐ & Abort‐Kosten
        cost_policy = 0.0
        if self.state['step_abort'] or not self.state['step_success']:
            cost_policy += self.abort_costs
        if getattr(self, 'policy_shaping_active', False):
            cost_policy += self.policyshaping_costs

        # 7) Optionaler Bias‐Term
        cost_bias = self.gamma_bias * (abs(P_bess) + abs(P_kess)) * dt_h

        # 8) Gesamt‐Kosten mit Gewichtungen
        step_cost = (
            self.w_energy * cost_energy
        + self.w_losses * cost_losses
        + self.w_peak   * cost_peak
        + self.w_deg    * cost_deg
        + self.w_soc    * cost_soc
        + self.w_bias   * cost_bias
        + cost_policy
        )

        # 9) Laufende Normalisierung (optional)
        self.running_cost_avg = 0.999 * self.running_cost_avg + 0.001 * step_cost

        # 10) Reward (negativ der Kosten)
        reward = - step_cost

        # 11) Logging aller Teil‐Komponenten
        self.state['cost_energy'] = cost_energy
        self.state['cost_losses'] = cost_losses
        self.state['cost_peak']   = cost_peak
        self.state['cost_deg']    = cost_deg
        self.state['cost_soc']    = cost_soc
        self.state['cost_policy'] = cost_policy
        self.state['reward_total'] = reward

        return reward

    
        # if self.normalize_reward:
        #     # total normalized costs (normalized by cumulative moving average)
        #     self.reward_longtime_average += (self.state['reward_total'] - self.reward_longtime_average) / self.n_steps_longtime
        #     self.state['reward_total_norm'] = self.state['reward_total'] / abs(self.reward_longtime_average)
        #     # return normalized reward
        #     return self.state['reward_total_norm']
        # else :
        #     self.state['reward_total_norm'] = 0
        #     return self.state['reward_total']


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> np.ndarray:
            # 0) Debug + Defaults
        # print(">> custom reset called")
        if not hasattr(self, "state_log"): self.state_log = []
        if not hasattr(self, "episode_statistics"): self.episode_statistics = {}
        if not hasattr(self, "episode_archive"): self.episode_archive = []
        if self.render_mode == "human":
            self.render()

        # delete long time storage, since it takes up too much memory during training
        self.state_log_longtime = []
        

        # save episode's stats
        if self.n_steps > 0 :
        
            print("n_steps not zero")

        # --- vorherige Episode nur dann auswerten, wenn es auch eine gab ---
        if getattr(self, "n_steps", 0) > 0 and hasattr(self, "episode_datetime_begin") and len(getattr(self, "state_log", [])) > 0:
            self.episode_df = pd.DataFrame(self.state_log)

            # Nur schreiben, wenn Feld existiert
            self.episode_statistics['datetime_begin'] = self.episode_datetime_begin.strftime('%Y-%m-%d %H:%M')
            self.episode_statistics['rewards_total'] = self.episode_df.get('reward_total', pd.Series(dtype=float)).sum()
            # Falls du diese Keys (reward_*) noch nicht füllst, nimm get(..., 0)
            self.episode_statistics['rewards_batteryaging'] = self.episode_df.get('cost_deg', pd.Series(dtype=float)).sum()
            self.episode_statistics['rewards_energy_loss'] = self.episode_df.get('cost_losses', pd.Series(dtype=float)).sum()
            self.episode_statistics['rewards_energy'] = self.episode_df.get('cost_energy', pd.Series(dtype=float)).sum()
            self.episode_statistics['rewards_peak'] = self.episode_df.get('cost_peak', pd.Series(dtype=float)).sum()
            self.episode_statistics['rewards_soc'] = self.episode_df.get('cost_soc', pd.Series(dtype=float)).sum()


            # Diese Spalten hast du evtl. (noch) nicht – daher defensiv:
            self.episode_statistics['energy_through_compensation'] = self.episode_df.get('energy_through_compensation', pd.Series(dtype=float)).sum()/3600
            self.episode_statistics['energy_after_compensation']   = self.episode_df.get('energy_after_compensation',   pd.Series(dtype=float)).sum()/3600

            self.episode_statistics['n_steps'] = self.n_steps
            self.episode_statistics["time"] = time.time() - self.episode_timer if hasattr(self, "episode_timer") else 0.0
            self.episode_statistics['average_SOC_BESS']  = self.episode_df.get('s_SOC_BESS', pd.Series(dtype=float)).mean()
            self.episode_statistics['average_SOC_KESS']  = self.episode_df.get('s_SOC_KESS', pd.Series(dtype=float)).mean()
            self.episode_statistics['charge_cycle_KESS'] = self.episode_df.get('s_SOC_KESS', pd.Series(dtype=float)).diff().abs().sum()
            self.episode_statistics['charge_cycle_BESS'] = self.episode_df.get('s_SOC_BESS', pd.Series(dtype=float)).diff().abs().sum()
            self.episode_statistics['compensated']       = self.episode_df.get('cost_peak', pd.Series(dtype=float)).sum()
            #self.episode_statistics['max_last']          = self.episode_df.get('energy_after_compensation', pd.Series(dtype=float)).max()

            self.episode_archive.append(list(self.episode_statistics.values()))

            
        # initialize additional state, since super().reset is used
        self.additional_state = {}
        ###neu nach dem 21.07.2025###
        # self.state['cost_energy'] = cost_energy
        # self.state['cost_peak'] = cost_peak
        # self.state['reward_total'] = reward
        self.peak_running            = 0.0
        self.running_cost_avg        = 1.0
        self.P_el_total_15min_buffer = []
        # set new FMU parameters and state for initialization
        self.model_parameters = {}
        self.model_parameters['init_SOC_BESS'] = self.np_random.uniform(self.SOC_low, self.SOC_high)
        self.model_parameters['init_SOC_KESS'] = self.np_random.uniform(self.SOC_low, self.SOC_high)
        #self.model_parameters['T_start_warmwater'] = self.np_random.uniform(self.env_settings['temperature_heat_init_min'], self.env_settings['temperature_heat_init_max'])
        #_ = self.np_random.uniform(self.env_settings['temperature_heat_init_min'], self.env_settings['temperature_heat_init_max'])  # necessary to get the same number of random calls to generate the same dates and temperatures for system a and b
        #self.state['s_temp_heat_storage_hi'] = self.model_parameters['T_start_warmwater']
        #self.state['s_temp_heat_storage_lo'] = self.model_parameters['T_start_warmwater']-self.env_settings['temperature_difference_top_bottom']
        #self.state['s_temp_heatsupply'] = self.model_parameters['T_start_warmwater']

        # get current slice of timeseries dataframe, extended by maximum prediction horizon (6h)
        # and one additional step because step 0 = init conditions
        self.ts_current = timeseries.df_time_slice(
            self.scenario_data,
            self.scenario_time_begin,
            self.scenario_time_end,
            self.episode_duration + (self.n_steps_6h + 1) * self.sampling_time,
            random=self.np_random if self.random_sampling else False,
        )
        #----------------------------------------
        def _ensure_ts_col(ts_df, canonical):
                if canonical in ts_df.columns: return
                if len(ts_df.columns) == len(self.scenario_data.columns) and not isinstance(ts_df.columns[0], str):
                    ts_df.columns = list(self.scenario_data.columns)
                    if canonical in ts_df.columns: return
                for c in ts_df.columns:
                    if isinstance(c, str) and c.endswith(canonical):
                        ts_df[canonical] = ts_df[c]; return
                if canonical in self.scenario_data.columns:
                    ts_df[canonical] = (
                        self.scenario_data[canonical]
                        .reindex(ts_df.index)
                        .interpolate(method="time", limit_direction="both")
                    ); return
                raise KeyError(f"'{canonical}' fehlt in ts_current. Hat: {list(ts_df.columns)[:40]}")
        _ensure_ts_col(self.ts_current, "power_electricity")
        _ensure_ts_col(self.ts_current, "electrical_energy_price")
        #---------------------------------


        # read current date time
        self.episode_datetime_begin = self.ts_current.index[0]
        self.additional_state["vs_time_daytime"] = self.episode_datetime_begin.hour

        # reset virtual states and internal counters
        self.P_el_total_15min_buffer = []
        self.additional_state["vs_electric_power_total_15min"] = 0


        # get scenario input for initialization (time step: 0)
        self.additional_state.update(self.update_predictions())

        # reset maximal peak electric power for penalty costs (necessary for peak shaving)
        self.max_limit = self.power_cost_max
        


        # receive observations from simulation
        return super().reset(seed=seed, options=options)

    def convert_disc_action(self, action_disc):
        """
        converts discrete actions from agent to continious FMU input space
        """
        float_action = []

        for idx, val in enumerate(action_disc):
            self.action_disc_index[idx] = np.clip(
                self.action_disc_index[idx] + (val - 1), 0, len(self.action_disc_step[idx]) - 1
            )
            float_action.append(self.action_disc_step[idx][self.action_disc_index[idx]])

        return np.array(float_action)


    def render_episodes(self):
        """
        output plot for all episodes
        see pandas visualization options on https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
        https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

        Parameters
        -----
        mode : (str)
        """
        # create dataframe
        episode_archive_df = pd.DataFrame(self.episode_archive, columns=list(self.episode_statistics.keys()))

        # write all data to csv after every episode
        episode_archive_df.to_csv(
            path_or_buf=os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(4)
                + "-"
                + str(self.env_id).zfill(2)
                + "_all-episodes.csv",
            ),
            sep=";",
            decimal=".",
        )

        # write another aggregated csv that contains all episodes (necessary for mpc and mpc_simple)
        csvpath = os.path.join(self.path_results, "all-episodes.csv")
        if os.path.exists(
            csvpath
        ):  # check if aggregated file already exists, which is the case when multiple runs are done with mpc and mpc_simple
            tocsvmode = "a"
            tocsvheader = False
        else:
            tocsvmode = "w"
            tocsvheader = True
        # write data to csv
        episode_archive_df.tail(1).to_csv(path_or_buf=csvpath, sep=";", decimal=".", mode=tocsvmode, header=tocsvheader)

        # plot settings
        # create figure and axes with custom layout/adjustments
        figure = plt.figure(figsize=(14, 14), dpi=200)
        axes = []
        axes.append(figure.add_subplot(2, 1, 1))
        axes.append(figure.add_subplot(2, 1, 2, sharex=axes[0]))
        
        # fig, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=200)
        x = np.arange(len(episode_archive_df.index))
        y = episode_archive_df

        # (1) Costs
        axes[0].plot(x, y['rewards_energy_loss'], label='Verluste', color=(1.0,0.75,0.0), linewidth=1, alpha=0.9)
        axes[0].plot(x, y['rewards_energy'], label='Arbeitspreis', color=(0.929,0.49,0.192), linewidth=1, alpha=0.9)
        axes[0].plot(x, y['rewards_peak'], label='Spitzenlast', color=(0.929,0.49,0.192), linewidth=1, alpha=0.9)
        #axes[0].plot(x, y['rewards_energy_electric'], label='Steuern & Umlagen', color=(0.184,0.333,0.592), linewidth=1, alpha=0.9)
        axes[0].plot(x, y['rewards_batteryaging'], label='Ladezyklen_Batterie', color=(0.65,0.65,0.65), linewidth=1, alpha=0.9)
        axes[0].set_ylabel('kum. Kosten')
        axes[0].set_xlabel('Episode')
        axes[0].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        axes[0].margins(x=0.0,y=0.1)
        axes[0].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)

        # (2) Rewards
        cost_total = (
            y["rewards_energy_loss"]
            + y["rewards_batteryaging"]
            + y["rewards_energy"]
            + y["rewards_peak"]
        )
        axes[1].plot(x, cost_total, label="Kosten", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9)
        axes[1].plot(
            x, y["rewards_batteryaging"], label="Ladezyklen_Batterie", color=(0.75, 0, 0), linewidth=1, alpha=0.9
        )
        axes[1].plot(x, y["rewards_energy"], label="Kompensation", color=(0.44, 0.19, 0.63), linewidth=1, alpha=0.9)
        axes[1].plot(x, y["rewards_peak"], label="Kompensation", color=(0.44, 0.19, 0.63), linewidth=1, alpha=0.9)
        axes[1].plot(x, y["rewards_total"], label="Gesamt", color=(0.1, 0.1, 0.1), linewidth=2)
        axes[1].set_ylabel("kum. + fikt. Kosten [€]")
        axes[1].set_xlabel("Episode")
        axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[1].margins(x=0.0, y=0.1)
        axes[1].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="solid", linewidth=1)

        plt.savefig(
            os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(4)
                + "-"
                + str(self.env_id).zfill(2)
                + "_all-episodes.png",
            )
        )
        plt.close(figure)

    def import_scenario(self, *scenario_paths: Mapping[str, Any], prefix_renamed: bool = True) -> pd.DataFrame:
        paths = []
        prefix = []
        int_methods = []
        scale_factors = []
        rename_cols = {}
        infer_datetime_from = []
        time_conversion_str = []

        for path in scenario_paths:
            paths.append(self.path_scenarios / path["path"])
            prefix.append(path.get("prefix", None))
            int_methods.append(path.get("interpolation_method", None))
            scale_factors.append(path.get("scale_factors", None))
            rename_cols.update(path.get("rename_cols", {})),
            infer_datetime_from.append(path.get("infer_datetime_cols", "string"))
            time_conversion_str.append(path.get("time_conversion_str", "%Y-%m-%d %H:%M"))

        return timeseries.scenario_from_csv(
            paths=paths,
            resample_time=self.sampling_time,
            start_time=self.scenario_time_begin,
            end_time=self.scenario_time_end,
            random=False,
            interpolation_method=int_methods,
            scaling_factors=scale_factors,
            rename_cols=rename_cols,
            prefix_renamed=prefix_renamed,
            infer_datetime_from=infer_datetime_from,
            time_conversion_str=time_conversion_str,
        )


    #def render(self, mode='human', name_suffix=''):
    def render(self,  name_suffix=''):
        mode = self.render_mode
        """
        output plots for last episode
        see pandas visualization options on https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
        https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

        Parameters
        -----
        mode : (str)
        """

        # save csv
        self.episode_df.to_csv(
            path_or_buf=os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(3)
                + "-"
                + str(self.env_id).zfill(2)
                + "_episode"
                + name_suffix
                + ".csv",
            ),
            sep=";",
            decimal=".",
        )
        

        # create figure and axes with custom layout/adjustments
        figure = plt.figure(figsize=(14, 26), dpi=200)
        axes = []
        axes.append(figure.add_subplot(7, 1, 1))
        axes.append(figure.add_subplot(7, 1, 2, sharex=axes[0]))
        axes.append(figure.add_subplot(7, 1, 3, sharex=axes[0]))
        axes.append(figure.add_subplot(7, 1, 4, sharex=axes[0]))
        axes.append(figure.add_subplot(7, 1, 5, sharex=axes[0]))
        axes.append(figure.add_subplot(7, 1, 6, sharex=axes[0]))
        axes.append(figure.add_subplot(7, 1, 7, sharex=axes[0]))
        
        plt.tight_layout()
        figure.subplots_adjust(left=0.125, bottom=0.05, right=0.9, top=0.95, wspace=0.2, hspace=0.05)

        # set x/y axe and datetime begin
        x = self.episode_df.index
        y = self.episode_df
        dt_begin = self.episode_datetime_begin
        sampling_time = self.sampling_time

        # (1) - Plot actions as heatmap
        axes[0].set_yticks(np.arange(len(self.state_config.actions)))
        axes[0].set_yticklabels(['BESS', 'KESS'])
        im = axes[0].imshow(y[self.state_config.actions].transpose(), cmap="hot", vmin=-1, vmax=1, aspect="auto", interpolation="none")
        # add colorbar
        ax_pos = axes[0].get_position().get_points().flatten()
        ax_colorbar=figure.add_axes([0.93,ax_pos[1]+0.05,0.01,ax_pos[3]-ax_pos[1]-0.1])  ## the parameters are the specified position you set 
        figure.colorbar(im, ax=axes[0], shrink=0.9, cax=ax_colorbar)

        timeRange = np.arange((1-dt_begin.minute/60)*60*60/sampling_time, self.episode_duration/sampling_time, 1*60*60/sampling_time,)
        dt_begin = dt_begin.replace(microsecond=0,second=0,minute=0)
        ticknames = []
        tickpos = []
        for i in timeRange :
                tickdate = dt_begin + timedelta(seconds=i*sampling_time)
                if tickdate.hour in [6, 12, 18] :
                        tickpos.append(i)
                        ticknames.append(tickdate.strftime('%H'))
                elif tickdate.hour == 0 :
                        tickpos.append(i)
                        ticknames.append(tickdate.strftime("%d.%m.'%y"))
        # Let the horizontal axes labeling appear on top
        axes[0].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, rotation=45)
        axes[0].set_xlabel("Zeit (UTC)")
        axes[0].xaxis.set_label_position('top')
        # ax.set_xticks(np.arange(df1.shape[1]+1)-.5, minor=True)
        axes[0].set_yticks(np.arange(len(self.state_config.actions) + 1) - 0.5, minor=True)
        axes[0].tick_params(which="minor", bottom=False, left=False)
        # grid settings
        axes[0].grid(which="minor", color="w", linestyle='solid', linewidth=3)
        axes[0].xaxis.grid(color=(1,1,1,0.1), linestyle='solid', linewidth=1)
        # add ticks and tick labels
        axes[0].set_xticks(tickpos)
        axes[0].set_xticklabels(ticknames)
        # Rotate the tick labels and set their alignment.
        plt.setp(axes[0].get_yticklabels(), rotation=30, ha="right", va="center", rotation_mode="anchor")

        # (2) - Lasten
        axes[1].plot(x,y['d_usage'], color='#1f77b4', label='Lasten')
        axes[1].plot(x,y['energy_after_compensation'], color='#ff7f0e', label='Lasten mit Kompensation',  alpha=0.6)
        #axes[1].plot(x, ['activation_energy']* len(x), color='grey', linestyle='dashed', linewidth=1, alpha=0.5, label='Lastgrenze')
        #axes[1].plot(x, 20000 * len(x), color='red', linestyle='dashed', linewidth=1, alpha=0.5, label='Lastgrenze regelbasiert ')
        #axes[1].plot(x, 5000 * len(x), color='grey', linestyle='dashed', linewidth=1, alpha=0.5, label='Lastgrenze regelbasiert')
        # settings
        axes[1].set_ylabel('Leistung [W]')
        axes[1].margins(x=0.0,y=0.1)
        axes[1].set_axisbelow(True)
        axes[1].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)
        axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
       
        axes[1].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        # (3) - reward_compansated
        axes[2].plot(x,y['s_actual_KESS'], color='#1f77b4', label='Leistung KESS')
        axes[2].plot(x,y['s_actual_BESS'], color='#ff7f0e', label='Leistung BESS')
        #axes[2].plot(x,y['reward_energy_loss'], color=(0,0,192/255), label='Leistungsverluste')
        # settings
        axes[2].set_ylabel('Leistung [W]')
        axes[2].margins(x=0.0,y=0.1)
        axes[2].set_axisbelow(True)
        axes[2].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)
        axes[2].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        axes[2].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)

# (3) - reward_compansated
        axes[3].plot(x,y['s_SOC_KESS'], color='#1f77b4', label='SOC KESS')
        axes[3].plot(x,y['s_SOC_BESS'], color='#ff7f0e', label='SOC BESS')
        # settings
        axes[3].set_ylabel('Ladestand')
        axes[3].margins(x=0.0,y=0.1)
        axes[3].set_axisbelow(True)
        axes[3].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)
        axes[3].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        axes[3].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)

# (4) - batteryusage 
        # axes[4].plot(x, y['s_SOC_BESS'].diff().abs().cumsum(), color=(192/255,0,0), label='laden BESS')
        # axes[4].plot(x, y['s_SOC_KESS'].diff().abs().cumsum(), color=(0,192/255,0), label='laden KESS')
      
        # axes[4].set_ylabel('Ladezyklen []')
        # axes[4].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        # axes[4].margins(x=0.0,y=0.1)
        # axes[4].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        # axes[4].set_axisbelow(True)
        # axes[4].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)
# (4) - kompensiert
        axes[4].plot(x, y['energy_after_compensation'].abs().cumsum()/3600, color='#1f77b4', label=' - mit speicher')
        axes[4].plot(x, y['d_usage'].abs().cumsum()/3600, color='#1f77b4', label=' - ohne speicher')
        axes[4].plot(x, y['energy_through_compensation'].cumsum()/3600 , color='gray', label='Differenz')
       
        axes[4].set_ylabel('zu kompensierende Leistung [Wh]')
        axes[4].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        axes[4].margins(x=0.0,y=0.1)
        axes[4].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        axes[4].set_axisbelow(True)
        axes[4].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)

        #axes[5].plot(x,y['s_losses_KESS'], color='#1f77b4', label='Leistung KESS')
        # axes[5].plot(x,y['s_losses_BESS'], color='#ff7f0e', label='Leistung BESS')
        # #axes[2].plot(x,y['reward_energy_loss'], color=(0,0,192/255), label='Leistungsverluste')
        # # settings
        # axes[5].set_ylabel('Leistung [W]')
        # axes[5].margins(x=0.0,y=0.1)
        # axes[5].set_axisbelow(True)
        # axes[5].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)
        # axes[5].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        # axes[5].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        textverlust='Sum_Verluste BESS: ' + str(round(self.episode_df['s_losses_BESS'].sum()/3600,2))+'Wh / Sum_Verluste_KESS: ' + str(round(self.episode_df['s_losses_load_KESS'].sum()/3600,2))+'Wh' + str(round(self.episode_df['s_losses_idle_KESS'].sum()/3600,2))+'Wh'
        texttemp=' / BESS temp.:'+ str(round(self.episode_df['s_temp_BESS'].mean(),2)) +'C°'
        textzyklen=' / BESS Zyklen:'+ str(round(self.episode_df['s_SOC_BESS'].diff().abs().sum(),2))
        textausgabe=textverlust + texttemp + textzyklen
        axes[5].text(0.5,0.5,textausgabe, ha='left', va='center', fontsize=13, color='black')
        
# (5) - kompensiert

        # axes[6].plot(x, y['reward_total'] , color='#1f77b4', label='reward gesamt')
        # axes[6].plot(x, y['reward_energy_compensated'] , color='#ff7f0e', label='reward compensation', alpha=0.6)
        # axes[6].plot(x, y['reward_other'] , color='grey', label='reward other')
        # axes[6].set_ylabel('[-]')
        # axes[6].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        # axes[6].margins(x=0.0,y=0.1)
        # axes[6].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        # axes[6].set_axisbelow(True)
        # axes[6].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)

        # (4) - Plot power
        #axes[4].plot(x, y['s_electric_power_total']*1e-3, color=(1.0,0.75,0.0), linewidth=2, alpha=0.5, label='Strom Netz')
        #axes[3].plot(x, y['vs_electric_power_total_15min']*1e-3, color=(1.0,0.75,0.0), linewidth=2, alpha=0.9, label='Strom Netz (Ø15m)')
        #axes[3].plot(x, y['vs_gas_power_total_15min']*1e-3, color=(0.65,0.65,0.65), linewidth=2, alpha=0.9, label='Erdgas Netz (Ø15m)')
        #axes[3].plot(x, y['d_production_heat_power']*1e-3, color=(0.75,0,0), linestyle='dashed', linewidth=1, alpha=0.9, label='Wärmelast Prod.')
        #axes[3].plot(x, y['d_production_electric_power']*1e-3, color=(1.0,0.75,0.0), linestyle='dashed', linewidth=1, alpha=0.9, label='Strom Prod.')
        #axes[3].plot(x, y['d_production_gas_power']*1e-3, color=(0.65,0.65,0.65), linestyle='dashed', linewidth=1, alpha=0.9, label='Erdgas Prod.')
        #axes[3].fill_between(x, (
            #y['s_electric_power_total']-y['d_production_electric_power'])*1e-3,
            #label='Strom TGA',
            #color=(1.0,0.75,0.0), linestyle='dashed', linewidth=0, alpha=0.4)
        #axes[3].fill_between(x, (
          #  y['s_gas_power_total']-y['d_production_gas_power'])*1e-3,
           # label='Erdgas TGA',
            #color=(0.65,0.65,0.65), linestyle='dashed', linewidth=0, alpha=0.4)

        #axes[3].set_ylabel('Leistung [kW]')
        #axes[3].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        #axes[3].margins(x=0.0,y=0.1)
        #axes[3].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        #axes[3].set_axisbelow(True)
        #axes[3].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)


        # (5) - Costs
        #axes[5].plot(x, y['reward_energy_electric'].cumsum(), label='Strom (netto)', color=(1.0,0.75,0.0), linewidth=1, alpha=0.9)
        #axes[4].plot(x, y['reward_energy_gas'].cumsum(), label='Erdgas (netto)', color=(0.65,0.65,0.65), linewidth=1, alpha=0.9)
        #axes[4].plot(x, y['reward_energy_taxes'].cumsum(), label='Steuern & Umlagen', color=(0.184,0.333,0.592), linewidth=1, alpha=0.9)
        #axes[4].plot(x, y['reward_power_electric'].cumsum(), label='el. Lastspitzen', color=(0.929,0.49,0.192), linewidth=1, alpha=0.9)
        #axes[4].set_ylabel('kum. Kosten [€]')
        #axes[4].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        #axes[4].margins(x=0.0,y=0.1)
        #axes[4].set_axisbelow(True)
        #axes[4].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)
        #axes[4].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        #axes[4].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)

        # (6) Rewards
        #cost_total =  y['reward_power_electric'].cumsum()
        #axes[5].plot(x, cost_total, label='Kosten', color=(0.65,0.65,0.65), linewidth=1, alpha=0.9)
        #axes[5].plot(x, y['reward_temperature_heat'].cumsum(), label='Wärmeversorgung', color=(0.75,0,0), linewidth=1, alpha=0.9)
        #axes[5].plot(x, y['reward_switching'].cumsum(), label='Schaltvorgänge', color=(0.44,0.19,0.63), linewidth=1, alpha=0.9)
        # #axes[5].plot(x, y['reward_other'].cumsum(), label='Sonstige', color=(0.1,0.1,0.1), linewidth=1, alpha=0.9)
        # #axes[5].plot(x, y['reward_total'].cumsum(), label='Gesamt', color=(0.1,0.1,0.1), linewidth=2)
        # axes[5].set_ylabel('Reward Strom [€-äquiv.]')
        # axes[5].set_xlabel('Zeit (UTC)')
        # axes[5].set_axisbelow(True)
        # axes[5].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        # axes[5].margins(x=0.0,y=0.1)
        # axes[5].grid(color=(0.9,0.9,0.9,0.1), linestyle='solid', linewidth=1)
        # # add ticks and tick labels
        # axes[5].set_xticks(tickpos)
        # axes[5].set_xticklabels(ticknames, rotation=45)

        # save and close figure
        plt.savefig(
            os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(3)
                + "-"
                + str(self.env_id).zfill(2)
                + "_episode"
                + name_suffix
                + ".png",
            )
        )
        plt.close(figure)

        if plotter_available:
            # HTML PLotter

            xaxis_title = "Zeit (UTC)"
            x2 = self.episode_df.index

            actions = Heatmap(x2, xaxis_title=xaxis_title, height=750, width=1900)
            actions.line(y["u_powerbess"], name="Batterie")
            actions.line(y["u_powerkess"], name="Flywheel")
            #TODO!!!!!!!! anpassen !!!!!!!!!!!!!!!!
            storages = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="Temperatur [°C]", height=750, width=1900)
            storages.line([self.temperature_cost_prod_heat_min - 273.15] * len(x), "T minimal", color="rgb(50,50,50)", dash="dash")
            storages.line(
                [self.temperature_cost_prod_heat_max - 273.15] * len(x), "T maximal", color="rgb(50,50,50)", dash="dash"
            )
   
            storages.line(y["s_actual_BESS"] - 273.15, "P_BESS", color="rgb(192,0,0)")
            storages.line(y["s_actual_KESS"] - 273.15, "P_KESS", color="rgb(192,0,0)", dash="dash")

            prices = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="Energiepreis (netto) [€/kWh]", height=750, width=1900)
            prices.line(y["s_price_electricity"], "Strom", color="rgb(255,191,0)")
            power = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="Leistung [kW]", height=750, width=1900)
            power.line(y["d_usage"] * 1e-3, "Strom Netz", color="rgb(255,191,0)")
            power.line(y["vs_electric_power_total_15min"] * 1e-3, "Strom Netz (Ø15m)", color="rgb(255,191,0)")
            power.line(y["energy_after_compensation"] * 1e-3, "Strom Prod.", width=1, dash="dash", color="rgb(255,191,0)")

            power.line(
                (y["energy_after_compensation"] - y["d_usage"]) * 1e-3,
                "Strom TGA",
                width=0,
                dash="dash",
                color="rgb(255,191,0)",
                fill="tozeroy",
            )


            costs = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="kum. Kosten [€]", height=500, width=1900)
            costs.line(y["reward_energy_electric"].cumsum(), "Strom (netto)", width=1, color="rgb(255,191,0)")
            #costs.line(y["reward_energy_taxes"].cumsum(), "Steuern & Umlagen", width=1, color="rgb(47,85,151)")
            costs.line(y["reward_power_electric"].cumsum(), "el. Lastspitzen", width=1, color="rgb(237,125,49)")

            rewards = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="Rewards [€-äquiv.]", height=500, width=1900)
            #rewards.line(cost_total, "Kosten", width=1, color="rgb(165,165,165)")
            rewards.line(y["reward_energy_loss"].cumsum(), "Verlusleistungen", width=1, color="rgb(191,0,0)")
            rewards.line(y["reward_batteryaging"].cumsum(), "Ladezyklen_Alterung", width=1, color="rgb(112,48,160)")
            rewards.line(y["reward_compensated_energy"].cumsum(), "Kompensierte Leistung", width=1, color="rgb(25,25,25)")
            rewards.line(y["reward_other"].cumsum(), "Gesamt", color="rgb(25,25,25)")
            rewards.line(y["reward_total"].cumsum(), "Gesamt", color="rgb(25,25,25)")

            plot = ETA_Plotter(actions, storages, prices, power, costs, rewards)
            plot.plot_html(
                os.path.join(
                    self.path_results,
                    self.config_run.name
                    + "_"
                    + str(self.n_episodes).zfill(3)
                    + "-"
                    + str(self.env_id).zfill(2)
                    + "_episode"
                    + name_suffix
                    + ".html",
                )
            )

        return


def reward_boundary(state, state_min, state_max, reward, penalty, smoothed=True, k=1):
    """
    reward function for boundaries enabling hard reward/penalties or smoothed by sigmoid function
    Parameters
    -----
        state : (float)
            the state value to be checked
        min : (float) or None
        max : (float) or None
        reward : (float)
            reward given if within min/max boundary
        penalty : (float)
            penalty given if outside of min/max boundary
        smoothed : (bool)
            should reward be smoothed by use if sigmoid function ?
        k : (float)
            modify width of sigmoid smoothing 1/(1+exp(-k*x)) - higher is steeper
    """
    # catch cases when min/max are not defined
    if state_min == None :
        state_min = -1e10
    if state_max == None :
        state_max = 1e10

    if smoothed :
        # return reward - (reward+penalty)*(expit(k*(state-state_max))+expit(k*(state_min-state)))
        return reward - (reward+penalty)*(expit(k*(state-state_max))+expit(k*(state_min-state))) - k*max(state-state_max,0) - k*max(state_min-state,0)
    else :
        return reward if (state > state_min and state < state_max) else -penalty

