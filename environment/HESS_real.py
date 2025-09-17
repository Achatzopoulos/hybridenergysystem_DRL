from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import tz
from eta_utility import get_logger, timeseries
from eta_utility.connectors.entso_e import ENTSOEConnection

# for energy price connector
from eta_utility.connectors.node import NodeEntsoE as Node
from eta_utility.eta_x import ConfigOptRun
from eta_utility.eta_x.envs import BaseEnvLive, StateConfig, StateVar
from eta_utility.type_hints import StepResult, TimeStep
from gymnasium import spaces
from scipy.special import expit

log = get_logger("eta_x.envs")


class SupplysystemETA(BaseEnvLive):
    """
    Live Environment for supplysystems in ETA

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
    version = "v0.1"
    description = "(c) Heiko Ranzau, Niklas Panten and Benedikt Grosch"
    config_name = "supplysystem_ETA_live"

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,  # config_run contains run_name, general_settings, path_settings, env_info
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        sampling_time: float,  # from settings section of config file
        episode_duration: TimeStep | str,  # from settings section of config file
        scenario_time_begin: datetime
        | str,  # CAUTION: Please use only quarter-hourly times, like 10:00, 10:15, 10:30, 10:45, 11:00, etc. and use the following date format "%Y-%m-%d %H:%M"  because otherwise the day-ahead prices will not fit to the current time
        scenario_time_end: datetime | str,
        # scenario_files: Sequence[Mapping[str, Any]],
        allow_policy_shaping,
        IsRuleBasedController,
        random_sampling,
        variant,
        discretize_action_space,
        reward_shaping,
        SOC_KESS_low,
        SOC_KESS_high,
        SOC_BESS_low,
        SOC_BESS_high,
        abort_costs,
        policyshaping_costs,
        power_cost_max,
        tax_el_per_kwh,
        tax_el_produced_per_kwh,
        peak_cost_per_kw,
        extended_observations,
        offset_policy_shaping_threshold,
        rel_path_live_connect_config,  # realtive path to json live connection config files
        live_connect_config_names: list,  # list of json names e.g. ["chp1","chp2"]
        **kwargs: Any,
    ):
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            verbose=verbose,
            callback=callback,
            sampling_time=sampling_time,
            episode_duration=episode_duration,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            **kwargs,
        )

        # make variables readable class-wide
        self.allow_policy_shaping = allow_policy_shaping
        self.offset_policy_shaping_threshold = offset_policy_shaping_threshold
        self.random_sampling = random_sampling
        self.discretize_action_space = discretize_action_space
        self.reward_shaping = reward_shaping
        self.abort_costs = abort_costs
        self.policyshaping_costs = policyshaping_costs
        self.reward_shaping = reward_shaping
        self.SOC_minBESS = 0.31                   # [-] min. SOC BESS
        self.SOC_maxBESS = 0.89                   # [-] max. SOC BESS
        self.SOC_minKESS = 0.31                  # [-] min. SOC KESS
        self.SOC_maxKESS = 0.89                  # [-] max. SOC KESS
        self.p_maxBESS = 120                     # [kW] max. Leistung BESS
        self.p_minBESS = -self.p_maxBESS              # [kW] min. Leistung BESS
        self.p_maxKESS = 40#120                     # [kW] max. Leistung KESS
        self.p_minKESS = -self.p_maxKESS              # [kW] min. Leistung KESS
        self.power_cost_max = power_cost_max
        self.tax_el_per_kwh = tax_el_per_kwh
        self.tax_el_produced_per_kwh = tax_el_produced_per_kwh
        #self.peak_cost_per_kw = peak_cost_per_kw
        self.variant = variant
        # self.energy_through_compensation = energy_through_compensation
        # self.energy_after_compensation = energy_after_compensation
        # self.activation_energy = activation_energy
        ###neue Init nach 21.07.2025###
        # --- Reward & cost parameters ---
        self.peak_running     = 0.0
        self.running_cost_avg = 1.0
        # Fester Arbeitspreis
        self.c_Arbeitspreis = 0.30   # €/kWh
        # Peak‐Leistungspreis (€/kW)
        self.c_power = 17.72
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

        # initialize integrators and longtime stats
        # self.n_steps_longtime = 0
        # self.reward_longtime_average = 0

        # define state variables
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
                is_agent_observation=extended_observations,
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
                is_agent_observation=extended_observations,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_electric_power_1h",
                is_agent_observation=extended_observations,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_electric_power_6h",
                is_agent_observation=extended_observations,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_delta_price_electricity_15m",
                is_agent_observation=extended_observations,
                low_value=-1e2,
                high_value=1e2,
            ),
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
            StateVar(
                name="time_daytime",
                ext_id="time.Systemzeit_Stunde",  # float
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=24,
            ),
            StateVar(
                name="time_month",
                ext_id="time.Systemzeit_Monat",  # datatype should be integer
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=12,
            ),
            StateVar(name="datetime_system"),
            # the following virtual states are used for reward-calculations and therefore not needed for live enironment
            # StateVar(
            #     name="vs_electric_power_total_15min",
            #     is_agent_observation=True,
            #     low_value=-100000,
            #     high_value=500000
            # ),

            StateVar(name="s_price_electricity_00h", is_agent_observation=True, low_value=-2e6, high_value=2e6),
            StateVar(name="s_price_electricity_01h", is_agent_observation=True, low_value=-2e6, high_value=2e6),
            StateVar(name="s_price_electricity_03h", is_agent_observation=True, low_value=-2e6, high_value=2e6),
            StateVar(name="s_price_electricity_06h", is_agent_observation=True, low_value=-2e6, high_value=2e6),
            # # used electricity in production
            # StateVar(
            #     name="electric_power_consumption",
            #     ext_id="electric_power_consumption",
            #     is_ext_output=True,
            #     low_value=-100000,
            #     high_value=500000,
            # ),
        )

        # build final state_config
        self.state_config = StateConfig(*state_var_tuple)

        # import all scenario files
        # self.scenario_data = self.import_scenario(*scenario_files)

        # get action_space
        self.n_action_space = len(
            self.state_config.actions
        )  # get number of actions agent has to give from state_config
        self.action_space = spaces.MultiDiscrete(np.full(self.n_action_space, 2))  # set 2 discrete actions (On, Off)
        self.action_disc_index = [0] * self.n_action_space  # initialize action

        # get observation_space (always continuous)
        self.observation_space = self.state_config.continuous_obs_space()

        # create paths to json files
        live_connect_config_path_list = []
        for element in live_connect_config_names:
            live_connect_config_path_list.append(self.path_env / rel_path_live_connect_config / element)

        self.live_connect_config_path_sequence: Sequence[
            Path
        ] = live_connect_config_path_list  # turn list into Sequence

        # init live connector object
        self._init_live_connector(files=self.live_connect_config_path_sequence)

        """
        # TODO: implement this functionality into utility functions
        if self.discretize_action_space:

            # get number of actions agent has to give from state_config
            self.n_action_space = len(self.state_config.actions)
            # set 3 discrete actions (increase,decrease,equal) per control variable
            self.action_space = spaces.MultiDiscrete(np.full(self.n_action_space, 3))
            # customized for chp, condensingboiler, immersionheater, heatpump, coolingtower, compressionchiller
            self.action_disc_step = [
                [0, 0.5, 0.75, 1],
                [0, 0.25, 0.5, 0.75, 1],
                [0, 0.25, 0.5, 0.75, 1],
                [0, 0.5, 0.75, 1],
                [0, 0.5, 0.75, 1],
                [0, 0.5, 0.75, 1],
            ]
            # initialize action
            self.action_disc_index = [0] * self.n_action_space
        else:
            self.action_space = self.state_config.continuous_action_space()
        """

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.
        """

        # initialize additional_state and create state backup
        # self.state_backup = self.state.copy()
        self.additional_state = {}

        try:
            self.policy_shaping_active
        except:
            self.policy_shaping_active = False

        # overwrite actions if out of boundaries (policy shaping), values are explicitly written for logging purposes
        if self.allow_policy_shaping:
            SOC_KESS_lower_threshold = self.SOC_minKESS - self.offset_policy_shaping_threshold
            SOC_KESS_upper_threshold = self.SOC_maxKESS + self.offset_policy_shaping_threshold
            SOC_BESS_lower_threshold = self.SOC_minBESS - self.offset_policy_shaping_threshold
            SOC_BESS_upper_threshold = self.SOC_maxBESS + self.offset_policy_shaping_threshold

            HNLT_Buffer_lower_threshold = self.HNLT_temperature_reward_min_T - self.offset_policy_shaping_threshold
            HNLT_Buffer_upper_threshold = self.HNLT_temperature_reward_max_T + self.offset_policy_shaping_threshold

            CN_Buffer_lower_threshold = self.CN_temperature_reward_min_T - self.offset_policy_shaping_threshold
            CN_Buffer_upper_threshold = self.CN_temperature_reward_max_T + self.offset_policy_shaping_threshold

            # check if reward shaping is necessary in this step
            if (
                self.state["HNHT_Buffer_fMidTemperature"] < HNHT_Buffer_lower_threshold
                or self.state["HNHT_Buffer_fMidTemperature"] > HNHT_Buffer_upper_threshold
            ):
                policy_shaping_necessary = True
            elif (
                self.state["HNLT_Buffer_fMidTemperature"] < HNLT_Buffer_lower_threshold
                or self.state["HNLT_Buffer_fMidTemperature"] > HNLT_Buffer_upper_threshold
            ):
                policy_shaping_necessary = True
            elif (
                self.state["CN_Buffer_fMidTemperature"] < CN_Buffer_lower_threshold
                or self.state["CN_Buffer_fMidTemperature"] > CN_Buffer_upper_threshold
            ):
                policy_shaping_necessary = True
            else:
                policy_shaping_necessary = False

            # if there was no policy shaping in the last step, but now there is, the Hysteresis Controller should be initalized
            if self.policy_shaping_active == False and policy_shaping_necessary == True:
                # ToDO: think about this later for the AI Application
                self.init_Hysterese_Controllers()

            if policy_shaping_necessary:
                # set policy_shaping for next iteration and reward function
                self.policy_shaping_active == True
                print("Policy Shaping active")

                action = self.control_rules(observation=self.state)

        self._actions_valid(action)

        assert self.state_config is not None, "Set state_config before calling step function."

        self.n_steps += 1
        self._create_new_state(self.additional_state)

        node_in = {}
        # Set actions in the opc ua server and read out the observations
        for idx, name in enumerate(self.state_config.actions):
            self.state[name] = action[idx]
            node_in.update({str(self.state_config.map_ext_ids[name]): action[idx]})

        if self.allow_limiting_CHP_switches:
            """
            this if-statement checks how many steps were performed since last chp switch and allows to change the state
            the node_in is adjusted accordingly
            """
            if self.n_steps > 1:  # only check, if minimum one step was performed
                action_chp1 = action[1]  # get current action
                action_chp2 = action[2]  # get current action

                if action_chp1 == self.action_chp1_backup:
                    self.chp1_switch_counter = self.chp1_switch_counter + 1
                    self.state["overwrite_CHP1_action"] = False
                else:
                    if self.chp1_switch_counter * self.sampling_time / 60 <= 20:  # switching is not ok
                        node_in.update(
                            {str(self.state_config.map_ext_ids["bSetStatusOn_CHP1"]): self.action_chp1_backup}
                        )
                        self.chp1_switch_counter = self.chp1_switch_counter + 1
                        self.state["overwrite_CHP1_action"] = True
                    else:  # switch is ok
                        self.chp1_switch_counter = 0
                        self.state["overwrite_CHP1_action"] = False

                if action_chp2 == self.action_chp2_backup:
                    self.chp2_switch_counter = self.chp2_switch_counter + 1
                    self.state["overwrite_CHP2_action"] = False
                else:
                    if self.chp2_switch_counter * self.sampling_time / 60 <= 20:  # switching is not ok
                        node_in.update(
                            {str(self.state_config.map_ext_ids["bSetStatusOn_CHP2"]): self.action_chp2_backup}
                        )
                        self.chp2_switch_counter = self.chp2_switch_counter + 1
                        self.state["overwrite_CHP2_action"] = True
                    else:
                        self.chp2_switch_counter = 0
                        self.state["overwrite_CHP2_action"] = False
            else:
                self.state["overwrite_CHP1_action"] = False
                self.state["overwrite_CHP2_action"] = False

        # log infos for user
        if self.state["overwrite_CHP1_action"] == True:
            print(f"The CHP1 state was overwritten from {action_chp1} to {self.action_chp1_backup}")
        if self.state["overwrite_CHP2_action"] == True:
            print(f"The CHP2 state was overwritten from {action_chp2} to {self.action_chp2_backup}")

        # save chp1 and chp2 action as backup
        if self.allow_limiting_CHP_switches and self.n_steps > 0:
            if self.state["overwrite_CHP1_action"] == False:
                self.action_chp1_backup = action[1]
            if self.state["overwrite_CHP2_action"] == False:
                self.action_chp2_backup = action[2]

        results = self.live_connector.step(node_in)

        self.state = {name: results[str(self.state_config.map_ext_ids[name])] for name in self.state_config.ext_outputs}
        # update self.state
        self.update_energy_price()  # state is updated in function
        self.state.update({"datetime_system": str(datetime.now())})
        self.state.update(self.get_scenario_state())  # get scenario data #changed_since
        self.action_names = self.state_config.actions
        self.state.update(dict(zip(self.action_names, action)))  # TODO: check this line of code

        self.state_log.append(self.state)

        self.save_csv_stepwise()  # call this to save the current self.state to a csv
        self.check_system_action_feedback()

        # logs
        print(f"Live Enironment running successfully for {self.n_steps*self.sampling_time/60} min.")
        log_T_HNHT = self.state["HNHT_Buffer_fMidTemperature"]
        log_T_HNLT = self.state["HNLT_Buffer_fMidTemperature"]
        log_T_CN = self.state["CN_Buffer_fMidTemperature"]
        print(
            f"The Buffer Temperatures are (HNHT, HNLT, CN): {round(log_T_HNHT, 2)} °C ; {round(log_T_HNLT, 2)} °C ; {round(log_T_CN, 2)} °C "
        )
        print(
            f"HNHT: CHP1: {self.state['bSetStatusOn_CHP1']}, CHP2: {self.state['bSetStatusOn_CHP2']}, Gasboiler: {self.state['bSetStatusOn_CondensingBoiler']}, VSI On/Off: {self.state['bSetStatusOn_VSIStorage']}, VSI bLoading: {self.state['bLoading_VSISystem']}"
        )
        print(
            f"HNLT: PWT1: {self.state['bSetStatusOn_HeatExchanger1']}, WP: {self.state['bSetStatusOn_HeatPump']}, AFA: {self.state['bSetStatusOn_OuterCapillaryTubeMats']}, HVFA On/Off: {self.state['bSetStatusOn_HVFASystem_HNLT']}, HVFA bLoading: {self.state['bLoading_HVFASystem_HNLT']}"
        )
        print(
            f"CN: eChiller: {self.state['bSetStatusOn_eChiller']}, HVFA On/Off: {self.state['bSetStatusOn_HVFASystem_CN']}, HVFA bLoading: {self.state['bLoading_HVFASystem_CN']}"
        )

        # self.state["step_success"], _ = self._update_state(_action)
        # check if state is in valid boundaries
        # if np.isnan(round(log_T_HNHT, 2)) or np.isnan(log_T_HNLT, 2) or np.isnan(round(log_T_CN, 2)):
        #     log.warning("Abort-Check not possible. Please check for Connection and Temperature Limits.")
        #     log.warning("Using same observations.")
        #     self.state["step_abort"] = False
        #     observations = self.observation_backup
        # else:
        #     self.state["step_abort"] = False if StateConfig.within_abort_conditions(self.state_config, self.state) else True
        #     observations = self._observations()
        #     self.observation_backup = observations

        self.state["step_abort"] = False if StateConfig.within_abort_conditions(self.state_config, self.state) else True
        observations = self._observations()

        reward = self.calc_reward(action)  # this will be 0

        # ToDo The following is not working in live, but do I need this?
        done = (
            self._done()
        )  # or not self.state["step_success"] # check if episode is over or not using the number of steps performed
        done = done if not self.state["step_abort"] else True

        return observations, reward, done, False, {}

    def update_predictions(self):

        pass

    def update_energy_price(self) -> dict[str, Any]:
        """
        This function is to get the day-ahead-prices from Entsoe
        """

        API_TOKEN = "fcd0ef8b-887a-4c92-8b33-39cc270b664e"  # API KEY generated by Tobias Lademann
        # Check out NodeEntsoE documentation for endpoint and bidding zone information
        node = Node(
            name="ENTSOE-DEU-LUX",
            url="https://web-api.tp.entsoe.eu/",
            protocol="entsoe",
            endpoint="Price",
            bidding_zone="DEU-LUX",
        )

        if self.n_steps == 0:
            current_time = datetime.utcnow()  # ask for current time with UTC format
            current_time = current_time.replace(second=0, microsecond=0, minute=0)  # round down to full hours
            self.current_time_last_timestep = current_time

            el_prices = {
                "s_price_electricity_00h": 0.05,
                "s_price_electricity_01h": 0.05,
                "s_price_electricity_03h": 0.05,
                "s_price_electricity_06h": 0.05,
            }
            self.el_prices_backup = el_prices

        try:
            # Start connection from one or multiple nodes
            server = ENTSOEConnection.from_node(node, api_token=API_TOKEN)

            current_time = datetime.utcnow()  # ask for current time with UTC format
            current_time = current_time.replace(second=0, microsecond=0, minute=0)  # round down to full hours

            # only get new prices, if current time is not equal to the one from last timestep
            if self.current_time_last_timestep != current_time or self.n_steps == 0:
                # Add 6 hours to the rounded time to get the target time
                six_hours_later = current_time + timedelta(hours=6)

                from_datetime = current_time
                to_datetime = six_hours_later
                # interval: interval between time steps. It is interpreted as seconds if given as integer. e. g. interval=60 means one data point per minute
                df_energy_price = server.read_series(from_time=from_datetime, to_time=to_datetime, interval=60 * 60)

                el_prices = {
                    "s_price_electricity_00h": df_energy_price["ENTSOE-DEU-LUX_60"][0] * 0.001,
                    "s_price_electricity_01h": df_energy_price["ENTSOE-DEU-LUX_60"][1] * 0.001,
                    "s_price_electricity_03h": df_energy_price["ENTSOE-DEU-LUX_60"][3] * 0.001,
                    "s_price_electricity_06h": df_energy_price["ENTSOE-DEU-LUX_60"][6] * 0.001,
                }

                # update backup variables
                self.el_prices_backup = el_prices
                self.current_time_last_timestep = current_time

            else:  # still same hour as last time
                el_prices = self.el_prices_backup

        except:
            el_prices = self.el_prices_backup  # use energy prices from last time

        # update state

        # constant_gas_price = 0.001
        gas_price = {"s_price_gas": self.constant_gas_price}
        self.state.update(gas_price)

        self.state.update(el_prices)

    def update_virtual_state(self):
        """
        For the live application no vs is needed.

        """

        virtual_state = {}

        return virtual_state

    def calc_reward(self, action):
        """
        For the live application, no reward is calculated, because some live values are missing.

        :return: Normalized or non-normalized reward
        :rtype: Real
        """

        self.state["reward_total"] = 0

        return self.state["reward_total"]

    def reset(self) -> np.ndarray:

        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """

        # delete long time storage, since it takes up too much memory during training
        # self.state_log_longtime = []

        # # save episode's stats
        if self.n_steps > 0:

            # create dataframe from state_log
            self.episode_df = pd.DataFrame(self.state_log)

            # derive certain episode statistics for logging and plotting
            # self.episode_statistics["datetime_begin"] = self.ts_current.index[0].strftime("%Y-%m-%d %H:%M") #ToDo
            self.episode_statistics["time"] = time.time() - self.episode_timer
            self.episode_statistics["n_steps"] = self.n_steps

            # # append episode store to episodes' archive
            self.episode_archive.append(list(self.episode_statistics.values()))

        # initialize additional state, since super().reset is used
        self.additional_state = {}

        # get current slice of timeseries dataframe, extended by maximum prediction horizon (6h)
        # and one additional step because step 0 = init conditions
        # self.ts_current = timeseries.df_time_slice(
        #     self.scenario_data,
        #     self.scenario_time_begin,
        #     self.scenario_time_end,
        #     self.episode_duration + (self.n_steps_6h + 1) * self.sampling_time,
        #     random=self.np_random if self.random_sampling else False,
        # )

        # read current date time
        # self.episode_datetime_begin = self.ts_current.index[0]
        # or rather use the following?
        # self.episode_datetime_begin = pd.to_datetime(self.scenario_time_begin+timedelta(seconds=self.sampling_time*self.n_steps_total))
        # self.additional_state["vs_time_daytime"] = self.episode_datetime_begin.hour

        # get scenario input for initialization (time step: 0)
        # self.additional_state.update(self.update_predictions())

        assert self.state_config is not None, "Set state_config before calling reset function."
        self._reset_state()
        self._init_live_connector(files=self.live_connect_config_path_sequence)

        self.state = {} if self.additional_state is None else self.additional_state

        if self.allow_limiting_CHP_switches:
            self.chp1_switch_counter = 0
            self.chp2_switch_counter = 0

        # Read out and store start conditions
        results = self.live_connector.read(*self.state_config.map_ext_ids.values())
        # call electricity price here
        self.update_energy_price()
        self.state.update({"datetime_system": str(datetime.now())})

        self.state.update(
            {name: results[str(self.state_config.map_ext_ids[name])] for name in self.state_config.ext_outputs}
        )
        self.state.update(self.get_scenario_state())

        log_T_HNHT = self.state["HNHT_Buffer_fMidTemperature"]
        log_T_HNLT = self.state["HNLT_Buffer_fMidTemperature"]
        log_T_CN = self.state["CN_Buffer_fMidTemperature"]
        log.info(
            f"The Buffer Temperatures are (HNHT, HNLT, CN): {round(log_T_HNHT, 2)} °C ; {round(log_T_HNLT, 2)} °C ; {round(log_T_CN, 2)} °C "
        )

        # receive observations from live connector
        observations = self._observations()

        self.state_log.append(self.state)

        return observations

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
            decimal=",",
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
        episode_archive_df.tail(1).to_csv(path_or_buf=csvpath, sep=";", decimal=",", mode=tocsvmode, header=tocsvheader)

        # plot settings
        # create figure and axes with custom layout/adjustments
        # figure with all episodes is not necessary in live application
        """
        figure = plt.figure(figsize=(14, 14), dpi=200)
        axes = []
        axes.append(figure.add_subplot(2, 1, 1))
        axes.append(figure.add_subplot(2, 1, 2, sharex=axes[0]))

        # fig, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=200)
        x = np.arange(len(episode_archive_df.index))
        y = episode_archive_df

        # (1) Costs
        axes[0].plot(
            x, y["rewards_energy_electric"], label="Strom (netto)", color=(1.0, 0.75, 0.0), linewidth=1, alpha=0.9
        )
        axes[0].plot(
            x, y["rewards_energy_gas"], label="Erdgas (netto)", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9
        )
        axes[0].plot(
            x, y["rewards_energy_taxes"], label="Steuern & Umlagen", color=(0.184, 0.333, 0.592), linewidth=1, alpha=0.9
        )
        axes[0].plot(
            x, y["rewards_power_electric"], label="el. Lastspitzen", color=(0.929, 0.49, 0.192), linewidth=1, alpha=0.9
        )
        axes[0].set_ylabel("kum. Kosten [€]")
        axes[0].set_xlabel("Episode")
        axes[0].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[0].margins(x=0.0, y=0.1)
        axes[0].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)

        # (2) Rewards
        cost_total = (
            y["rewards_energy_electric"]
            + y["rewards_energy_gas"]
            + y["rewards_energy_taxes"]
            + y["rewards_power_electric"]
        )
        axes[1].plot(x, cost_total, label="Kosten", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9)
        axes[1].plot(
            x, y["reward_temperature_HNHT"], label="HNHT", color=(0.75, 0, 0), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x, y["reward_temperature_HNLT"], label="HNLT", color=(0, 0.75, 0.25), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x, y["reward_temperature_CN"], label="CN", color=(0.36, 0.61, 0.84), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x, y["rewards_switching"], label="Schaltvorgänge", color=(0.44, 0.19, 0.63), linewidth=1, alpha=0.9
        )
        axes[1].plot(x, y["rewards_other"], label="Sonstige", color=(0.1, 0.1, 0.1), linewidth=1, alpha=0.9)
        axes[1].plot(x, y["rewards_total"], label="Gesamt", color=(0.1, 0.1, 0.1), linewidth=2)
        axes[1].set_ylabel("kum. + fikt. Kosten [€]")
        axes[1].set_xlabel("Episode")
        axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[1].margins(x=0.0, y=0.1)
        axes[1].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)

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
        """

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

    def check_system_action_feedback(self):

        """
        This function checks if the actions, which were set equals the status which is feedbacked.
        If this is not the case, there might be a communication issue and the user should take further actions.

        """

        try:
            if not self.state["bSetStatusOn_HeatExchanger1"] == self.state["Out_bSetStatusOn_HeatExchanger1"]:
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_HeatExchanger1")
            if not self.state["bSetStatusOn_CHP1"] == self.state["Out_bSetStatusOn_CHP1"]:
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_CHP1")
            if not self.state["bSetStatusOn_CHP2"] == self.state["Out_bSetStatusOn_CHP2"]:
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_CHP2")
            if not self.state["bSetStatusOn_CondensingBoiler"] == self.state["Out_bSetStatusOn_CondensingBoiler"]:
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_CondensingBoiler")
            if not self.state["bSetStatusOn_VSIStorage"] == self.state["Out_bSetStatusOn_VSIStorage"]:
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_VSIStorage")
            if not self.state["bSetStatusOn_HVFASystem_HNLT"] == self.state["Out_bSetStatusOn_HVFASystem_HNLT"]:
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_HVFASystem_HNLT")
            if not self.state["bSetStatusOn_eChiller"] == self.state["Out_bSetStatusOn_eChiller"]:
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_eChiller")
            if not self.state["bSetStatusOn_HVFASystem_CN"] == self.state["Out_bSetStatusOn_HVFASystem_CN"]:
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_HVFASystem_CN")
            if (
                not self.state["bSetStatusOn_OuterCapillaryTubeMats"]
                == self.state["Out_bSetStatusOn_OuterCapillaryTubeMats"]
            ):
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_OuterCapillaryTubeMats")
            if not self.state["bSetStatusOn_HeatPump"] == self.state["Out_bSetStatusOn_HeatPump"]:
                log.warning("Contradiction in Actions and Observations:")
                log.warning("bSetStatusOn_HeatPump")
        except:
            log.warning("Actions of real system could not be checked.")

    def save_csv_stepwise(self):

        # create path to save the stepwise csv if not existing already
        try:
            self.path_to_save_csv
        except:
            self.path_to_save_csv = os.path.join(
                self.path_results, self.config_run.name + "_" + "stepwise_log" + ".csv"
            )

        # convert self.state dict to pandas df
        df_to_save = pd.DataFrame(self.state, index=[0])

        if os.path.isfile(self.path_to_save_csv):
            # if csv already exists, append
            try:
                """
                When the user does not open the csv while running, everything goes fine.
                If you want to open it while the code is still running, make a copy and open the copy!
                """
                with open(self.path_to_save_csv, "a") as f:
                    df_to_save.to_csv(f, header=False, index=False, sep=";", decimal=",", lineterminator="\n")
            except:
                log.info("Got exception while saving csv. Please close the csv quickly. The code is still running.")
                # find new path to save csv
                for i in range(99):
                    self.path_to_save_csv = os.path.join(
                        self.path_results, self.config_run.name + "_" + "stepwise_log" + str(i) + ".csv"
                    )
                    if not os.path.isfile(self.path_to_save_csv):
                        break
                df_to_save.to_csv(self.path_to_save_csv, index=False, sep=";", decimal=",", lineterminator="\n")
        else:
            # first time step, so no csv existing yet. Therefore create a new one.
            df_to_save.to_csv(self.path_to_save_csv, index=False, sep=";", decimal=",", lineterminator="\n")

    def render(self, mode="human", name_suffix=""):
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
            decimal=",",
        )

        return

    def init_Hysterese_Controllers(self):
        fTargetTemperature_HNHT = 70
        fTargetTemperature_HNLT_Cooling = 40
        fTargetTemperature_HNLT_Heating = 35
        fTargetTemperature_CN = 18

        self.Controller_CHP_Prio = HysteresisController(hysteresis_range=4, target=15)
        self.Controller_CHP1 = HysteresisController(hysteresis_range=0, target=0)  # this is set later
        self.Controller_CHP2 = HysteresisController(hysteresis_range=0, target=0)  # this is set later
        self.Controller_CondensingBoiler = HysteresisController(hysteresis_range=10, target=fTargetTemperature_HNHT + 1)

        self.Controller_VSI_Unloading = HysteresisController(hysteresis_range=4, target=fTargetTemperature_HNHT - 0)

        self.Controller_VSI_Unloading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNHT, inverted=True
        )
        self.Controller_VSI_Loading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNHT
        )

        self.Controller_OuterCapillaryTubeMats = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Cooling + 2, inverted=True  # 42
        )  # cooling application
        self.Controller_OuterCapillaryTubeMats_Permission = HysteresisController(
            hysteresis_range=6, target=0
        )  # target is set later

        self.Controller_HeatExchanger1 = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Heating + 4
        )  # 39

        self.Controller_eChiller = HysteresisController(hysteresis_range=6, target=fTargetTemperature_CN, inverted=True)

        self.Controller_HeatPump = HysteresisController(hysteresis_range=5, target=fTargetTemperature_HNLT_Heating + 0)
        self.Controller_HeatPump_Permission = HysteresisController(
            hysteresis_range=3, target=fTargetTemperature_CN - 3, inverted=True
        )

        self.Controller_HVFA_CN_Loading_Permission = HysteresisController(
            hysteresis_range=2, target=fTargetTemperature_CN, inverted=True
        )
        self.Controller_HVFA_CN_Unloading_Permission = HysteresisController(
            hysteresis_range=2, target=fTargetTemperature_CN
        )
        self.Controller_Buffer_HVFA_CN_Loading = HysteresisController(hysteresis_range=2, target=fTargetTemperature_CN)
        self.Controller_Buffer_HVFA_CN_Unloading = HysteresisController(
            hysteresis_range=2, target=fTargetTemperature_CN, inverted=True
        )

        self.Controller_Controller_HVFA_HNLT_Loading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Cooling
        )
        self.Controller_Controller_HVFA_HNLT_Unloading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Heating, inverted=True
        )
        self.Controller_Buffer_HVFA_HNLT_Loading = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Cooling, inverted=True
        )
        self.Controller_Buffer_HVFA_HNLT_Unloading = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Heating + 2
        )

    # def control_rules(self, observation: np.ndarray) -> np.ndarray:
    def control_rules(self, observation) -> np.ndarray:
        """
        Controller of the model.

        :param observation: Observation from the environment.
        :returns: actions
        """

        assert isinstance(observation, dict), "observation is not from type dict"

        # action = dict.fromkeys(self.action_names, 0) #this creates a dict with the action names and 0 as values
        action = {
            "bSetStatusOn_HeatExchanger1": 0,
            "bSetStatusOn_CHP1": 0,
            "bSetStatusOn_CHP2": 0,
            "bSetStatusOn_CondensingBoiler": 0,
            "bSetStatusOn_VSIStorage": 0,
            "bLoading_VSISystem": 0,
            "bSetStatusOn_HVFASystem_HNLT": 0,
            "bLoading_HVFASystem_HNLT": 0,
            "bSetStatusOn_eChiller": 0,
            "bSetStatusOn_HVFASystem_CN": 0,
            "bLoading_HVFASystem_CN": 0,
            "bSetStatusOn_OuterCapillaryTubeMats": 0,
            "bSetStatusOn_HeatPump": 0,
        }

        # get observations
        HNHT_Buffer_fMidTemperature = observation["HNHT_Buffer_fMidTemperature"]
        HNLT_Buffer_fMidTemperature = observation["HNLT_Buffer_fMidTemperature"]
        CN_Buffer_fMidTemperature = observation["CN_Buffer_fMidTemperature"]

        VSI_fUpperTemperature = observation["HNHT_VSI_fUpperTemperature"]
        VSI_fLowerTemperature = observation["HNHT_VSI_fLowerTemperature"]

        # CN_HVFA_fUpperTemperature = observation["CN_HVFA_fUpperTemperature"]
        CN_HVFA_fLowerTemperature = observation["CN_HVFA_fLowerTemperature"]
        HNLT_HVFA_fLowerTemperature = observation["HNLT_HVFA_fLowerTemperature"]

        T_Mean = observation["weather_T_amb_Mean"]
        T_amb = observation["weather_T_amb"]

        # define parameters
        fTargetTemperature_HNHT = 70

        # ask for controller outputs for actions which are more complex
        if self.Controller_CHP_Prio.update(actual_value=T_Mean) == 1:
            CHP1_hysteresis_range = 14
            CHP2_hysteresis_range = 12
            fOffset_TargetTemperature_CHP1 = 1
            fOffset_TargetTemperature_CHP2 = 0

            # change hysteresis settings
            self.Controller_CHP1.change_hysteresis_range(CHP1_hysteresis_range)
            self.Controller_CHP1.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP1)

            self.Controller_CHP2.change_hysteresis_range(CHP2_hysteresis_range)
            self.Controller_CHP2.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP2)
        else:
            CHP1_hysteresis_range = 12
            CHP2_hysteresis_range = 14
            fOffset_TargetTemperature_CHP1 = 0
            fOffset_TargetTemperature_CHP2 = 1
            self.Controller_CHP1.change_hysteresis_range(CHP1_hysteresis_range)
            self.Controller_CHP1.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP1)

            self.Controller_CHP2.change_hysteresis_range(CHP2_hysteresis_range)
            self.Controller_CHP2.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP2)

        # set actions
        action["bSetStatusOn_CHP1"] = self.Controller_CHP1.update(actual_value=HNHT_Buffer_fMidTemperature)
        action["bSetStatusOn_CHP2"] = self.Controller_CHP2.update(actual_value=HNHT_Buffer_fMidTemperature)

        action["bSetStatusOn_CondensingBoiler"] = self.Controller_CondensingBoiler.update(
            actual_value=HNHT_Buffer_fMidTemperature
        )

        # VSI
        Controller_Output_VSI_Unloading = self.Controller_VSI_Unloading.update(actual_value=HNHT_Buffer_fMidTemperature)
        Controller_Output_VSI_Unloading_Permission = self.Controller_VSI_Unloading_Permission.update(
            actual_value=VSI_fUpperTemperature
        )
        Controller_Output_VSI_Loading_Permission = self.Controller_VSI_Loading_Permission.update(
            actual_value=VSI_fLowerTemperature
        )

        if Controller_Output_VSI_Unloading and Controller_Output_VSI_Unloading_Permission:
            action["bSetStatusOn_VSIStorage"] = 1
        elif action["bSetStatusOn_CHP1"] == 1 or action["bSetStatusOn_CHP2"] == 1:
            if Controller_Output_VSI_Unloading == 0 and Controller_Output_VSI_Loading_Permission == 1:
                action["bSetStatusOn_VSIStorage"] = 1
            else:
                action["bSetStatusOn_VSIStorage"] = 0
        else:
            action["bSetStatusOn_VSIStorage"] = 0

        if Controller_Output_VSI_Unloading == 1 and Controller_Output_VSI_Unloading_Permission == 1:
            action["bLoading_VSISystem"] = 0
        else:
            action["bLoading_VSISystem"] = 1

        # HNHT-HNLT Linkage
        action["bSetStatusOn_HeatExchanger1"] = self.Controller_HeatExchanger1.update(
            actual_value=HNLT_Buffer_fMidTemperature
        )

        # HNLT
        Controller_Output_OuterCapillaryTubeMats = self.Controller_OuterCapillaryTubeMats.update(
            actual_value=HNLT_Buffer_fMidTemperature
        )
        self.Controller_OuterCapillaryTubeMats_Permission.change_target(HNLT_Buffer_fMidTemperature - 6)
        Controller_Output_OuterCapillaryTubeMats_Permission = self.Controller_OuterCapillaryTubeMats_Permission.update(
            actual_value=T_amb
        )
        if Controller_Output_OuterCapillaryTubeMats and Controller_Output_OuterCapillaryTubeMats_Permission:
            action["bSetStatusOn_OuterCapillaryTubeMats"] = 1
        else:
            action["bSetStatusOn_OuterCapillaryTubeMats"] = 0

        # HNLT HVFA
        Controller_Controller_HVFA_HNLT_Loading_Permission_Output = (
            self.Controller_Controller_HVFA_HNLT_Loading_Permission.update(HNLT_HVFA_fLowerTemperature)
        )
        Controller_Controller_HVFA_HNLT_Unloading_Permission_Output = (
            self.Controller_Controller_HVFA_HNLT_Unloading_Permission.update(HNLT_HVFA_fLowerTemperature)
        )
        Controller_Buffer_HVFA_HNLT_Loading_Output = self.Controller_Buffer_HVFA_HNLT_Loading.update(
            HNLT_Buffer_fMidTemperature
        )
        Controller_Buffer_HVFA_HNLT_Unloading_Output = self.Controller_Buffer_HVFA_HNLT_Unloading.update(
            HNLT_Buffer_fMidTemperature
        )

        if Controller_Controller_HVFA_HNLT_Loading_Permission_Output and Controller_Buffer_HVFA_HNLT_Loading_Output:
            action["bSetStatusOn_HVFASystem_HNLT"] = 1
        elif (
            Controller_Controller_HVFA_HNLT_Unloading_Permission_Output and Controller_Buffer_HVFA_HNLT_Unloading_Output
        ):
            action["bSetStatusOn_HVFASystem_HNLT"] = 1
        else:
            action["bSetStatusOn_HVFASystem_HNLT"] = 0

        if Controller_Controller_HVFA_HNLT_Loading_Permission_Output and Controller_Buffer_HVFA_HNLT_Loading_Output:
            action["bLoading_HVFASystem_HNLT"] = 1
        else:
            action["bLoading_HVFASystem_HNLT"] = 0

        # CN
        action["bSetStatusOn_eChiller"] = self.Controller_eChiller.update(CN_Buffer_fMidTemperature)

        # CN HVFA
        Controller_HVFA_CN_Loading_Permission_Output = self.Controller_HVFA_CN_Loading_Permission.update(
            CN_HVFA_fLowerTemperature
        )
        Controller_HVFA_CN_Unloading_Permission_Output = self.Controller_HVFA_CN_Unloading_Permission.update(
            CN_HVFA_fLowerTemperature
        )

        Controller_Buffer_HVFA_CN_HVFA_Loading_Output = self.Controller_Buffer_HVFA_CN_Loading.update(
            CN_Buffer_fMidTemperature
        )
        Controller_Buffer_HVFA_CN_HVFA_Unoading_Output = self.Controller_Buffer_HVFA_CN_Unloading.update(
            CN_Buffer_fMidTemperature
        )

        if Controller_HVFA_CN_Loading_Permission_Output and Controller_Buffer_HVFA_CN_HVFA_Loading_Output:
            action["bSetStatusOn_HVFASystem_CN"] = 1
        elif Controller_HVFA_CN_Unloading_Permission_Output and Controller_Buffer_HVFA_CN_HVFA_Unoading_Output:
            action["bSetStatusOn_HVFASystem_CN"] = 1
        else:
            action["bSetStatusOn_HVFASystem_CN"] = 0

        if Controller_HVFA_CN_Loading_Permission_Output and Controller_Buffer_HVFA_CN_HVFA_Loading_Output:
            action["bLoading_HVFASystem_CN"] = 1
        else:
            action["bLoading_HVFASystem_CN"] = 0

        # HNLT-CN Linkage
        Controller_Output_HeatPump = self.Controller_HeatPump.update(HNLT_Buffer_fMidTemperature)
        Controller_Output_HeatPump_Permission = self.Controller_HeatPump_Permission.update(CN_Buffer_fMidTemperature)
        if Controller_Output_HeatPump and Controller_Output_HeatPump_Permission:
            action["bSetStatusOn_HeatPump"] = 1
        else:
            action["bSetStatusOn_HeatPump"] = 0

        # print(action)
        actions = []
        actions.append(list(action.values()))
        # print(actions)
        actions = actions[0]

        return np.array(actions)


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
    if state_min == None:
        state_min = -1e10
    if state_max == None:
        state_max = 1e10

    if smoothed:
        # return reward - (reward+penalty)*(expit(k*(state-state_max))+expit(k*(state_min-state)))
        return (
            reward
            - (reward + penalty) * (expit(k * (state - state_max)) + expit(k * (state_min - state)))
            - k * max(state - state_max, 0)
            - k * max(state_min - state, 0)
        )
    else:
        return reward if (state > state_min and state < state_max) else -penalty


class HysteresisController:
    def __init__(self, hysteresis_range, target, inverted=False, init_value=0):
        self.hysteresis_range = hysteresis_range  # This is the hysteresis range
        self.target = target  # This is the target temperature
        self.inverted = inverted  # This should be True e.g. for eChiller, which should be 1 when T is too high
        self.output = init_value  # The output is always init with 0

    def update(self, actual_value):
        if self.inverted == False:
            if self.output == 0:  # controller is off
                # controller output still 0 if input value below threshhold
                self.output = 1 if actual_value <= self.target - self.hysteresis_range / 2 else 0
            else:  # controller is on
                self.output = 0 if actual_value >= self.target + self.hysteresis_range / 2 else 1

        else:
            if self.output == 0:  # controller is off
                # controller output still 0 if input value below threshhold
                self.output = 1 if actual_value >= self.target + self.hysteresis_range / 2 else 0
            else:  # controller is on
                self.output = 0 if actual_value <= self.target - self.hysteresis_range / 2 else 1

        return self.output

    def change_hysteresis_range(self, hysteresis_range):
        self.hysteresis_range = hysteresis_range

    def change_target(self, target):
        self.target = target
