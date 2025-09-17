import gymnasium
gymnasium.logger.set_level(40)  # to prevent gym complaining about using just float32
from gymnasium import spaces
from gymnasium.utils import seeding
from eta_utility.simulators import FMUSimulator
from itertools import count
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # to desable SettingWithCopyWarning
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime, timedelta
import locale
from scipy.special import expit
import math



class Hybridenergysystem(gymnasium.Env):
    """The main FMU environment class extending a gymnasium.env class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    # set info
    version = "v0.2"
    description = "(c) Niklas Panten"
    fmu_name = "hybridenergysystem_varPow"
    _ids = count(0)

    def __init__(self,render_mode:None, env_id, run_name, general_settings, path_settings, env_settings, verbose=1, callback=None):
        """Initialize the environment
        Parameters
        -----
            sampling_time : (int)
            episode_duration : (int)
            control_interval : (int)
            path_settings : (dict)
            env_settings : (dict)
            callback : (method)
                callback method will be called after each episode with all data within the environment class
        """
        
        # call inherited init function
        super(Hybridenergysystem, self).__init__()

        # set instance id and run_name
        self.env_id = env_id
        self.run_name = run_name

        # check for integrity of parameters
        req_general_settings = ["sampling_time", "episode_duration", "control_interval", "n_environments"]
        req_path_settings = ["path_root", "path_results", "relpath_scenarios"]
        req_env_settings = ["seed",
                            "variant",
                            "discretize_action_space",
                            "discretize_state_space",
                            "n_action_disc_steps",
                            "normalize_state_space",
                            "normalize_reward",
                            "reward_shaping",
                            "scenario_time_begin",
                            "scenario_time_end",
                            "scenario_csv_factory",
                            "scenario_csv_energymarkets",
                            "scenario_csv_usage",
                            "SOC_low",
                            "SOC_high",
                            "scenario_factory_scale_electric",
                            "abort_costs",
                            "activation_energy"
                            "policyshaping_costs",
                            "power_cost_max"
                            ]
        if all(name in general_settings for name in req_general_settings) and all(name in path_settings for name in req_path_settings) and all(name in env_settings for name in req_env_settings) :
            print("[INFO] All required parameters are delivered to environment (" + str(self.env_id) + ").")
        else :
            print("[WARNING] Not all required parameters are delivered to environment (" + str(self.env_id) + ").")

        # settings
        self.sampling_time = int(general_settings['sampling_time'])
        self.n_steps_episode_end = int(round(int(general_settings['episode_duration'])/self.sampling_time, 0))
        self.episode_duration = int(self.n_steps_episode_end*self.sampling_time)
        self.control_interval = int(general_settings['control_interval'])
        self.n_environments = int(general_settings['n_environments'])
        self.env_settings = env_settings.copy()
        self.env_settings.update({'scenario_time_begin': datetime.strptime(env_settings['scenario_time_begin'], '%Y-%m-%d %H:%M')})
        self.env_settings.update({'scenario_time_end': datetime.strptime(env_settings['scenario_time_end'], '%Y-%m-%d %H:%M')})
        if env_settings['seed'] == '' :
            seed = None
        else :
            seed = int(env_settings['seed']) + self.env_id

        # set seeding for pseudorandom numbers
        self.np_random, self.seed = seeding.np_random(seed)

        # set locale for german date formats
        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
        
        # path settings
        self.path_root = path_settings['path_root']
        self.path_results = path_settings['path_results']
        self.path_env = os.path.dirname(__file__)
        self.path_fmu = os.path.join(self.path_env, self.fmu_name+'.fmu')
        self.path_scenario_factory = os.path.join(self.path_root, path_settings['relpath_scenarios'], env_settings['scenario_csv_factory']+'.csv')
        self.path_scenario_usage = os.path.join(self.path_root, path_settings['relpath_scenarios'], env_settings['scenario_csv_usage']+'.csv')
        self.path_scenario_markets = os.path.join(self.path_root, path_settings['relpath_scenarios'], env_settings['scenario_csv_energymarkets']+'.csv')

        # save verbose/debug level
        self.verbose = verbose

        # store callback function in object
        self.callback = callback

        extended_state = False if ('reduced_state' in env_settings['variant']) else True
        mpc_state = True if ('mpc_state' in env_settings['variant']) else False

        # variables dataframe [name, FMU name, is FMU action?, is FMU observation?, is agent action?, is agent state?, min, max]
        self.state_config = pd.DataFrame(  
            # controls
            [['u_powerbess', 'Input', True, False, True, False, -1, 1, None, None],
             ['u_powerkess', 'Input1', True, False, True, False, -1, 1, None, None],
             ['u_maxPowerBESS', 'Input3', True, False, False, False, 0, 120000, None, None],
             ['u_maxPowerKESS', 'Input4', True, False, False, False, 0, 100000, None, None],
            # disturbances
            ## fmu model without disturbance
            # predictions
            # ['p_production_delta_heat_power_1h', '', False, False, False, extended_state, -2e6, 2e6, None, None],
            # ['p_production_delta_heat_power_6h', '', False, False, False, False, -2e6, 2e6, None, None],
            #['p_production_delta_electric_power_15m', '', False, False, False, extended_state, -2e6, 2e6, None, None],
            #['p_production_delta_electric_power_1h', '', False, False, False, False, -2e6, 2e6, None, None],
            #['p_production_delta_electric_power_6h', '', False, False, False, False, -2e6, 2e6, None, None],
            #['p_delta_price_electricity_15m', '', False, False, False, False, -0.2, 0.2, None, None],
            #['p_delta_price_electricity_1h', '', False, False, False, extended_state, -0.2, 0.2, None, None],
            #['p_delta_price_electricity_6h', '', False, False, False, extended_state, -0.2, 0.2, None, None],
            # virtual states
            # ['vs_time_daytime', '', False, False, False, extended_state, 0, 24, None, None],   # additional agent states
            # ['vs_electric_power_total_15min', '', False, False, False, True, -100000, 500000, None, None],
            # ['vs_gas_power_total_15min', '', False, False, False, False, -100000, 500000, None, None],
            #  ['time_weekday', '', False, False, False, True, 0, 7, None, None],    # additional agent states
            #  ['time_month', '', False, False, False, True, 0, 12, None, None],     # additional agent states
            # states
            #['s_price_electricity', '', False, False, False, True, -0.09, 0.17, None, None],     # additional agent states
            # ['s_price_gas', '', False, False, False, False, 0.015, 0.025, None, None],     # additional agent states
             ['s_actual_BESS', 'Out4', False, True, False, True, -200000, 200000, None, None],
             ['s_losses_BESS', 'Out7', False, True, False, True, 0, 2000000, None, None],
             ['s_temp_BESS', 'Out14', False, True, False, False, -5000000, 5000000, None, None],
             ['s_actual_KESS', 'Out5', False, True, False, True, -5000000, 5000000, None, None],
             ['s_losses_KESS', 'Out8', False, True, False, True, 0, 2000000, None, None],
            # ['s_temp_heatsupply', 'Temp_warmwater_before_production.T', False, True, False, False, 293, 363, None, None],
             ['s_SOC_BESS', 'Out1', False, True, False, True, 0, 1, None, None],
             ['s_SOC_KESS', 'Out2', False, True, False, True, 0, 1, None, None],
            # ['s_C_deg_BESS', 'Out16', False, True, False, True, 0, 1, None, None],
             ['d_usage', '', False, False, False, True, -100000, 200000, None, None],
             ['compensated_output','',False, False, False, True, -200000, 200000, None, None],
             ['relativ_usage','',False, False, False, True, -200000, 200000, None, None],
             #['p_usage_next', '', False, False, False, True, -100000, 100000, None, None],
             ['p_usage_15min', '', False, False, False, True, -100000, 100000, None, None],
             ['p_usage_5min', '', False, False, False, True, -100000, 100000, None, None],
             ['p_usage_1min', '', False, False, False, True, -100000, 100000, None, None]
            # ['P_gs_combinedheatpower', 'cHP.P_gs', False, True, False, mpc_state, 0, 200000, None, None],  # included in observation for MILP approach
            # ['s_u_combinedheatpower', 'cHP.s_u', False, True, False, True, 0, 1, None, None],
            # ['s_u_condensingboiler', 'condensingBoiler.s_u', False, True, False, True, 0, 1, None, None],
            # ['P_gs_condensingboiler', 'condensingBoiler.P_gs', False, True, False, mpc_state, 0, 7548000, None, None],  # added for MILP approach
            # ['s_u_immersionheater', 'immersionHeater.s_u', False, True, False, True, 0, 1, None, None],
            # ['P_el_immersionheater', 'immersionHeater.P_el', False, True, False, mpc_state, 0, 150000, None, None]  # added for MILP approach
            ],
            columns=['name','fmu_name', 'is_fmu_input', 'is_fmu_output', 'is_agent_action', 'is_agent_observation', 'low_value', 'high_value', 'abort_condition_min', 'abort_condition_max'])

        # save state_config to csv for info (only first environment)
        if self.env_id == 1 :
            self.state_config.to_csv(
                path_or_buf = os.path.join(self.path_results, self.run_name + '_state_config.csv'),
                sep = ';',
                decimal = '.')

        # extract several lists out of state_config for name mapping
        self.action_fmu_name = self.state_config.loc[self.state_config.is_agent_action == True].fmu_name.values
        self.names = {}
        self.names['actions'] = self.state_config.loc[self.state_config.is_agent_action == True].name.values
        self.names['observations'] = self.state_config.loc[self.state_config.is_agent_observation == True].name.values
        self.names['inputs'] = self.state_config.loc[self.state_config.is_fmu_input == True].name.values
        self.names['outputs'] = self.state_config.loc[self.state_config.is_fmu_output == True].name.values
        self.names['abort_conditions_min'] = self.state_config.loc[self.state_config.abort_condition_min.notnull()].name.values
        self.names['abort_conditions_max'] = self.state_config.loc[self.state_config.abort_condition_max.notnull()].name.values
        self.fmu_names = {}
        self.fmu_names['inputs'] = self.state_config.loc[self.state_config.is_fmu_input == True].fmu_name.values
        self.fmu_names['outputs'] = self.state_config.loc[self.state_config.is_fmu_output == True].fmu_name.values
        self.observation_low_values = dict(zip(self.names['observations'], self.state_config.loc[self.state_config.name.isin(self.names['observations'])].low_value.values))
        self.observation_high_values = dict(zip(self.names['observations'], self.state_config.loc[self.state_config.name.isin(self.names['observations'])].high_value.values))
        self.env_settings['abort_conditions_min'] = self.state_config.loc[self.state_config.abort_condition_min.notnull()].abort_condition_min.values
        self.env_settings['abort_conditions_max'] = self.state_config.loc[self.state_config.abort_condition_max.notnull()].abort_condition_max.values

        # initialize data stores (data_store must be in order of state_config)
        self.data_store_default = dict.fromkeys(self.state_config.name.values, 0)   # initialize with keys to fix order according to vars_df
        self.data_store = self.data_store_default.copy()
        self.reward_store = {}
        self.episode_store = {}

        # initialize archives
        self.data_archive = []
        self.episode_archive = []

        # initialize counters
        self.n_episodes = 0
        self.n_steps = 0

        # set integer prediction steps (15m,1h,6h)
        self.n_steps_next= 1
        self.n_steps_1m = 60 // self.sampling_time
        self.n_steps_5m = 300 // self.sampling_time
        self.n_steps_15m = 900 // self.sampling_time
        self.n_steps_1h = 3600 // self.sampling_time
        self.n_steps_6h = 21600 // self.sampling_time

        # initialize integrators
        self.P_el_total_15min_buffer = []
        self.P_gs_total_15min_buffer = []

        # initialize longtime stats
        self.n_steps_longtime = 0
        self.reward_longtime_average = 0

        # initialize episode timer
        self.episode_datetime_begin = datetime.now()
        self.episode_time_start = time.time()

        #eng = matlab.engine.start_matlab('-desktop background=True -r "shareMATLABForFMUCoSim"')
        # set FMU parameters for initialization
        ## no parameters to initialize
        self.fmu_parameters = {}
        #self.fmu_parameters['T_start_warmwater'] = self.np_random.uniform(self.env_settings['temperature_heat_init_min'], self.env_settings['temperature_heat_init_max'])
        #_ = self.np_random.uniform(self.env_settings['temperature_heat_init_min'], self.env_settings['temperature_heat_init_max'])  # necessary to get the same number of random calls to generate the same dates and temperatures for system a and b

        # load scenario files
        print("[INFO] Trying to load scenario files into environment (" + str(self.env_id) + ").")
        try:
            #self.scenario_factory_full = pd.read_csv(self.path_scenario_factory, index_col=0, parse_dates=True, sep = ';', decimal = '.')
            self.scenario_usage_full = pd.read_csv(self.path_scenario_usage, index_col=0, parse_dates=True, sep = ';', decimal = '.')
            self.scenario_markets_full = pd.read_csv(self.path_scenario_markets, index_col=0, parse_dates=True, sep = ';', decimal = ',')
            print("[INFO] Scenario files loaded successfully into environment (" + str(self.env_id) + ").")

        except EnvironmentError as error:
            print(
                "[ERROR] An error occured while loading the scenario files into environment (" + str(self.env_id) + ")."
                +"\n\t"+error.strerror
                +"\n\t"+error.filename
            )

        # resample to match sampling_time
       # self.scenario_factory_full = self.scenario_factory_full.resample(str(self.sampling_time)+'S').asfreq()
        self.scenario_usage_full = self.scenario_usage_full.resample(str(self.sampling_time)+'S').asfreq()
        self.scenario_markets_full = self.scenario_markets_full.resample(str(self.sampling_time)+'S').asfreq()

        # missing data adjustments in scenario data (fill gaps in market prices with previous prices and interpolate weather data)
        # self.scenario_factory_full = self.scenario_factory_full.interpolate(method='time')
        self.scenario_markets_full = self.scenario_markets_full.fillna(method='pad')
        self.scenario_usage_full = self.scenario_usage_full.fillna(method='pad')   
        

        # scale factory scenario (only power columns)
        #self.scenario_factory_full.iloc[:, 0] = self.scenario_factory_full.iloc[:, 0].multiply(self.env_settings['scenario_factory_scale_electric'])
        #self.scenario_factory_full.iloc[:, 3] = self.scenario_factory_full.iloc[:, 3].multiply(self.env_settings['scenario_factory_scale_gas'])
        #self.scenario_factory_full.iloc[:, 1] = self.scenario_factory_full.iloc[:, 1].multiply(self.env_settings['scenario_factory_scale_heat'])

        # initialize the FMUSimulator instance
        self.simulator = FMUSimulator(  self.env_id, self.path_fmu,
                                        start_time=0.0,
                                        stop_time=self.episode_duration,  # 1*24*60*60 = 1 day
                                        step_size=int(self.sampling_time/self.control_interval),
                                        names_inputs=self.fmu_names['inputs'],
                                        names_outputs=self.fmu_names['outputs'],
                                        init_values=self.fmu_parameters) 

        # define the lower and higher bound of observation_space for normalization issues
        self.n_observation_space = self.state_config.loc[self.state_config.is_agent_observation == True].name.count()
        if self.env_settings['discretize_state_space'] :
            # not supported currently
            print("[ERROR] Discrete state space is not supported currently. Please modify config settings.")
            quit()
        else :
            self.state_low = self.state_config.loc[self.state_config.is_agent_observation == True].low_value.values
            self.state_high = self.state_config.loc[self.state_config.is_agent_observation == True].high_value.values
            self.observation_space = spaces.Box(self.state_low, self.state_high, dtype=np.float32)

        # define the lower and higher bound of action_space for normalization issues
        self.n_action_space = self.state_config.loc[self.state_config.is_agent_action == True].name.count()
        self.action_low = self.state_config.loc[self.state_config.is_agent_action == True].low_value.values
        self.action_high = self.state_config.loc[self.state_config.is_agent_action == True].high_value.values
        if self.env_settings['discretize_action_space'] :
            # set 3 discrete actions (increase,decrease,equal) per control variable
            self.action_space = spaces.MultiDiscrete(np.full(self.n_action_space, 3))
            # customized for 
            self.action_disc_step = [[-1,-0.8,-0.6,-0.4,-0.2,-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8,1], [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]]     #BESS,KESS
            # initialize action
            self.action_disc_index = [0] * self.n_action_space
        else :
            self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)
        
        #calculate average use ETA
        #self.scenario_usage_mean  = self.scenario_usage_full.mean()
        #self.scenario_usage_std  = self.scenario_usage_full.std()
        

        # reset
        self.reset()


    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # backup data_store
        self.data_store_backup = self.data_store.copy()

        # convert discrete actions into continious space if discrete action space is chosen
        if self.env_settings['discretize_action_space'] :
            self.action = self.convert_disc_action(action)
        else :
            self.action = action

        # save actions into data_store
        for idx, item in enumerate(self.names['actions']):
            self.data_store[item] = self.action[idx]

        #policyshaping 
        self.policy_shaping_active = False
        if self.data_store['d_usage'] > self.env_settings['activation_energy'] + 130000 :
            self.data_store['u_powerbess'] = -1
            self.data_store['u_powerkess'] = -1
            self.policy_shaping_active = True
        
        #     
        # overwrite actions if out of boundaries (policy shaping)
        #T_heat_max = max(self.data_store['s_temp_heat_storage_hi'],self.data_store['s_temp_heatsupply']) - 273.15
        #T_heat_min = min(self.data_store['s_temp_heat_storage_hi'],self.data_store['s_temp_heatsupply']) - 273.15
        # T_heat_max = max(self.data_store['s_temp_heat_storage_hi'],self.data_store['s_temp_heatsupply'])
        # T_heat_min = min(self.data_store['s_temp_heat_storage_hi'],self.data_store['s_temp_heatsupply'])
        self.policy_shaping_active = False
        #if T_heat_max > 95 :
        # if T_heat_max >= self.env_settings['temperature_cost_prod_heat_max'] :
        #    self.data_store['u_combinedheatpower'] = 0
        #    self.data_store['u_condensingboiler'] = 0
        #    self.data_store['u_immersionheater'] = 0
        #    self.data_store['u_heatpump'] = 0
        #    self.policy_shaping_active = True
        #elif T_heat_min < 45 :
        # elif T_heat_min <= self.env_settings['temperature_cost_prod_heat_min'] :
        #    self.data_store['u_combinedheatpower'] = 1
        #    self.data_store['u_condensingboiler'] = 1
        #    self.data_store['u_immersionheater'] = 1
        #    self.data_store['u_heatpump'] = 0
        #    self.policy_shaping_active = True

        # get scenario input for current time step
        self.set_scenario_state(self.n_steps)

        # simulate one time step
        step_success = self.simulate()

        # calculate virtual states #todo auslagern
            # running 15min average electric power
        #self.P_el_total_15min_buffer.append(self.data_store['s_electric_power_total'])
        #if len(self.P_el_total_15min_buffer) > self.n_steps_15m :
        #    self.P_el_total_15min_buffer.pop(0)
        #self.data_store['vs_electric_power_total_15min'] = sum(self.P_el_total_15min_buffer) / len(self.P_el_total_15min_buffer)
            # running 15min average gas power
        #self.P_gs_total_15min_buffer.append(self.data_store['s_gas_power_total'])
        #if len(self.P_gs_total_15min_buffer) > self.n_steps_15m :
        #    self.P_gs_total_15min_buffer.pop(0)
        #self.data_store['vs_gas_power_total_15min'] = sum(self.P_gs_total_15min_buffer) / len(self.P_gs_total_15min_buffer)
        #    # daytime
        #self.data_store['vs_time_daytime'] += self.sampling_time/3600
        #if self.data_store['vs_time_daytime'] > 24 :
        #    self.data_store['vs_time_daytime'] -= 24
            # datestamp, see strptime documentation for timecodes
        #self.data_store['Date [%d-%m-%Y %H:%M]'] = self.episode_datetime_begin + timedelta(seconds=self.n_steps * self.sampling_time)

        # check for boundary and abort conditions within data_store
        step_abort = False
        for index, key in enumerate(self.names['abort_conditions_min']):
            if self.data_store[key] < self.env_settings['abort_conditions_min'][index] :
                step_abort = True
        if not step_abort :
            for index, key in enumerate(self.names['abort_conditions_max']):
                if self.data_store[key] > self.env_settings['abort_conditions_max'][index] :
                    step_abort = True

        # add debug infos into debug_store
        self.data_store['step_success'] = step_success
        self.data_store['step_abort'] = step_abort
        self.data_store['step_counter'] = self.n_steps

        # calculate reward
        reward = self.calc_reward()

        # append data stores to episode's archives
        self.data_archive.append(list(self.data_store.values()))

        # check if episode is over or not
        if self.n_steps >= self.n_steps_episode_end or step_abort or not step_success:
            done = True
        else:
            done = False

        # generate agent state from data_store
        state = []
        for key in self.names['observations']:
            if self.env_settings['normalize_state_space']:
                state.append( (self.data_store[key] - self.observation_low_values[key]) / (self.observation_high_values[key] - self.observation_low_values[key])) 
            else:
                state.append(self.data_store[key])

        # render episode when fail
        if not step_success or step_abort :
            self.reset()
            self.render(name_suffix='fail')

        # update counters
        self.n_steps += 1
        self.n_steps_longtime += 1

        # return info to agent
        return np.array(state), reward, done, {}


    def simulate(self):
        """
            takes data_store and does one simulator step. Results will be written back to data_store
        """
        # generate input_fmu_step dict from data_store
        step_inputs = []
        for key in self.names['inputs']:
            step_inputs.append(self.data_store[key])

        # start timer for simulation step time debugging
        step_time_start = time.time()

        # push the input to the FMU, simulate one timestep and receive the outputs from the FMU
        step_success = True
        for i in range(self.control_interval):  # do multiple FMU steps in one environment-step
            try:
                step_outputs = self.simulator.step(step_inputs, nr_substeps=self.control_interval)  # actually do the step in the FMU

            except Exception as e:
                step_success = False
                if self.verbose >= 2 :
                    print(e)
                pass

        # stop timer for simulation step time debugging
        step_time_stop = time.time()
        self.data_store['step_time'] = step_time_stop - step_time_start

        # save step_outputs into data_store
        if step_success :
            for idx,item in enumerate(self.names['outputs']):
                self.data_store[item] = step_outputs[idx]
        
        return step_success


    def set_scenario_state(self, step):
        """
            Calculates the step reward. Needs to be called from step() method after state update.
        """

        # production
        #print(self.episode_datetime_begin)
        #print(self.episode_time_end)
        self.data_store['d_usage'] = self.scenario_usage.iloc[step,0]  # [W] electric power demand of ETA
        self.data_store['relativ_usage'] = self.env_settings['activation_energy'] - self.data_store['d_usage'] 

        #print[self.scenario_usage_full]
        #self.data_store['d_production_electric_power'] = self.scenario_factory.iloc[step, 0]  # [W] electric power demand of production
        #self.data_store['d_production_gas_power'] = self.scenario_factory.iloc[step, 3]  # [W] gas consumed by production
        #self.data_store['d_production_heat_power'] = self.scenario_factory.iloc[step, 1]  # [W] heat power demand of production
        #self.data_store['d_time_till_availability'] = self.scenario_factory.iloc[step, 4]  # [W] heat power generation of production that has to be cooled
        # energy markets
        #self.data_store['s_price_electricity'] = self.scenario_markets.iloc[step, 0]/1000  # [€/kWh]
       # self.data_store['s_price_gas'] = self.scenario_markets.iloc[step, 1]/1000  # [€/kWh]

        # (perfect) scenario prediction for agent state
        self.data_store['p_usage_next'] = self.scenario_usage.iloc[step-1:step+1, 0].mean()
        self.data_store['p_usage_15min'] = self.scenario_usage.iloc[step:step+self.n_steps_15m, 0].mean() - self.data_store['d_usage']
        self.data_store['p_usage_5min'] = self.scenario_usage.iloc[step:step+self.n_steps_5m, 0].mean() - self.data_store['d_usage']
        self.data_store['p_usage_1min'] = self.scenario_usage.iloc[step:step+self.n_steps_1m, 0].mean() - self.data_store['d_usage']
        #self.data_store['p_production_delta_heat_power_1h'] = self.scenario_factory.iloc[step:step+self.n_steps_1h, 1].mean() - self.data_store['d_production_heat_power'] # [W] heat power demand of production
        #self.data_store['p_production_delta_heat_power_6h'] = self.scenario_factory.iloc[step:step+self.n_steps_6h, 1].mean() - self.data_store['d_production_heat_power'] # [W] heat power demand of production
        #self.data_store['p_production_delta_electric_power_15m'] = self.scenario_factory.iloc[step:step+self.n_steps_15m, 0].mean() - self.data_store['d_production_electric_power'] # [W] electric power demand of production
        #self.data_store['p_production_delta_electric_power_1h'] = self.scenario_factory.iloc[step:step+self.n_steps_1h, 0].mean() - self.data_store['d_production_electric_power'] # [W] electric power demand of production
        #self.data_store['p_production_delta_electric_power_6h'] = self.scenario_factory.iloc[step:step+self.n_steps_6h, 0].mean() - self.data_store['d_production_electric_power'] # [W] electric power demand of production
        #self.data_store['p_delta_price_electricity_15m'] = self.scenario_markets.iloc[step:step+self.n_steps_15m, 0].mean()/1000  - self.data_store['s_price_electricity'] # [€/kWh]
        #self.data_store['p_delta_price_electricity_1h'] = self.scenario_markets.iloc[step:step+self.n_steps_1h, 0].mean()/1000 - self.data_store['s_price_electricity'] # [€/kWh]
        #self.data_store['p_delta_price_electricity_6h'] = self.scenario_markets.iloc[step:step+self.n_steps_6h, 0].mean()/1000 - self.data_store['s_price_electricity'] # [€/kWh]
        # self.data_store['p_production_delta_heat_power_1h'] = self.scenario_factory.iloc[step+self.n_steps_1h, 1] - self.data_store['d_production_heat_power'] # [W] heat power demand of production
        # self.data_store['p_production_delta_heat_power_6h'] = self.scenario_factory.iloc[step+self.n_steps_6h, 1] - self.data_store['d_production_heat_power'] # [W] heat power demand of production
        # self.data_store['p_production_delta_electric_power_15m'] = self.scenario_factory.iloc[step+self.n_steps_15m, 0] - self.data_store['d_production_electric_power'] # [W] electric power demand of production
        # self.data_store['p_production_delta_electric_power_1h'] = self.scenario_factory.iloc[step+self.n_steps_1h, 0] - self.data_store['d_production_electric_power'] # [W] electric power demand of production
        # self.data_store['p_production_delta_electric_power_6h'] = self.scenario_factory.iloc[step+self.n_steps_6h, 0] - self.data_store['d_production_electric_power'] # [W] electric power demand of production
        # self.data_store['p_delta_price_electricity_15m'] = self.scenario_markets.iloc[step+self.n_steps_15m, 0]/1000  - self.data_store['s_price_electricity'] # [€/kWh]
        # self.data_store['p_delta_price_electricity_1h'] = self.scenario_markets.iloc[step+self.n_steps_1h, 0]/1000 - self.data_store['s_price_electricity'] # [€/kWh]
        # self.data_store['p_delta_price_electricity_6h'] = self.scenario_markets.iloc[step+self.n_steps_6h, 0]/1000 - self.data_store['s_price_electricity'] # [€/kWh]

        return


    def calc_reward(self):
        """Calculates the step reward. Needs to be called from step() method after state update.

        :return: Normalized or non-normalized reward
        :rtype: Real
        """   
        #battery use
        if  self.data_store['s_SOC_BESS'] < 0.5  :  # self.scenario_usage_mean:
            self.data_store['reward_storage_status'] = - math.exp(0.5 - self.data_store['s_SOC_BESS']) 
        else:
            self.data_store['reward_storage_status'] =int(0)
        
        # energy loss
        self.data_store['reward_energy_loss'] = - (self.data_store['s_losses_BESS'] + self.data_store['s_losses_KESS'])

        #reward compesation 
        self.data_store['output'] =self.data_store['d_usage'] + self.data_store['s_actual_BESS'] + self.data_store['s_actual_KESS'] - self.data_store['reward_energy_loss']       

        if  self.data_store['output'] > self.env_settings['activation_energy'] :  
            self.data_store['reward_energy_compensated'] =-abs((self.env_settings['activation_energy']  -  self.data_store['output'])/1000)**2 - 400
        else:
            self.data_store['reward_energy_compensated'] =int(0)
        
        #reference compesation 

        if  self.data_store['output'] > self.env_settings['activation_energy'] :  
            self.data_store['reference_energy_compensated'] =-abs(self.env_settings['activation_energy']  -  self.data_store['output'])
        else:
            self.data_store['reference_energy_compensated'] =int(0)


        #reference compensation withoutstorage
        self.data_store['output1'] =self.data_store['d_usage'] 

        if  self.data_store['output1'] > self.env_settings['activation_energy'] : 
            self.data_store['reward_energy_withoutcompensated'] =-(abs(self.env_settings['activation_energy']  -  self.data_store['output1']))
        else:
            self.data_store['reward_energy_withoutcompensated'] =int(0) 

        #diff
        self.data_store['comp_diff'] = self.data_store['reward_energy_withoutcompensated'] - self.data_store['reference_energy_compensated']
                
        # energy costs
        base_power_electric =  self.data_store['output'] # total consumption 
        #self.data_store['reward_energy_electric'] = -self.data_store['s_price_electricity']*base_power_electric*self.sampling_time/3600/1000    
        

        # other costs
            # abort costs
        self.data_store['reward_other'] = 0
            # policyshaping costs
        if self.policy_shaping_active :
            self.data_store['reward_other'] -= self.env_settings['policyshaping_costs']

        # total reward
        self.data_store['reward_total'] =   self.data_store['reward_energy_compensated'] + self.data_store['reward_other'] 
                                            # self.data_store['reward_energy_loss'] + \
                                            # self.data_store['reward_other'] + \
                                            # self.data_store['reward_energy_electric'] + \
                                            
                                           # self.data_store['reward_storage_status']

        if self.env_settings['normalize_reward'] :
            # total normalized costs (normalized by cumulative moving average)
            self.reward_longtime_average += (self.data_store['reward_total'] - self.reward_longtime_average) / self.n_steps_longtime
            self.data_store['reward_total_norm'] = self.data_store['reward_total'] / abs(self.reward_longtime_average)
            # return normalized reward
            return self.data_store['reward_total_norm']
        else :
            self.data_store['reward_total_norm'] = 0
            return self.data_store['reward_total']


    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """    

        # save episode's stats
        if self.n_steps > 0 :

            # increase counter for successfull episodes
            self.n_episodes += 1

            # create episode dataframe
            self.episode_df = pd.DataFrame(
                self.data_archive,
                columns=list(self.data_store.keys())
                )

            self.episode_store['datetime_begin'] = self.episode_datetime_begin.strftime('%Y-%m-%d %H:%M')
            self.episode_store['rewards_total'] = self.episode_df['reward_total'].sum()
            self.episode_store['rewards_total_norm'] = self.episode_df['reward_total_norm'].sum()
            self.episode_store['rewards_storage_status'] = self.episode_df['reward_storage_status'].sum()
            self.episode_store['rewards_energy_loss'] = self.episode_df['reward_energy_loss'].sum()
            #self.episode_store['rewards_energy_electric'] = self.episode_df['reward_energy_electric'].sum()
            self.episode_store['rewards_energy_compensated'] = self.episode_df['reward_energy_compensated'].sum()
            self.episode_store['rewards_other'] = self.episode_df['reward_other'].sum()
            self.episode_store['rewards_other'] = self.episode_df['reward_other'].sum()
            self.episode_store['time_steps'] = self.episode_df['step_time'].sum()
            self.episode_store['comp_diffs'] = self.episode_df['comp_diff'].sum()/3600
            self.episode_store['time'] = time.time() - self.episode_time_start
            self.episode_store['n_steps'] = self.n_steps
            
            self.episode_store['average_SOC_BESS']    = self.episode_df['s_SOC_BESS'].mean()
            self.episode_store['average_SOC_KESS']    = self.episode_df['s_SOC_KESS'].mean()
            self.episode_store['charge_cycle_KESS']   = self.episode_df['s_SOC_KESS'].diff().abs().sum()
            self.episode_store['charge_cycle_BESS']   = self.episode_df['s_SOC_BESS'].diff().abs().sum()
            self.episode_store['compensated']         = self.episode_df['reward_energy_compensated'].sum()  
            self.episode_store['max_last']         = self.episode_df['output'].max()  
            # append episode store to episodes' archive
            self.episode_archive.append(list(self.episode_store.values()))

            # callback if defined at initialization
            if self.callback is not None:
                self.callback(self)

        # reset data store
        self.data_store = self.data_store_default.copy()

        # set new FMU parameters and state for initialization
        self.fmu_parameters['init_SOC_BESS'] = self.np_random.uniform(self.env_settings['SOC_low'], self.env_settings['SOC_high'])
        self.fmu_parameters['init_SOC_KESS'] = self.np_random.uniform(self.env_settings['SOC_low'], self.env_settings['SOC_high'])
        #self.fmu_parameters['T_start_warmwater'] = self.np_random.uniform(self.env_settings['temperature_heat_init_min'], self.env_settings['temperature_heat_init_max'])
        #_ = self.np_random.uniform(self.env_settings['temperature_heat_init_min'], self.env_settings['temperature_heat_init_max'])  # necessary to get the same number of random calls to generate the same dates and temperatures for system a and b
        #self.data_store['s_temp_heat_storage_hi'] = self.fmu_parameters['T_start_warmwater']
        #self.data_store['s_temp_heat_storage_lo'] = self.fmu_parameters['T_start_warmwater']-self.env_settings['temperature_difference_top_bottom']
        #self.data_store['s_temp_heatsupply'] = self.fmu_parameters['T_start_warmwater']

        # reset the FMU after every episode with new parameters
        self.simulator.reset(self.fmu_parameters)

        # get random time within scenario range
        dt = self.env_settings['scenario_time_begin'] + self.np_random.uniform()*(self.env_settings['scenario_time_end']-self.env_settings['scenario_time_begin'])
        # round random time down to last 15-minute interval for episode begin time
        self.episode_datetime_begin = datetime(dt.year, dt.month, dt.day, dt.hour, 15*(dt.minute // 15))
        self.episode_datetime_begin_seconds = self.episode_datetime_begin.timetuple().tm_yday * 24*60*60 + self.episode_datetime_begin.hour*60*60 + self.episode_datetime_begin.minute*60
        # calculte episode end time which includes a 24h addon for forecast values
        self.episode_time_end = self.episode_datetime_begin + timedelta(hours=24, seconds=self.episode_duration)
        # cut out the scenario data for episode time range from from full scenario dataframe
        #self.scenario_factory = self.scenario_factory_full[self.episode_datetime_begin:self.episode_time_end]
        self.scenario_usage = self.scenario_usage_full[self.episode_datetime_begin:self.episode_time_end]
        self.scenario_markets = self.scenario_markets_full[self.episode_datetime_begin:self.episode_time_end]
        # set time of day from episode_datetime_begin
        self.data_store['vs_time_daytime'] = self.episode_datetime_begin.hour + self.episode_datetime_begin.minute/60

        # get scenario input for initialization (time step: 0)
        self.set_scenario_state(0)

        # reset virtual states and internal counters
        self.n_steps = 0
        self.P_el_total_15min_buffer = []
        self.P_gs_total_15min_buffer = []
        #self.data_store['vs_electric_power_total_15min'] = 0
        #self.data_store['vs_gas_power_total_15min'] = 0

        # reset maximal peak electric power for penalty costs (necessary for peak shaving)
        #self.max_limit = self.env_settings['power_cost_max']

        # get start states by simulating one time step
        self.simulate()

        # generate agent state from data_store
        state = []
        for key in self.names['observations']:
            if self.env_settings['normalize_state_space']:
                state.append( (self.data_store[key] - self.observation_low_values[key]) / (self.observation_high_values[key] - self.observation_low_values[key])) 
            else:
                state.append(self.data_store[key])

        # empty archive lists for new episode
        self.data_archive = []

        # start timer for episode calculation duration
        self.episode_time_start = time.time()

        # return default_state to agent
        return np.array(state)


    def convert_disc_action(self, action_disc):
        """
        converts discrete actions from agent to continious FMU input space
        """
        action = []
        self.action_disc_index

        for idx, val in enumerate(action_disc):
            self.action_disc_index[idx] = np.clip(self.action_disc_index[idx] + (val-1),0,len(self.action_disc_step[idx])-1)
            action.append(self.action_disc_step[idx][self.action_disc_index[idx]])

        return action


    def get_info(self):
        """
        get info about environment
        Returns
        -----
        version : (str)
        description : (str)
        """
        return self.version, self.description


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
        episode_archive_df = pd.DataFrame(
            self.episode_archive,
            columns=list(self.episode_store.keys())
        )

        # write all data to csv after every episode
        episode_archive_df.to_csv(
            path_or_buf = os.path.join(self.path_results, self.run_name + "_" + str(self.n_episodes).zfill(4) + "-" + str(self.env_id).zfill(2) + '_all-episodes.csv'),
            sep = ';',
            decimal = '.')

        # write another aggregated csv that contains all episodes (necessary for mpc and mpc_simple)
        csvpath = os.path.join(self.path_results, 'all-episodes.csv')
        if os.path.exists(csvpath): # check if aggregated file already exists, which is the case when multiple runs are done with mpc and mpc_simple
            tocsvmode = 'a'
            tocsvheader = False
        else:
            tocsvmode = 'w'
            tocsvheader = True
        # write data to csv
        episode_archive_df.tail(1).to_csv(
            path_or_buf = csvpath,
            sep = ';',
            decimal = '.',
            mode=tocsvmode,
            header=tocsvheader)

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
        axes[0].plot(x, y['rewards_energy_compensated'], label='Kompensation', color=(0.929,0.49,0.192), linewidth=1, alpha=0.9)
        #axes[0].plot(x, y['rewards_energy_electric'], label='Steuern & Umlagen', color=(0.184,0.333,0.592), linewidth=1, alpha=0.9)
        axes[0].plot(x, y['rewards_storage_status'], label='Speicher Stand', color=(0.65,0.65,0.65), linewidth=1, alpha=0.9)
        axes[0].set_ylabel('kum. Kosten')
        axes[0].set_xlabel('Episode')
        axes[0].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        axes[0].margins(x=0.0,y=0.1)
        axes[0].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)

        # (2) Rewards
        cost_loss = y['rewards_energy_loss']/(3600) 
        axes[1].plot(x, cost_loss, label='Verluste', color=(0.65,0.65,0.65), linewidth=1, alpha=0.9)
        axes[1].set_ylabel('Wh')
        axes[1].set_xlabel('Episode')
        axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        axes[1].margins(x=0.0,y=0.1)
        axes[1].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)

        plt.savefig(os.path.join(self.path_results, self.run_name + "_" + str(self.n_episodes).zfill(4) + "-" + str(self.env_id).zfill(2) + '_all-episodes.png'))
        plt.close(figure)


    def render(self, mode='human', name_suffix=''):
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
            path_or_buf = os.path.join(self.path_results, self.run_name + "_" + str(self.n_episodes).zfill(3) + "-" + str(self.env_id).zfill(2) + '_episode' + name_suffix + '.csv'),
            sep = ';',
            decimal = '.')
        

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
        axes[0].set_yticks(np.arange(len(self.names['actions'])))
        axes[0].set_yticklabels(['BESS', 'KESS'
                                                                    ])
        im = axes[0].imshow(y[self.names['actions']].transpose(), cmap="hot", vmin=-1, vmax=1, aspect="auto")
        # add colorbar
        ax_pos = axes[0].get_position().get_points().flatten()
        ax_colorbar=figure.add_axes([0.93,ax_pos[1]+0.05,0.01,ax_pos[3]-ax_pos[1]-0.1])  ## the parameters are the specified position you set 
        figure.colorbar(im, ax=axes[0], shrink=0.9, cax=ax_colorbar)

        timeRange = np.arange((1-dt_begin.minute/60)*60*60/sampling_time, self.episode_duration/sampling_time, 1*60*60/sampling_time)
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
        axes[0].set_yticks(np.arange(len(self.names['actions'])+1)-.5, minor=True)
        axes[0].tick_params(which="minor", bottom=False, left=False)
        # grid settings
        axes[0].grid(which="minor", color="w", linestyle='-', linewidth=3)
        axes[0].xaxis.grid(color=(1,1,1,0.1), linestyle='-', linewidth=1)
        # add ticks and tick labels
        axes[0].set_xticks(tickpos)
        axes[0].set_xticklabels(ticknames)
        # Rotate the tick labels and set their alignment.
        plt.setp(axes[0].get_yticklabels(), rotation=30, ha="right", va="center", rotation_mode="anchor")

        # (2) - Lasten
        axes[1].plot(x,y['d_usage'], color='#1f77b4', label='Lasten')
        axes[1].plot(x,y['output'], color='#ff7f0e', label='Lasten mit Kompensation',  alpha=0.6)
        axes[1].plot(x, [self.env_settings['activation_energy']]* len(x), color='grey', linestyle='--', linewidth=1, alpha=0.5, label='Lastgrenze')
        #axes[1].plot(x, 20000 * len(x), color='red', linestyle='--', linewidth=1, alpha=0.5, label='Lastgrenze regelbasiert ')
        #axes[1].plot(x, 5000 * len(x), color='grey', linestyle='--', linewidth=1, alpha=0.5, label='Lastgrenze regelbasiert')
        # settings
        axes[1].set_ylabel('Leistung [W]')
        axes[1].margins(x=0.0,y=0.1)
        axes[1].set_axisbelow(True)
        axes[1].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)
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
        axes[2].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)
        axes[2].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        axes[2].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)

# (3) - reward_compansated
        axes[3].plot(x,y['s_SOC_KESS'], color='#1f77b4', label='SOC KESS')
        axes[3].plot(x,y['s_SOC_BESS'], color='#ff7f0e', label='SOC BESS')
        # settings
        axes[3].set_ylabel('Ladestand')
        axes[3].margins(x=0.0,y=0.1)
        axes[3].set_axisbelow(True)
        axes[3].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)
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
        # axes[4].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)
# (4) - kompensiert
        axes[4].plot(x, y['reference_energy_compensated'].abs().cumsum()/3600, color='#1f77b4', label=' - mit speicher')
        axes[4].plot(x, y['reward_energy_withoutcompensated'].abs().cumsum()/3600, color='#ff7f0e', label=' - ohne speicher')
        axes[4].plot(x, y['comp_diff'].cumsum()/3600 , color='gray', label='Differenz')
        axes[4].set_ylabel('zu kompensierende Leistung [Wh]')
        axes[4].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        axes[4].margins(x=0.0,y=0.1)
        axes[4].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        axes[4].set_axisbelow(True)
        axes[4].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)

        #axes[5].plot(x,y['s_losses_KESS'], color='#1f77b4', label='Leistung KESS')
        # axes[5].plot(x,y['s_losses_BESS'], color='#ff7f0e', label='Leistung BESS')
        # #axes[2].plot(x,y['reward_energy_loss'], color=(0,0,192/255), label='Leistungsverluste')
        # # settings
        # axes[5].set_ylabel('Leistung [W]')
        # axes[5].margins(x=0.0,y=0.1)
        # axes[5].set_axisbelow(True)
        # axes[5].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)
        # axes[5].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        # axes[5].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        textverlust='Sum_Verluste BESS: ' + str(round(self.episode_df['s_losses_BESS'].sum()/3600,2))+'Wh / Sum_Verluste_KESS: ' + str(round(self.episode_df['s_losses_KESS'].sum()/3600,2))+'Wh'
        texttemp=' / BESS temp.:'+ str(round(self.episode_df['s_temp_BESS'].mean(),2)) +'C°'
        textzyklen=' / BESS Zyklen:'+ str(round(self.episode_df['s_SOC_BESS'].diff().abs().sum(),2))
        textausgabe=textverlust + texttemp + textzyklen
        axes[5].text(0.5,0.5,textausgabe, ha='left', va='center', fontsize=13, color='black')
        
# (5) - kompensiert

        axes[6].plot(x, y['reward_total'] , color='#1f77b4', label='reward gesamt')
        axes[6].plot(x, y['reward_energy_compensated'] , color='#ff7f0e', label='reward compensation', alpha=0.6)
        axes[6].plot(x, y['reward_other'] , color='grey', label='reward other')
        axes[6].set_ylabel('[-]')
        axes[6].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        axes[6].margins(x=0.0,y=0.1)
        axes[6].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        axes[6].set_axisbelow(True)
        axes[6].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)

        # (4) - Plot power
        #axes[4].plot(x, y['s_electric_power_total']*1e-3, color=(1.0,0.75,0.0), linewidth=2, alpha=0.5, label='Strom Netz')
        #axes[3].plot(x, y['vs_electric_power_total_15min']*1e-3, color=(1.0,0.75,0.0), linewidth=2, alpha=0.9, label='Strom Netz (Ø15m)')
        #axes[3].plot(x, y['vs_gas_power_total_15min']*1e-3, color=(0.65,0.65,0.65), linewidth=2, alpha=0.9, label='Erdgas Netz (Ø15m)')
        #axes[3].plot(x, y['d_production_heat_power']*1e-3, color=(0.75,0,0), linestyle='--', linewidth=1, alpha=0.9, label='Wärmelast Prod.')
        #axes[3].plot(x, y['d_production_electric_power']*1e-3, color=(1.0,0.75,0.0), linestyle='--', linewidth=1, alpha=0.9, label='Strom Prod.')
        #axes[3].plot(x, y['d_production_gas_power']*1e-3, color=(0.65,0.65,0.65), linestyle='--', linewidth=1, alpha=0.9, label='Erdgas Prod.')
        #axes[3].fill_between(x, (
            #y['s_electric_power_total']-y['d_production_electric_power'])*1e-3,
            #label='Strom TGA',
            #color=(1.0,0.75,0.0), linestyle='--', linewidth=0, alpha=0.4)
        #axes[3].fill_between(x, (
          #  y['s_gas_power_total']-y['d_production_gas_power'])*1e-3,
           # label='Erdgas TGA',
            #color=(0.65,0.65,0.65), linestyle='--', linewidth=0, alpha=0.4)

        #axes[3].set_ylabel('Leistung [kW]')
        #axes[3].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        #axes[3].margins(x=0.0,y=0.1)
        #axes[3].tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
        #axes[3].set_axisbelow(True)
        #axes[3].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)


        # (5) - Costs
        #axes[5].plot(x, y['reward_energy_electric'].cumsum(), label='Strom (netto)', color=(1.0,0.75,0.0), linewidth=1, alpha=0.9)
        #axes[4].plot(x, y['reward_energy_gas'].cumsum(), label='Erdgas (netto)', color=(0.65,0.65,0.65), linewidth=1, alpha=0.9)
        #axes[4].plot(x, y['reward_energy_taxes'].cumsum(), label='Steuern & Umlagen', color=(0.184,0.333,0.592), linewidth=1, alpha=0.9)
        #axes[4].plot(x, y['reward_power_electric'].cumsum(), label='el. Lastspitzen', color=(0.929,0.49,0.192), linewidth=1, alpha=0.9)
        #axes[4].set_ylabel('kum. Kosten [€]')
        #axes[4].legend(bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1, fontsize='x-small')
        #axes[4].margins(x=0.0,y=0.1)
        #axes[4].set_axisbelow(True)
        #axes[4].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)
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
        # axes[5].grid(color=(0.9,0.9,0.9,0.1), linestyle='-', linewidth=1)
        # # add ticks and tick labels
        # axes[5].set_xticks(tickpos)
        # axes[5].set_xticklabels(ticknames, rotation=45)

        # save and close figure
        plt.savefig(os.path.join(self.path_results, self.run_name + "_" + str(self.n_episodes).zfill(3) + "-" + str(self.env_id).zfill(2) + '_episode' + name_suffix + '.png'))
        plt.close(figure)

        return


    def close(self):
        """
        close and clean up the environment
        """
        # close the fmu-session
        self.simulator.close()
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

