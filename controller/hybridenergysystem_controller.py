import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

class SimpleControl(BaseAlgorithm):

    def __init__(self,
                 policy,
                 env,
                 policy_base=BasePolicy,
                 learning_rate=1,
                 verbose=1,
                 _init_setup_model: bool = True,
                 ):
        """
        Simple TestAgent to test environments
        
        Parameters
        -----
        policy : (obj)
            The policy model to use (not relevant here)
        param env : (obj)
            The environment to learn from
        """

        self.env = env
        #
        super(SimpleControl, self).__init__(
            policy,
            env,
            BasePolicy,
            learning_rate=1,
            verbose=1)

        if _init_setup_model:
            self._setup_model()


    def _setup_model(self) -> None:
        super(SimpleControl, self)._setup_model()
        # set environment
        self.set_env(self.env)
        # get names of actions and observations
        # self.names_actions = self.env.envs[0].names['actions']                # before Monitor(..) Class was wrapped around Environments
        # self.names_observations = self.env.envs[0].names['observations']      # before Monitor(..) Class was wrapped around Environments
        self.names_actions = self.env.envs[0].names['actions']
        self.names_observations = self.env.envs[0].names['observations']
        # set initial state
        self.initial_state = np.zeros(self.action_space.shape)


        

    @staticmethod
    def hysteresis_control(u0, y, limit, invert=False):
        """ simple hysteresis control law
        Parameters
        ------
        u0: (float)
            current/old setpoint value
        y: (float or tuple of floats)
            current controlled value
        limits: (tuple of floats)
            lower and upper limit
        """
        # set controlled value both for lower and upper limit if only one value is given
        actual = [0,0]
        if not hasattr(y, "__len__") :
            actual[0] = y
            actual[1] = y
        else :
            actual[0] = y[0]
            actual[1] = y[1]

        # currently off
        if u0 <= 0 :
            if invert : # cooling mode
                # switch ON to DECREASE control variable when UPPER limit is reached - otherwise stay OFF
                return (actual[0] > limit[0]) * 1.0
            else : # heating mode
                # switch ON to INCREASE control variable when LOWER limit is reached - otherwise stay OFF
                return (actual[0] < limit[0]) * 1.0

        # currently on
        else :
            if invert : # cooling mode
                # stay ON to DECREASE control variable until LOWER limit is reached - then switch OFF
                return (actual[1] > limit[1] and not actual[0] < limit[2]) * 1.0
            else : # heating mode
                # stay ON to INCREASE control variable until UPPER limit is reached - then switch OFF
                return (actual[1] < limit[1] and not actual[0] > limit[2]) * 1.0


    def predict(self, observation, state=None, mask=None, deterministic=False):

        # make sure observation is a numpy array
        observation = np.array(observation)
        
        # handle multiple parallel environments or not
        observations = []
        vectorized_env = True  # self._is_vectorized_observation(observation, self.observation_space)
        if not vectorized_env :
            observations[0] = observation
        else :
            observations = observation

        # reset actions
        actions = []
        for observation in observations:
            # convert observation array to "human-readable" dictionary with keys
            observation = dict( zip(self.names_observations,  observation) )
            # initialize action dictionary
            action = dict.fromkeys(self.names_actions, 0)

            # control variables 
            verluste = observation['s_losses_BESS'] + observation['s_losses_KESS']
            consumption = observation['p_usage_next'] + verluste
            


            #################
            # CONTROL RULES #
            #################

            # Aufladen
            if consumption < 5000 and observation['s_SOC_KESS'] > 0.88 and observation['compensated_output']<20000 :
                action['u_powerbess'] = 0.3
            else:   
                action['u_powerbess'] = 0

            if consumption < 5000 and observation['compensated_output']<20000  :
                    action['u_powerkess'] = 0.3
            else:
                    action['u_powerkess'] = 0.0 

            # Entladen
            if consumption > 24000  and observation['s_SOC_KESS'] < 0.4 :               
                if consumption > 36000:
                    action['u_powerbess'] = - 1.0
                else:
                    action['u_powerbess'] = -(consumption - 24000)/120000        #12kW BESS 


            if consumption > 24000: 
                if consumption > 64000:
                    action['u_powerkess'] = - 1.0
                else:
                    action['u_powerkess'] = -(consumption - 24000)/40000        #40kW KESS

            # if observation['u_powerbess'] <= -1 : # aufladen
            #     action['u_powerbess'] = (consumption < 5000) * 1.0
            # else : #already on
            #     action['u_powerbess'] = (consumption < 22000) * 1.0

            # condensingboiler
            #if observation['s_u_condensingboiler'] <= 0 : # off
            #    action['u_powerkess'] = (temp_heat_hi < 71) * 1.0
            #else : #already on
            #    action['u_powerkess'] = (temp_heat_lo < 68 and temp_heat_hi < 85) * 1.0


            actions.append( list(action.values()) )

        states = state
        actions = np.array(actions, dtype=np.float32)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, states


    def load(self, load_path, env=None, **kwargs):
        """
            load method
        """
        print("[INFO] You called load() but there is nothing to load in static TestAgent.")
        return self


    def save(self, save_path):
        """
            load method
        """
        print("[INFO] You called save() but there is nothing to save in static TestAgent.")


    def _get_pretrain_placeholders(self):
        """
        Return the placeholders needed for the pretraining:
        - obs_ph: observation placeholder
        - actions_ph will be population with an action from the environement
            (from the expert dataset)
        - deterministic_actions_ph: e.g., in the case of a gaussian policy,
            the mean.

        :return: ((tf.placeholder)) (obs_ph, actions_ph, deterministic_actions_ph)
        """
    
    def get_parameter_list(self):
        """
        Get tensorflow Variables of model's parameters

        This includes all variables necessary for continuing training (saving / loading).

        :return: (list) List of tensorflow Variables
        """
        pass

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="SIC"):
        """
            learn method
        """
        print("[INFO] You called learn() but there is nothing to learn in static TestAgent.")


    def setup_model(self):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """
        print("[INFO] You called setup_model() but there is nothing to setup in static TestAgent.")


    def action_probability(self, observation, state=None, mask=None):
        """
        Get the model's action probability distribution from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :return: (np.ndarray) the model's action probability distribution
        """
        print("[INFO] You called action_probability() but there is no action_probability in static TestAgent.")


class PID():
    
    def __init__(self,kp = 0.5, ki = 0.1, kd = 0, dt = 60, set_point = 0, windupMax=0):
        
        """ Initialize PID object 
            # PID parameters:
            Set PID parameters. By setting some parameters to zero you can get
            a P or PI controller (or even a pure integrator if you wish)
            
            # Set point: Constant value to be reached by the measured variable y
            Set point is set to zero by default
            
            # outValue: last output value returned from the output method
            
            # windupMax: threshold for the anti-windup to kick in. If set to zero
            no anti-windup system is provided. Default value is zero.
        
        """
        
        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Set point is set to zero by default
        self.set_point = set_point
        
        # PID terms
        self.pTerm = 0
        self.iTerm = 0
        self.dTerm = 0
        
        # Sampling time
        self.dt = dt
        
        # PID last output value returned     
        self.outValue = 0
        
        # Last y measured
        self.last_y = 0
    
        # Anti windup        
        self.windupMax = windupMax
           
    def output(self,y_measured):

        """ 
            Calculate PID output value 
            
            Formula:
            outValue(t) = kd * e(t) + ki * cumulativesum(e(t) * dt) + kd * de(t)/dt
            
            Note:
            if the set point is constant, then de(t)/dt = -dy/dt
            this little tweak helps us avoiding impulsive terms (spikes) due to 
            the derivative of the error (since the error changes instantly when
            switching the set point, its derivative ends up being infinite).        
        """
        
        # Calculate current error
        error = self.set_point - y_measured
        
        # Calculate output
        # P term
        self.pTerm = self.kp * error
        # I term            
        self.iTerm += self.ki * error * self.dt
        # D term
        self.dTerm = self.kd * (self.last_y - y_measured)/self.dt
        
        # Check for windup problems if anti-windup is enabled
        self.antiWindUp()
        
        # Update variables
        self.last_y = y_measured
        
        # Output value to be returned
        self.outValue = self.pTerm + self.iTerm + self.dTerm
        
        return self.outValue
    
    def antiWindUp(self):

        """ 
            Anti wind-up 
    
            As far as the anti wind-up system is concerned, I implemented it as 
            it is explained in most control theory books. It only operates on 
            the integral term. It does not cap the proportional and the 
            derivative terms.
        """
        
        if self.windupMax != 0:
            if self.iTerm > self.windupMax:
                self.iTerm = self.windupMax
            elif self.iTerm < -self.windupMax:
                self.iTerm = -self.windupMax