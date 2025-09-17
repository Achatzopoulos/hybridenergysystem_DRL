import os
#import matlabrun
from eta_utility.eta_x import ETAx
from eta_utility import get_logger
from eta_utility.eta_x.common.schedules import LinearSchedule

log = get_logger()


"""[In this file lists of experiments to be performed can be created.]
"""

if __name__ == '__main__':  # necessary for multiprocessing
    
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - -##
    ##                          DRL                         ##
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - -##

    root_path = os.path.abspath(os.path.dirname(__file__))  # Necessary to get corrert path to this file
    sched_LR = LinearSchedule(0.0002, 0.00002)  # (star value, end value)

    # TRAIN on System A
    # myETAx = ETAx(root_path=root_path,
    #               config_name='hybridenerysystem_ppo', relpath_config='config/',
    #               config_overwrite={'settings': {'n_environments': 4, 'n_episodes_learn': 10000,
    #                                              'n_episodes_play': 100, 'episode_duration': 259200},
    #                                 'agent_specific': {'learning_rate': sched_LR.value,
    #                                                    'batch_size': 256,
    #                                                    'policy_kwargs': {'net_arch': [500, dict(pi=[400, 300], vf=[400, 300])]}},
    #                                 'environment_specific': {'plot_interval': 100}})
    # myETAx.learn('hybridenergysystem_DRL', 'train', '30 seconds sampling time - 3000 episodes each')

    # PLAY on System A
    myETAx = ETAx(root_path=root_path,
                  config_name='hybridenergysystem_ppo', relpath_config='config/',
                  config_overwrite={'settings': {'n_environments': 1, 'n_episodes_learn': 0,
                                                 'n_episodes_play':10, 'episode_duration': 86400},
                                                 'agent_specific': {'learning_rate': sched_LR.value,
                                                       'batch_size': 256,
                                                       'policy_kwargs': {
                                                           'net_arch': [500, dict(pi=[400, 300], vf=[400, 300])]}},
                                                           'environment_specific': {'plot_interval': 1}})
    myETAx.play('hybridenergysystem_DRL_play', 'play_9', 'dayplay1')