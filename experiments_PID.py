import os
from eta_utility.eta_x import ETAx
from eta_utility import get_logger

log = get_logger()


"""[In this file lists of experiments to be performed can be created.]
"""

if __name__ == '__main__':  # necessary for multiprocessing
    
    root_path = os.path.abspath(os.path.dirname(__file__))  # Necessary to get corrert path to this file

    ## - - - - - - - - - - - - - - - - - - - - - - - - - - -##
    ##                     CONVENTIONAL                     ##
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - -##

    # # Conventional Controller on System A, 100 chosen days (selected by choosing the seed)
    myETAx = ETAx(root_path=root_path,
                  config_name='hybridenergysystem_pid',
                  relpath_config='config/',
                  config_overwrite={'settings':{'episode_duration': 86400, 'sampling_time': 30, 'n_episodes_play': 10, 'n_environments': 1}})
    myETAx.play('hybridenenergysystem_pid', 'pid_basiert', '1 day PID gesteuert')
##sd