from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    ##################
    #                #
    #    TRAINING    #
    #                #
    ##################

    config_experiment_hess_learn = {
        "settings": {"n_environments": 1, "n_episodes_learn": 100, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            # "policy_kwargs": {"net_arch": [dict(pi=[500, 400, 300], vf=[500, 400, 300])]},
            "device": "cpu",  # "cuda" on systems with cuda installed
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="hybridenergysystem_ppo",
        config_overwrite=config_experiment_hess_learn,
        relpath_config="config/",
    )
    #print("VecNormalize path:", experiment_1.config_run.path_vec_normalize)
    # >>> EINMAL VOR DEM TRAINING (z.B. direkt im experiments_DRL.py) <<<


    _orig_load = VecNormalize.load

    def _safe_load(path, venv):
        try:
            return _orig_load(path, venv)
        except AssertionError as e:
            if "spaces must have the same shape" in str(e):
                print(f"[VecNormalize] Shape mismatch für {path}. Starte mit frischer Normalisierung.")
                # gleiche Defaults wie üblich; bei Bedarf anpassen
                return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
            raise

    VecNormalize.load = _safe_load

    experiment_1.learn("ppo_hess_wpredictions_2", "experiment_hess_wpredictions_90episodes", reset=True)


    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    config_experiment_hess_play = {
        "settings": {"n_environments": 1, "n_episodes_play": 10, "episode_duration": 86400 * 3, "plot_interval": 1},
        # "agent_specific": {"policy_kwargs": {"net_arch": [dict(pi=[500, 400, 300], vf=[500, 400, 300])]}},
        "environment_specific": {
        "scenario_time_begin": "2017-01-01 00:00",
        "scenario_time_end": "2017-12-27 00:00",
            "random_sampling": True,
            "scenario_files": [
                {
                    "path": "Factory_2017.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {
                        "power_electricity": 12.5,
                        "time_availability": 1,
                    },
                },
                {
                    "path": "EnergyMarkets_2017.csv",
                    "interpolation_method": "ffill",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {"electrical_energy_price": 0.001},
                },
            ],
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="hybridenergysystem_ppo",
        config_overwrite=config_experiment_hess_play,
        relpath_config="config/",
    )

    #experiment_1.play("ppo_test2", "experiment_1")


if __name__ == "__main__":
    main()

