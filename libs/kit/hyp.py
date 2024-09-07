#!/usr/bin/env python

""" Hyperparameter optimization framework based on optuna """


import importlib
import subprocess
import argparse
from datetime import datetime
import os
import optuna
import random

import numpy as np

from kit.data import DD


args, args_unknown = None, None


def ls_studies(storage):
    """lists all studies in the database"""

    study_summaries = optuna.study.get_all_study_summaries(storage=storage)
    text = f"{'Study Name':<50s} {'N-Trials':>10s} {'Best':>5s}\n"
    for study_summary in study_summaries:
        if study_summary.best_trial is not None:
            wert = f"{study_summary.best_trial.value:6.2f}"
            best_job = f"{study_summary.best_trial.user_attrs['JOB.ID']:>15s}"
        else:
            wert = f"{'---':6}"
            best_job = f"{'':>15s}"
        text += f"{study_summary.study_name:<50s} {study_summary.n_trials:>10} {wert} {best_job}\n"
    print(text)


def get_study(storage, study_name, direction="min"):
    """returns a study from the database"""

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=f"{direction}imize",
    )
    setattr(study, "storage", storage)
    return study


def rm_study(study):
    """removes a study from the database"""

    optuna.delete_study(study_name=study.study_name, storage=study.storage)


def ls_trials(study, hparams=None, ascending=None):
    """lists all trials in a study"""

    trials = [(trial.value, trial) for trial in study.trials]
    if ascending is not None:
        trials = sorted(
            trials,
            key=lambda x: x[0]
            if x[0] is not None
            else (float("inf") if ascending else float("-inf")),
        )
    for _, trial in trials:
        print_trial(trial, hparams)


def print_trial(trial, hparams=None, show_date=False, show_intermediate=False):
    """prints a trial in a nice format

    :param trial: the trial to be printed
    :param hparams: the hyperparameters to be printed (if None, all are printed)
    :param show_date: whether to show the start date of the trial
    :param show_intermediate: whether to show the number of intermediate values
    :return: None
    """

    params = ""

    datetime_start = (
        trial.user_attrs["datetime_start"]
        if "datetime_start" in trial.user_attrs
        else trial.datetime_start.timestamp()
    )
    datetime_complete = (
        trial.user_attrs["datetime_complete"]
        if "datetime_complete" in trial.user_attrs
        else (
            trial.datetime_complete.timestamp()
            if trial.datetime_complete is not None
            else None
        )
    )
    if datetime_complete is not None:
        duration = (
            datetime.fromtimestamp(datetime_complete)
            - datetime.fromtimestamp(datetime_start)
        ).total_seconds()
    else:
        duration = None

    for key, value in trial.params.items():
        if not hparams or (key in hparams):
            params += (
                f"{key}: {f'{value:4}' if isinstance(value, int) else f'{value:8.2e}'} "
            )

    text = f"V: {trial.value:.3f} " if trial.value is not None else "V: ----- "
    if show_intermediate:
        text += f"({len(trial.intermediate_values):2}) "
    text += f"{str(trial.state).split('.')[1]:8} "
    if show_date:
        text += f"{datetime.fromtimestamp(datetime_start).strftime('%Y-%m-%d %H:%M')}, "
    text += f"{duration/3600:4.1f} h, " if duration is not None else "---- h, "
    # text += f"{trial.number:3,} "
    text += f"{trial.number:<4} "
    text += (
        f"{trial.user_attrs['JOB.ID']:<12s} "
        if "JOB.ID" in trial.user_attrs
        else f"{'':<12s} "
    )
    text += f"\n\t{params}"
    print(text)


def cp_trial(trial, params="NA", value="NA"):
    """copies a trial"""

    params = trial.params if params == "NA" else params
    # pylint: disable=protected-access
    value, state = (
        (trial.value, trial.state)
        if value == "NA"
        else (value, optuna.trial._state.TrialState.COMPLETE)
    )

    datetime_start = (
        trial.user_attrs["datetime_start"]
        if "datetime_start" in trial.user_attrs
        else trial.datetime_start
    )
    datetime_complete = (
        trial.user_attrs["datetime_complete"]
        if "datetime_complete" in trial.user_attrs
        else trial.datetime_complete
    )
    user_attrs = trial.user_attrs
    user_attrs["datetime_start"] = datetime_start.timestamp()
    if datetime_complete is not None:
        user_attrs["datetime_complete"] = datetime_complete.timestamp()
    else:
        user_attrs["datetime_complete"] = None

    return optuna.trial.create_trial(
        params=params,
        distributions=trial.distributions,
        value=value,
        intermediate_values=trial.intermediate_values,
        user_attrs=user_attrs,
        state=state,
    )


def cp_study(study, study_name, adjust_trials=()):
    """copies a study"""

    old_trials = study.get_trials()
    keep_trials = []

    for trial in old_trials:
        if trial.number in adjust_trials:
            if adjust_trials[trial.number] != "del":
                keep_trials.append(cp_trial(trial, value=adjust_trials[trial.number]))
        else:
            keep_trials.append(cp_trial(trial))

    new_study = get_study(study.storage, study_name)
    rm_study(new_study)
    new_study = get_study(study.storage, study_name)
    new_study.add_trials(keep_trials)
    return new_study


def new_trial_optuna(x):
    """runs a trial

    The call argument specifies a python function to call with the cli arguments and
    the trial; this function returns the parameters to subprocess.run as result.
    The last line of the output of the subprocess is interpreted as the result of
    the trial. If the last line is "ERROR", the trial is considered
    failed and None is returned. Otherwise, the last line is interpreted as a float
    and returned.
    """

    global args, args_unknown

    x.set_user_attr("JOB.ID", args.job)

    tmp = args.call.split(".")
    module, function = tmp[:-1], tmp[-1]
    module = importlib.import_module(".".join(module))
    run_params = getattr(module, function)(args, x)
    print(f"Run: {run_params}")
    result = subprocess.run(run_params, capture_output=True, check=False)
    print("Finished")
    result = result.stdout.decode("UTF-8").split("\n")[-1]
    if result != "ERROR":
        result = float(result)

    print(f"Result: {result}")
    return result


def get_hyp_params(yaml_file_path):
    """ returns a DD with the command and the 
    populated hyperparameters from the YAML file

    :param yaml_file_path: path to the YAML file specifying the hyperparameters
        and how to sample them
    :return: DD with the command and the 
        populated hyperparameters
    """

    if not os.path.exists(yaml_file_path):
        raise Exception(f"YAML file {yaml_file_path} does not exist")
    
    hyp_params = DD.from_yaml(yaml_file_path, evaluate=False)

    d_cmd = DD.from_dict({'COMMAND': hyp_params.COMMAND})
    for hp in hyp_params.HYPERPARAMETERS:
        kind, hp_type = hyp_params.HYPERPARAMETERS[hp].KIND.split(":")
        value = None
        if kind == "fixed":
            value = hyp_params.HYPERPARAMETERS[hp].VALUE
        elif kind == "uniform":
            value = random.uniform(
                float(hyp_params.HYPERPARAMETERS[hp].MIN), 
                float(hyp_params.HYPERPARAMETERS[hp].MAX)
            )
        elif kind == "log_uniform":
            _min = np.log(float(hyp_params.HYPERPARAMETERS[hp].MIN))
            _max = np.log(float(hyp_params.HYPERPARAMETERS[hp].MAX))
            value = np.exp(random.uniform(_min, _max))
        elif kind == "env":
            value = os.environ[hyp_params.HYPERPARAMETERS[hp].NAME]
            if "EVAL" in hyp_params.HYPERPARAMETERS[hp]:
                func = eval(hyp_params.HYPERPARAMETERS[hp].EVAL)
                value = func(value)

        if value is None:
            raise Exception(f"Hyperparameter {hp} not well defined in YAML file {yaml_file_path}")

        value = eval(f"{hp_type}(value)")

        d_cmd[hp] = value

    return d_cmd


def get_cmd_with_hyp_params(yaml_file_path):
    """ returns a list with the command and the 
    populated hyperparameters from the YAML file

    :param yaml_file_path: path to the YAML file specifying the hyperparameters
        and how to sample them
    :return: list with the command and the 
        populated hyperparameters
    """

    d_cmd = get_hyp_params(yaml_file_path)
    l_cmd = [d_cmd['COMMAND']]
    for hp_key, hp_value in d_cmd.items():
        if hp_key != 'COMMAND':
            l_cmd += [f"--{hp_key}", str(hp_value)]

    return l_cmd


def new_trial_random(yaml_file_path):
    if not os.path.exists(yaml_file_path):
        raise Exception(f"YAML file {yaml_file_path} does not exist")
    
    cmd = get_cmd_with_hyp_params(yaml_file_path)

    for arg in args_unknown:
        cmd += [arg]

    subprocess.run(cmd, capture_output=False, check=False)


def main():
    """main function"""

    if not args.storage.endswith('.yaml'):
        study = get_study(args.storage, args.study, args.direction)
        study.optimize(new_trial_optuna, n_trials=1)
    else:
        new_trial_random(args.storage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--call",
        type=str,
        default="",
        help=(
            "python function to call with the cli arguments and the trial; "
            "returns the parameters to subprocess.run as result"
        ),
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=r"sqlite:///hyp.db",
        help="the path of the hyperparameter database",
    )
    parser.add_argument(
        "--study", type=str, default="hyp", help="the name of the study"
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="min",
        help="the direction to optimize the value to (min/max)",
    )
    parser.add_argument("--job", type=str, default="", help="the job number")
    parser.add_argument("--env", type=str, default="py")
    args, args_unknown = parser.parse_known_args()

    main()
