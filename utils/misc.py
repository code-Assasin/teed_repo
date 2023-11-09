import os


def run_name_generator(params):
    last_word = "trial"
    if params["run_name"] is None:
        run_name = last_word
    else:
        run_name = params["run_name"]
    return run_name


def save_path_generator(params, save_dir):
    checkpoint_path = os.path.join(save_dir, params["run_name"])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    return checkpoint_path
