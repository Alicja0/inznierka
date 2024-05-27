import datetime
import os
import math
import yaml


def get_unique_exp_name(exp_name: str):
    return f'{exp_name}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'


def setup_experiment_output(experiment_config: dict):
    experiment_name: str = get_unique_exp_name(experiment_config["experiment_name"])
    full_experiment_path: str = os.path.join(experiment_config["output_directory"], experiment_name)
    os.mkdir(full_experiment_path)
    with open(os.path.join(full_experiment_path, "config.yaml"), 'w') as stream:
        yaml.dump(experiment_config, stream)
    return full_experiment_path


def calculate_metrics(conf_matrix):
    # Extract true positives, false positives, true negatives, and false negatives
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Calculate recall (sensitivity or true positive rate)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate balanced accuracy
    balanced_accuracy = (recall + specificity) / 2

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'balanced_accuracy': balanced_accuracy
    }
