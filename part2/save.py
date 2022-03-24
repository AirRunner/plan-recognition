from os import path
import json


def save_results(result, model_name):
    results_file = "results.json"
    if path.isfile(results_file):
        with open(results_file, 'r') as res_json:
            result_dict = json.load(res_json)
    else:
        result_dict = dict()

    result_dict[model_name] = {
        "0-25": result[0],
        "0-50": result[1],
        "0-75": result[2],
        "0-100": result[3]
    }

    with open(results_file, 'w') as res_json:
        json.dump(result_dict, res_json, indent=4)
