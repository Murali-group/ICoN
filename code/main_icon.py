import sys
import json
from typing import Union
from utils.common import *
from utils.param_process import *
from train_icon_noise import Trainer

def main(config_path: Union[Path, dict]):
    for run_no in range(5):
        trainer = Trainer(config_path)
        print(trainer.params.gat_shapes)
        if trainer.params.mode=='train':
            trainer.train()
        trainer.forward(run_no) #save the model and features in run specific files.
        print('Finished with run: ', run_no)
        del trainer

if __name__ == "__main__":
    if len(sys.argv) == 1: #if no argument passed, use a default config file.
        config_path = Path('code/config/icon_no-noise_yeast.json')
    else:
        config_path = Path(sys.argv[1])

    print("config_path: ", config_path)
    with config_path.open() as f:
        init_config_dict = json.load(f)
    config_dicts = wrapper_generate_combination_dict(init_config_dict)
    for config_dict in config_dicts:
        main(config_dict)


