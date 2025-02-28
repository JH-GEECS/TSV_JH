import os
from pprint import pprint

import hydra

import wandb
from omegaconf import DictConfig, OmegaConf

from src.eval.aggregation import create_task_vector
from src.eval.eval_utils import perform_eval_with_merged_vector
from src.utils.variables_and_paths import ALL_DATASETS


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:

    if cfg.DATASETS == "":
        cfg.DATASETS = ALL_DATASETS[: cfg.num_tasks]
    else:
        cfg.num_tasks = len(cfg.DATASETS)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in cfg.DATASETS]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    OmegaConf.set_struct(cfg, True)

    # set up experiment for WandB
    print(cfg.method.full_name)
    print()
    wandb.init(
        config=OmegaConf.to_container(cfg),
        mode=cfg.wandb.mode,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        dir="logs/",
    )
    wandb.config.update({"method.full_name1": cfg.method.full_name})
    wandb.config.update({"method.keep": cfg.method.k})
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, True)

    # create final task vector
    task_vector_dict, eval_masks, svd_dict = create_task_vector(cfg)
    print("*" * 100)
    print("*" * 37, "Created task vector dict", "*" * 37)
    print("*" * 100)
    print("\n" * 3)

    # perform evaluation and log results
    print("*" * 100)
    print("*" * 39, "Starting Evaluation.", "*" * 39)
    print("*" * 100)
    # 여기서 logger function만 갈아끼우기
    additive_accuracies = perform_eval_with_merged_vector(
        cfg, task_vector_dict, eval_masks, svd_dict
    )
    pprint(additive_accuracies, width=1)
    wandb.log(additive_accuracies)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    # 이 부분을 hydra 구동 시에 병렬화로 변경해주기 ray로 바꿔서
    my_app()
