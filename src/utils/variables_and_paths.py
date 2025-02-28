from pathlib import Path
from typing import Literal, Optional
from omegaconf import DictConfig 
import os


TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
MODELS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]
OPENCLIP_CACHEDIR = Path(Path.home(), "openclip-cachedir", "open_clip").as_posix()
CACHEDIR = None

ALL_DATASETS = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SVHN",
    "SUN397",
    "STL10",
    "OxfordIIITPet",
    "Flowers102",
    "CIFAR100",
    "PCAM",
    "FER2013",
    "CIFAR10",
    "Food101",
    "FashionMNIST",
    "RenderedSST2",
    "EMNIST",
    "KMNIST",
]

DATASETS_8 = ALL_DATASETS[:8]
DATASETS_14 = ALL_DATASETS[:14]
DATASETS_20 = ALL_DATASETS[:20]


def cleanup_dataset_name(dataset_name: str):
    return dataset_name.replace("Val", "") + "Val"


# TODO: 여기서 뒤집어 까면 될듯 하다.
# model이 의미하는 바가 MODELS인 듯하다.
def is_TA_mode(config: DictConfig, task_name: str)->bool:
    """
    1. config에서 TA_mode라는 옵션을 받도록 하고 default를 False가 되도록 하자
    2. 그리고 task name까지 받아서 8개에 해당하면 반환 하도록 하자.
    """
    _ta_mode = config.get("TA_mode", False)
    is_target_task = task_name in DATASETS_8  
    
    if _ta_mode and is_target_task:
        print("currently load weight from TA authors")
        return True
    else:
        return False

def get_dir_dict(config: DictConfig, is_ta_mode: bool)->dict:
    # import ipdb; ipdb.set_trace()
    if is_ta_mode:
        _save_dir = ["/"] + config.save_dir.rstrip('/').split("/")
        save_dir = os.path.join(*_save_dir[:-1], _save_dir[-1] + "_TA")
    
    else:
        # weight_root = config.weight_root
        save_dir = config.save_dir
    
    return {
        "save": save_dir,
        # "weight_root": weight_root
    }


# def get_zeroshot_path(root, dataset, model):
#     return Path(
#         root, model, cleanup_dataset_name(dataset), f"nonlinear_zeroshot.pt"
#     ).as_posix()
def get_zeroshot_path(root, dataset, model, config: Optional[DictConfig] = None):
    return Path(root, model, f"nonlinear_zeroshot.pt").as_posix()

# def get_finetuned_path(root, dataset, model):
#     return Path(
#         root, model, cleanup_dataset_name(dataset), f"nonlinear_finetuned.pt"
#     ).as_posix()
def get_finetuned_path(root, dataset, model, config: Optional[DictConfig] = None):
    
    _dataset = dataset[:-3] if dataset.endswith("Val") else dataset
    
    _TA_mode = is_TA_mode(config, _dataset) if config else False
    
    if _TA_mode:
        return os.path.join(root, model+ "_TA", _dataset, "finetuned.pt")
        
    else:
        return Path(root, model, cleanup_dataset_name(dataset), f"nonlinear_finetuned.pt").as_posix()


def get_single_task_accuracies_path(model):
    return Path(
        "results/single_task", model, f"nonlinear_ft_accuracies.json"
    ).as_posix()
