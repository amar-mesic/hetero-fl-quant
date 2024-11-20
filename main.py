import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    print(OmegaConf.to_yaml(cfg))
    
    print("Hello World!")


if __name__ == "__main__":
    main()
    