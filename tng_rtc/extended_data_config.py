from gr00t.experiment.data_config import DATA_CONFIG_MAP
from .custom_data_config import So101DataConfig, LargeHorizonFourierGr1ArmsOnlyDataConfig

# Register custom data config
DATA_CONFIG_MAP.update({
    "so101_custom_config": So101DataConfig(),
    "large_horizon_fourier_gr1_arms_only": LargeHorizonFourierGr1ArmsOnlyDataConfig(),
})
