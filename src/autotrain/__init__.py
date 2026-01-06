import os


os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import warnings


try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

from autotrain.logging import Logger


warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")
warnings.filterwarnings("ignore", category=UserWarning, module="datasets")
warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

logger = Logger().get_logger()
__version__ = "0.8.37.dev0"


def is_colab():
    try:
        import google.colab

        return True
    except ImportError:
        return False


def is_unsloth_available():
    try:
        from unsloth import FastLanguageModel

        return True
    except Exception as e:
        logger.warning("Unsloth not available, continuing without it")
        logger.warning(e)
        return False
