from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 10,
        "lr": 1e-4,
        "seq_len": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "transformer_weights",
        "preload": None,
        "model_base": "tmodel",
        "tokenizer_file": "tokenizer_{0}.json",
        "expirement_name": "runs/tmodel",
        "vocab_size": 10000,
        "d_model": 512
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_base = config['model_base']
    model_filename = f"{model_base}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

