import os

import torch
import logging
import numpy as np
from transformers import CsmForConditionalGeneration, Trainer, TrainingArguments, AutoProcessor, BitsAndBytesConfig
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import prepare_model_for_kbit_training
from lora import transcribe_audio_files
from torch.utils.data import Dataset
from bitsandbytes.optim import PagedAdamW8bit

from torch.utils.tensorboard import SummaryWriter


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)
logger = logging.getLogger(__name__)

PARENT_DIR = "/home/ubuntu/TTS-data-preparator/metadata"
SHORT_META_FILES = [
        "Aiganysh-neutral.txt",
        "Aiganysh-strict.txt",
        "Timur-neutral.txt",
        "Timur-strict.txt"
    ]
META_FILES = [os.path.join(PARENT_DIR, meta) for meta in SHORT_META_FILES]
OUTPUT_DIR = "finetuned_model"
KEEP_LAST_N_CHECKPOINTS = 5
NUM_EPOCHS = 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
GRADIENT_CHECKPOINTING = True
LEARNING_RATE = 5e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "sesame/csm-1b"
MAX_AUDIO_FILES = 0

LORA_LR = 1e-4
MODULES_TO_SAVE_LR = 1e-5
R = 32
ALPHA = 64
LORA_DROPOUT = 0.05

TARGET_MODULES = [
    # Backbone model attention layers
    "k_proj",
    "q_proj",
    "v_proj",
    "o_proj",

    # Backbone model MLP layers
    "gate_proj",
    "up_proj",
    "down_proj"
]

MODULES_TO_SAVE = ["embed_text_tokens",
                   "embed_tokens",
                   "lm_head",
                   "inputs_embeds_projector",
                   "codebooks_head"]

class ConversationDataset(Dataset):
    def __init__(self, audio_text_pairs, processor):
        self.pairs = audio_text_pairs
        self.processor = processor
        self.sample_rate = processor.feature_extractor.sampling_rate

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        audio = item.load_audio(self.sample_rate)

        inputs = self.processor(
            text=f"<|begin_of_text|>[{item.speaker_id}]{item.text}<|end_of_text|><|AUDIO|><|audio_eos|>",
            audio=audio,
            output_labels=True,
            text_kwargs={"padding": True},
            audio_kwargs={"sampling_rate": self.sample_rate},
            common_kwargs={"return_tensors": "pt"},
        )
        cleaned = {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                   for k, v in inputs.items() if torch.is_tensor(v)}
        return cleaned


def prepare_csm_model_for_training():
    logger.info(f"Loading CSM model: {MODEL_NAME}")

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=MODULES_TO_SAVE,
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = CsmForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    logger.info("Applying LoRA to model using PEFT...")
    peft_config = LoraConfig(
        r=R,
        lora_alpha=ALPHA,
        target_modules=TARGET_MODULES,
        modules_to_save=MODULES_TO_SAVE,
        lora_dropout=LORA_DROPOUT,
        bias="all",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.backends.cuda.enable_flash_sdp(True)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    model, processor = prepare_csm_model_for_training()
    audio_text_pairs = transcribe_audio_files(metafile_paths=META_FILES)
    if not audio_text_pairs:
        logger.error(f"No audio files found or transcribed in {META_FILES}")
        return

    dataset = ConversationDataset(
        audio_text_pairs,
        processor=processor,
    )

    writer = SummaryWriter('runs/sesame_explorer')

    logger.info(f"Dataset created with {len(dataset)} samples")
    wandb.init(
        project="sesame-explorer",
        config={
            "model_name": MODEL_NAME,
            "lora_rank": R,
            "lora_alpha": ALPHA,
        },
        reinit=True,
    )

    writer.add_graph(model, dataset[0])
    writer.close()



if __name__ == "__main__":
    main()