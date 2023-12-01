from dataclasses import dataclass
import torch
import random
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_scheduler,
)
from tqdm import tqdm
from evaluation_detox.metric_tools.content_similarity import calc_bleu
from metric import joint_metrics
from util import DEBUG, find_project_root


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class FinetuneConfig:
    model_name: str
    num_epochs: int
    lr: float
    scheduler: str
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finetune(
    cfg: FinetuneConfig,
    train_loader: DataLoader,
    eval_loader: DataLoader,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    num_training_steps = cfg.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name=cfg.scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    test_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    detox_source = [
        "You are wrong as shit!",
        "I don't even want to give it a fuck",
        "Why do they keep fucking things up",
        "Nobody likes your idiot idea!",
        "Have a nice day!",
    ]

    text_inputs = test_tokenizer(
        detox_source, truncation=True, padding=True, return_tensors="pt"
    ).to(device=cfg.device)

    validation_labels = []
    validation_preds = []

    epochs = cfg.num_epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, unit="batch", disable=not DEBUG) as tepoch:
            for i, batch in enumerate(tepoch):
                # print(batch)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

                tepoch.set_postfix(
                    loss=f"{loss.item():.4f}", avg_loss=f"{total_loss/(i+1):.4f}"
                )

        model.eval()

        # Try to detoxify some examples
        detox_ids = model.generate(
            text_inputs["input_ids"], num_beams=4, min_length=0, max_length=100
        )
        detox_result = test_tokenizer.batch_decode(
            detox_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for toxic, detox in zip(detox_source, detox_result):
            tqdm.write(f"Toxic: {toxic}")
            tqdm.write(f"Detox: {detox}")
            tqdm.write("")

        all_label = []
        all_pred = []
        for batch in tqdm(eval_loader):
            input_ids = batch["input_ids"]
            label_ids = batch["labels"]
            output_ids = model.generate(
                input_ids, num_beams=4, min_length=0, max_length=100
            )
            labels = test_tokenizer.batch_decode(
                label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            outputs = test_tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            all_label.extend(labels)
            all_pred.extend(outputs)
        validation_labels.append(all_label)
        validation_preds.append(all_pred)

        bleu = calc_bleu(all_label, all_pred)
        tqdm.write(f"Epoch {epoch+1}/{epochs}: BLEU: {bleu:.4f}")
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    for i, (labels, preds) in enumerate(zip(validation_labels, validation_preds)):
        joint_metrics(preds, labels, model_name=f"epoch {i}")

    checkpoint_path = find_project_root() / "outputs" / "checkpoints"
    model.save_pretrained(checkpoint_path)

    return model


@dataclass
class StyleTransferConfig:
    model_name: str
    num_epochs: int
    lr: float
    scheduler: str
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def style_transfer(
    cfg: StyleTransferConfig,
    train_loader: DataLoader,
    eval_loader: DataLoader,
):
    pass


if __name__ == "__main__":
    from dataloader import DataLoaderConfig, paradetox_dataloader

    cfg = FinetuneConfig(
        model_name="facebook/bart-large", num_epochs=3, lr=0.0001, scheduler="linear"
    )
    dl_cfg = DataLoaderConfig(batch_size=16, tokenizer_source="facebook/bart-large")
    loaders = paradetox_dataloader(dl_cfg)
    finetune(cfg, loaders.train, loaders.validation)
