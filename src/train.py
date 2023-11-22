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
import evaluate
from tqdm import tqdm
from util import DEBUG


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

    metric = evaluate.load("accuracy")

    detox_source = [
        "You are wrong as shit!",
        "I don't even want to give it a fuck",
        "Why do they keep fucking things up",
        "Nobody likes your idiot idea!",
        "Don't pretend as if you've got a brain",
    ]

    text_inputs = test_tokenizer(
        detox_source, truncation=True, padding=True, return_tensors="pt"
    ).to(device=cfg.device)

    print(text_inputs)

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
            detox_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        tqdm.write(f"Toxic: {detox_source}")
        tqdm.write(f"Detox: {detox_result}")
        for toxic, detox in zip(detox_source, detox_result):
            tqdm.write(f"Toxic: {toxic}")
            tqdm.write(f"Detox: {detox}")
            tqdm.write("")
    # for batch in eval_loader:
    #    with torch.no_grad():
    #        outputs = model(**batch)
    #    logits = outputs.logits
    #    predictions = torch.argmax(logits, dim=-1)
    #    metric.add_batch(predictions=predictions, references=batch["labels"])
    # accuracy = metric.compute()
    # tqdm.write(f"Epoch {epoch+1}/{epochs}: Accuracy: {accuracy:.4f}")
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return model


if __name__ == "__main__":
    from dataloader import DataLoaderConfig, paradetox_dataloader

    cfg = FinetuneConfig(
        model_name="facebook/bart-large", num_epochs=4, lr=0.0001, scheduler="linear"
    )
    dl_cfg = DataLoaderConfig(batch_size=16, tokenizer_source="facebook/bart-large")
    loaders = paradetox_dataloader(dl_cfg)
    finetune(cfg, loaders.train, loaders.validation)
