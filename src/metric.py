from typing import List
import gc
import os

from box import Box
import numpy as np
import torch
from evaluation_detox.metric_tools.style_transfer_accuracy import (
    classify_preds,
)
from evaluation_detox.metric_tools.joint_metrics import get_gm, get_j

from evaluation_detox.metric_tools.content_similarity import (
    calc_bleu,
    flair_sim,
    wieting_sim,
)
from evaluation_detox.metric_tools.fluency import (
    # calc_flair_ppl,
    calc_gpt_ppl,
    do_cola_eval,
)
from util import find_project_root

proj_root = find_project_root()
args = Box(
    {
        "batch_size": 32,
        "cola_classifier_path": str(proj_root / "data/model/cola"),
        "wieting_tokenizer_path": str(
            proj_root / "data/model/wieting/sim.sp.30k.model"
        ),
        "wieting_model_path": str(proj_root / "data/model/wieting/sim.pt"),
        "t1": 75.0,  # this is default value
        "t2": 70.0,  # this is default value
        "t3": 12.0,  # this is default value
    }
)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def transfer_accuracy(preds: List[str]):
    """
    Calls a pre-trained RoBERTa model for toxicity classification
    """
    accuracy_by_sent = classify_preds(args, preds)
    avg_accuracy = np.mean(accuracy_by_sent)
    return accuracy_by_sent, avg_accuracy


def fluency(preds: List[str]):
    # charwise_ppl = calc_flair_ppl(preds)
    tokenwise_ppl = calc_gpt_ppl(preds)
    cola_by_sent = do_cola_eval(args, preds)
    avg_cola = sum(cola_by_sent) / len(preds)
    # return charwise_ppl, tokenwise_ppl, cola_by_sent, avg_cola
    return tokenwise_ppl, cola_by_sent, avg_cola


def similarity(preds: List[str], inputs: List[str]):
    bleu = calc_bleu(inputs, preds)
    emb_sim = flair_sim(args, inputs, preds).mean()
    similarity_by_sent = wieting_sim(args, inputs, preds)
    avg_similarity = similarity_by_sent.mean()
    return bleu, emb_sim, similarity_by_sent, avg_similarity


def joint_metrics(
    preds: List[str], inputs: List[str], save: bool = True, model_name="TEST"
):
    accuracy_by_sent, avg_accuracy = transfer_accuracy(preds)
    cleanup()

    # charwise_ppl, tokenwise_ppl, cola_by_sent, avg_cola = fluency(preds)
    tokenwise_ppl, cola_by_sent, avg_cola = fluency(preds)
    cleanup()

    bleu, emb_sim, similarity_by_sent, avg_similarity = similarity(preds, inputs)
    cleanup()

    # gm = get_gm(args, avg_accuracy, emb_sim, charwise_ppl)
    joint = get_j(args, accuracy_by_sent, similarity_by_sent, cola_by_sent, preds)

    if save:
        result_path = str(proj_root / "outputs/results.md")
        if not os.path.exists(result_path):
            with open(result_path, "w") as f:
                f.writelines(
                    "| Model | ACC | EMB_SIM | SIM | TokenPPL | FL | J | BLEU |\n"
                )
                f.writelines(
                    "| ----- | --- | ------- | --- | -------- | -- | - | ---- |\n"
                )

        with open(result_path, "a") as res_file:
            # res_file.writelines(
            #    f"{name}|{avg_accuracy:.4f}|{emb_sim:.4f}|{avg_similarity:.4f}|{charwise_ppl:.4f}|"
            #    f"{tokenwise_ppl:.4f}|{avg_cola:.4f}|{gm:.4f}|{joint:.4f}|{bleu:.4f}|\n"
            # )
            res_file.writelines(
                f"|{model_name}|{avg_accuracy:.4f}|{emb_sim:.4f}|{avg_similarity:.4f}|"
                f"{tokenwise_ppl:.4f}|{avg_cola:.4f}|{joint:.4f}|{bleu:.4f}|\n"
            )


if __name__ == "__main__":
    preds = ["Your idea is bad", "Your idea is shit", "Your fucking idea is pure shit"]
    inputs = [
        "Your fucking idea is pure shit",
        "Your fucking idea is pure shit",
        "Your fucking idea is pure shit",
    ]
    joint_metrics(preds, inputs)
