import os
import json
import math
import random
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_CONTRIBUTORS = 30
MAX_EXPLS = 3
MAX_LEN = 128
SEED = 42

SAVED_MODEL_DIR = "./saved_hybrid_shap_model"
TRAIN_JSONL = "./data/split_shap_train_full.jsonl"
VAL_JSONL = "./data/split_shap_val_full.jsonl"
TEST_JSONL = "./data/split_shap_test_full.jsonl"
ROC_SAVE_PATH = "roc_curve_hybrid_shap_rl.png"
RL_SAVE_DIR = "./saved_hybrid_shap_model_rl"

EPOCHS = 6
LR = 1e-4
BATCH_SIZE = 16
PPO_CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
EPSILON = 1e-8

MIN_EXPLANATION_LENGTH = 8
MAX_EXPLANATION_LENGTH = 60

CLASSIFICATION_REWARD_WEIGHT = 0.7
CALIBRATION_REWARD_WEIGHT = 0.2
EXPLANATION_REWARD_WEIGHT = 0.1
EXPLANATION_TERMINAL_PUNCTUATION = (".", "!", "?")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_shap_features(contributors: list) -> np.ndarray:
    vec = np.zeros(TOP_K_CONTRIBUTORS, dtype=np.float32)
    for i, c in enumerate(contributors[:TOP_K_CONTRIBUTORS]):
        vec[i] = float(c.get("shap_value", 0.0))
    return vec


def build_llm_text(row: dict) -> str:
    top_features = ", ".join(c.get("feature", "NA") for c in row.get("contributors", [])[:3])
    explanations = " ".join(row.get("summary_explanations", []))
    return f"Top drivers: {top_features}. {explanations}"


def create_example(row: dict) -> dict:
    label = 0 if int(row["ground"]) == 0 else 1
    text = build_llm_text(row)
    features = extract_shap_features(row.get("contributors", []))
    expls = row.get("summary_explanations", [])[:MAX_EXPLS]
    expls = expls + ["No explanation available."] * (MAX_EXPLS - len(expls))
    top_feats = [c.get("feature", "NA") for c in row.get("contributors", [])[:3]]
    return {
        "text": text,
        "label": label,
        "features": features.tolist(),
        "explanations": expls,
        "top_features": top_feats,
    }


def load_jsonl(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        rows = [create_example(json.loads(line)) for line in f if line.strip()]
    return Dataset.from_list(rows)


class HybridModel(nn.Module):
    def __init__(self, base_model, feature_dim: int):
        super().__init__()
        self.model = base_model
        hidden_size = base_model.config.hidden_size
        self.tabular = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size + 64),
            nn.Linear(hidden_size + 64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    @staticmethod
    def mean_pool(hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask, features):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        llm_feat = self.mean_pool(outputs.hidden_states[-1], attention_mask)
        tab_feat = self.tabular(features)
        combined = torch.cat([llm_feat, tab_feat], dim=1)
        logits = self.classifier(combined)
        return {"logits": logits}


def load_trained_hybrid(saved_dir: str, device: str):
    with open(os.path.join(saved_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(os.path.join(saved_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(saved_dir, "tokenizer"))
    tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModel.from_pretrained(
        meta["model_name"],
        quantization_config=bnb_cfg,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    base = PeftModel.from_pretrained(base, os.path.join(saved_dir, "lora_adapter"))

    model = HybridModel(base, meta["feature_dim"])
    head_state = torch.load(
        os.path.join(saved_dir, "classifier_head.pt"),
        map_location=device,
    )
    model.tabular.load_state_dict(head_state["tabular"])
    model.classifier.load_state_dict(head_state["classifier"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer, scaler, meta


def normalize_dataset(ds: Dataset, scaler) -> Dataset:
    def _map_fn(ex):
        x = scaler.transform([ex["features"]])[0]
        return {"features": x.astype(np.float32).tolist()}

    return ds.map(_map_fn)


def tokenize_dataset(ds: Dataset, tokenizer) -> Dataset:
    def _tok(ex):
        t = tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        t["labels"] = ex["label"]
        t["features"] = ex["features"]
        t["explanations"] = ex["explanations"]
        t["top_features"] = ex["top_features"]
        return t

    return ds.map(_tok)


def collate_fn(batch):
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
        "features": torch.tensor([x["features"] for x in batch], dtype=torch.float32),
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        "explanations": [x["explanations"] for x in batch],
        "top_features": [x["top_features"] for x in batch],
    }


def entropy_binary(p1: float) -> float:
    p1 = max(EPSILON, min(1 - EPSILON, p1))
    p0 = 1.0 - p1
    return -(p0 * math.log(p0) + p1 * math.log(p1))


def build_state_features(base_probs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    p_anom = base_probs[:, 1]
    p_norm = base_probs[:, 0]
    margin = (p_anom - p_norm).unsqueeze(1)
    ent = torch.tensor([entropy_binary(float(x)) for x in p_anom], device=base_probs.device).unsqueeze(1)
    abs_mean = features.abs().mean(dim=1, keepdim=True)
    abs_max = features.abs().max(dim=1, keepdim=True).values
    state = torch.cat([base_probs, margin, ent, abs_mean, abs_max, features], dim=1)
    return state


def explanation_reward(expl: str, top_feats: list) -> float:
    text = (expl or "").lower()
    tokens = text.split()
    if not tokens:
        return -0.5
    coverage = 0.0
    for f in top_feats:
        if str(f).lower() in text:
            coverage += 1.0
    coverage /= max(1, len(top_feats))
    length = len(tokens)
    length_reward = 1.0 if MIN_EXPLANATION_LENGTH <= length <= MAX_EXPLANATION_LENGTH else 0.2
    naturalness = 1.0 if any(text.endswith(p) for p in EXPLANATION_TERMINAL_PUNCTUATION) else 0.4
    return 0.6 * coverage + 0.3 * length_reward + 0.1 * naturalness


def select_actions(action_logits: torch.Tensor, greedy: bool) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Categorical(logits=action_logits)
    actions = torch.argmax(action_logits, dim=1) if greedy else dist.sample()
    logp = dist.log_prob(actions)
    return actions, logp


class PPOPolicy(nn.Module):
    def __init__(self, input_dim: int, n_actions: int = 2 * MAX_EXPLS):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.actor(h), self.critic(h).squeeze(-1)


@dataclass
class BatchOutput:
    states: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def collect_batch(policy, hybrid_model, loader, device):
    all_states, all_actions, all_old_logp = [], [], []
    all_returns, all_adv = [], []

    for batch in loader:
        labels = batch["labels"].to(device)
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "features": batch["features"].to(device),
        }
        with torch.no_grad():
            logits = hybrid_model(**inputs)["logits"].float()
            base_probs = torch.softmax(logits, dim=1)
            states = build_state_features(base_probs, inputs["features"])
            action_logits, values = policy(states)
            actions, logp = select_actions(action_logits, greedy=False)

        pred_cls = (actions // MAX_EXPLS).long()
        expl_idx = (actions % MAX_EXPLS).long()

        rewards = []
        for i in range(labels.size(0)):
            y = int(labels[i].item())
            pred = int(pred_cls[i].item())
            cls_reward = 2.0 if pred == y else -2.0
            if y == 1 and pred == y:
                cls_reward += 0.5
            p_true = float(base_probs[i, y].item())
            calib_reward = math.log(p_true + EPSILON)
            chosen_expl = batch["explanations"][i][int(expl_idx[i].item())]
            er = explanation_reward(chosen_expl, batch["top_features"][i])
            total_r = (
                CLASSIFICATION_REWARD_WEIGHT * cls_reward
                + CALIBRATION_REWARD_WEIGHT * calib_reward
                + EXPLANATION_REWARD_WEIGHT * er
            )
            rewards.append(total_r)

        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = rewards - values.detach()

        all_states.append(states.detach())
        all_actions.append(actions.detach())
        all_old_logp.append(logp.detach())
        all_returns.append(rewards.detach())
        all_adv.append(adv.detach())

    states = torch.cat(all_states, dim=0)
    actions = torch.cat(all_actions, dim=0)
    old_logp = torch.cat(all_old_logp, dim=0)
    returns = torch.cat(all_returns, dim=0)
    advantages = torch.cat(all_adv, dim=0)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return BatchOutput(states, actions, old_logp, returns, advantages)


def ppo_update(policy, optimizer, memory: BatchOutput):
    logits, values = policy(memory.states)
    dist = torch.distributions.Categorical(logits=logits)
    new_logp = dist.log_prob(memory.actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_logp - memory.old_logp)
    s1 = ratio * memory.advantages
    s2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * memory.advantages
    policy_loss = -torch.min(s1, s2).mean()
    value_loss = nn.MSELoss()(values, memory.returns)
    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


@torch.no_grad()
def evaluate(policy, hybrid_model, ds, batch_size=16):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    y_true, y_pred, y_prob, picked_expls = [], [], [], []

    policy.eval()
    hybrid_model.eval()
    for batch in loader:
        labels = batch["labels"].to(DEVICE)
        inputs = {
            "input_ids": batch["input_ids"].to(DEVICE),
            "attention_mask": batch["attention_mask"].to(DEVICE),
            "features": batch["features"].to(DEVICE),
        }
        logits = hybrid_model(**inputs)["logits"].float()
        base_probs = torch.softmax(logits, dim=1)
        states = build_state_features(base_probs, inputs["features"])
        action_logits, _ = policy(states)
        act_probs = torch.softmax(action_logits, dim=1)
        actions, _ = select_actions(action_logits, greedy=True)
        pred_cls = (actions // MAX_EXPLS).long()
        expl_idx = (actions % MAX_EXPLS).long()

        p_anom = act_probs[:, MAX_EXPLS:].sum(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred_cls.cpu().tolist())
        y_prob.extend(p_anom.cpu().tolist())
        for i in range(labels.size(0)):
            picked_expls.append(batch["explanations"][i][int(expl_idx[i].item())])

    auc = roc_auc_score(y_true, y_prob)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
    }
    return metrics, y_true, y_pred, y_prob, picked_expls


def save_roc(y_true, y_prob, auc_score, path):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index_threshold_idx = np.argmax(tpr - fpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, color="steelblue", label=f"ROC (AUC={auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.scatter(
        fpr[youden_index_threshold_idx],
        tpr[youden_index_threshold_idx],
        color="crimson",
        label=f"Youden threshold={thresholds[youden_index_threshold_idx]:.3f}",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — RL Refined Hybrid SHAP Classifier")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    hybrid_model, tokenizer, scaler, meta = load_trained_hybrid(SAVED_MODEL_DIR, DEVICE)
    print(f"Loaded trained hybrid model from: {SAVED_MODEL_DIR}")

    train_ds = load_jsonl(TRAIN_JSONL)
    val_ds = load_jsonl(VAL_JSONL)
    test_ds = load_jsonl(TEST_JSONL)
    print(f"Loaded: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_ds = normalize_dataset(train_ds, scaler)
    val_ds = normalize_dataset(val_ds, scaler)
    test_ds = normalize_dataset(test_ds, scaler)

    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)
    test_ds = tokenize_dataset(test_ds, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    state_dim = 2 + 1 + 1 + 1 + 1 + int(meta["feature_dim"])
    policy = PPOPolicy(state_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.01)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        policy.train()
        memory = collect_batch(policy, hybrid_model, train_loader, DEVICE)
        total_loss, pol_loss, val_loss = ppo_update(policy, optimizer, memory)

        val_metrics, _, _, _, _ = evaluate(policy, hybrid_model, val_ds, batch_size=BATCH_SIZE)
        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"loss={total_loss:.4f} policy={pol_loss:.4f} value={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)

    test_metrics, y_true, y_pred, y_prob, picked_expls = evaluate(policy, hybrid_model, test_ds, batch_size=BATCH_SIZE)

    print("\n===== SAMPLE TEST PREDICTIONS WITH EXPLANATIONS =====")
    for i in range(min(6, len(y_true))):
        print(
            f"[{i}] GT={y_true[i]} Pred={y_pred[i]} P(anom)={y_prob[i]:.4f}\n"
            f"    Explanation: {picked_expls[i]}"
        )

    print("\nTEST METRICS (RL-REFINED)")
    print(f"Accuracy ={test_metrics['accuracy']:.4f}")
    print(f"Precision={test_metrics['precision']:.4f}")
    print(f"Recall   ={test_metrics['recall']:.4f}")
    print(f"F1       ={test_metrics['f1']:.4f}")
    print(f"AUC-ROC  ={test_metrics['auc']:.4f}")

    save_roc(y_true, y_prob, test_metrics["auc"], ROC_SAVE_PATH)
    print(f"[SAVE] AUC-ROC curve saved -> {ROC_SAVE_PATH}")

    os.makedirs(RL_SAVE_DIR, exist_ok=True)
    policy_path = os.path.join(RL_SAVE_DIR, "ppo_policy.pt")
    torch.save(policy.state_dict(), policy_path)
    rl_meta = {
        "state_dim": state_dim,
        "n_actions": 2 * MAX_EXPLS,
        "epochs": EPOCHS,
        "lr": LR,
        "clip": PPO_CLIP,
        "seed": SEED,
        "base_model_dir": SAVED_MODEL_DIR,
        "roc_path": ROC_SAVE_PATH,
    }
    with open(os.path.join(RL_SAVE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(rl_meta, f, indent=2)
    print(f"[SAVE] RL policy saved -> {policy_path}")
    print(f"[SAVE] RL artifacts dir -> {RL_SAVE_DIR}")


if __name__ == "__main__":
    main()
