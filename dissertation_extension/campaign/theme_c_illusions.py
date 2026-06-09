"""
Theme C: optical illusions -- is the human "adversarial mode" the same as the
machine's, and does training a machine to be confused the way humans are confused
make it more robust?

Sub-experiments (each config is one row):
  C1/C2 (per illusion x label-scheme x arch x seed): train a small net on a 2-AFC
        perceptual task with either PHYSICAL labels (ground truth) or HUMAN labels
        (the way a human perceives the illusion). Measure:
          - physical-task accuracy,
          - human-agreement on the illusion (ambiguous) subset = does the model
            spontaneously fall for the illusion like a human?
          - adversarial robustness (FGSM + PGD ASR) on the perceptual task.
        Does human-like labelling flatten the model (lower ASR)?
  C3 (per real dataset x aug-mode x seed): take a standard classifier and add an
        auxiliary "illusion confusion" task (multi-task). Does sharing the human
        illusion bias regularise the main task toward adversarial robustness?
  C4 (cross-illusion generalisation): train on illusion A, test physical accuracy
        and human-agreement on illusion B -- does illusion susceptibility transfer?
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import common as C

KINDS = C.ILLUSION_KINDS                # 6
SCHEMES = ["physical", "human"]
ARCHS = ["cnn", "mlp"]
SEEDS = [0, 1, 2]
REAL_DATASETS = ["fashion_mnist", "kmnist", "cifar10", "svhn"]
ILLU_SZ = 32
ILLU_EPS = 0.1
ILLU_META = {"channels": 1, "size": ILLU_SZ, "n_classes": 2}


def build_specs():
    specs = []
    # C1/C2
    for k in KINDS:
        for sch in SCHEMES:
            for arch in ARCHS:
                for s in SEEDS:
                    specs.append({"theme": "C_illusions", "id": f"C12_{k}_{sch}_{arch}_s{s}",
                                  "params": {"sub": "C12", "kind": k, "scheme": sch,
                                             "arch": arch, "seed": s}})
    # C3 multitask robustness
    for ds in REAL_DATASETS:
        for aug in ["baseline", "illusion_aux"]:
            for s in SEEDS:
                specs.append({"theme": "C_illusions", "id": f"C3_{ds}_{aug}_s{s}",
                              "params": {"sub": "C3", "dataset": ds, "aug": aug, "seed": s}})
    # C4 cross-illusion generalisation (single seed)
    for ka in KINDS:
        for kb in KINDS:
            if ka == kb:
                continue
            specs.append({"theme": "C_illusions", "id": f"C4_{ka}_to_{kb}",
                          "params": {"sub": "C4", "train_kind": ka, "test_kind": kb, "seed": 0}})
    return specs


def _train_illusion_model(kind, scheme, arch, seed, n_train=2500, epochs=6):
    Xtr, PHYtr, HUMtr, LABtr = C.make_illusion_dataset(kind, n=n_train, label_scheme=scheme,
                                                        sz=ILLU_SZ, seed=seed)
    model = C.build_model(arch, ILLU_META, width=16 if arch == "cnn" else 256, seed=seed)
    C.train_model(model, Xtr, LABtr, epochs=epochs, opt="adam", lr=0.002, ncls=2)
    return model


def _illusion_eval(model, kind, seed):
    """Evaluate on a fresh illusion test set: physical acc, human-agreement on the
    ambiguous subset (phy != hum), and adversarial ASR vs physical labels."""
    Xte, PHY, HUM, _ = C.make_illusion_dataset(kind, n=800, label_scheme="physical",
                                                sz=ILLU_SZ, seed=seed + 777)
    lg, _ = C.logits_and_acc(model, Xte, PHY)
    pred = lg.argmax(1).to(PHY.device)
    phys_acc = float((pred == PHY).float().mean())
    ambiguous = (PHY != HUM)
    if ambiguous.sum() > 0:
        human_agree = float((pred[ambiguous] == HUM[ambiguous]).float().mean())
    else:
        human_agree = float("nan")
    fr = C.attack_success(model, Xte, PHY, attack="fgsm", eps=ILLU_EPS)
    pr = C.attack_success(model, Xte, PHY, attack="pgd", eps=ILLU_EPS, steps=7)
    return phys_acc, human_agree, fr["asr"], pr["asr"], float(ambiguous.float().mean())


def _run_c12(p):
    model = _train_illusion_model(p["kind"], p["scheme"], p["arch"], p["seed"])
    phys_acc, human_agree, fgsm_asr, pgd_asr, amb_frac = _illusion_eval(model, p["kind"], p["seed"])
    metrics = {"sub": "C12", "kind": p["kind"], "scheme": p["scheme"], "arch": p["arch"],
               "seed": p["seed"], "physical_acc": phys_acc, "human_agreement": human_agree,
               "fgsm_asr": fgsm_asr, "pgd_asr": pgd_asr, "ambiguous_frac": amb_frac}
    lines = [
        f"Theme C/C12 | illusion susceptibility + robustness | kind={p['kind']} "
        f"scheme={p['scheme']} arch={p['arch']} seed={p['seed']}",
        "",
        f"physical_acc       = {phys_acc:.4f}",
        f"human_agreement    = {human_agree:.4f}  (1.0 => model falls for illusion like a human)",
        f"fgsm_asr           = {fgsm_asr:.4f}",
        f"pgd_asr            = {pgd_asr:.4f}",
        "",
        "Compare scheme=physical vs scheme=human at fixed kind/arch/seed: if the",
        "human-labelled model has lower PGD ASR, training human-like confusion buys",
        "adversarial robustness.",
    ]
    return "\n".join(lines), metrics


class MultiTaskNet(nn.Module):
    """Shared CNN trunk; main classification head + auxiliary illusion (2-way) head."""
    def __init__(self, meta, width=32):
        super().__init__()
        self.trunk = C.SmallCNN(meta["channels"], meta["size"], meta["n_classes"], width=width)
        # reuse trunk.features; replace head usage by tapping flattened features
        feat = meta["size"] // 8
        self.flat_dim = width * 4 * feat * feat
        self.main_head = nn.Sequential(nn.Linear(self.flat_dim, 256), nn.ReLU(), nn.Linear(256, meta["n_classes"]))
        self.aux_head = nn.Sequential(nn.Linear(self.flat_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self._meta = meta

    def features(self, x):
        return self.trunk.features(x).flatten(1)

    def forward(self, x):
        return self.main_head(self.features(x))

    def aux_forward(self, x):
        return self.aux_head(self.features(x))


def _run_c3(p):
    ds = p["dataset"]
    meta = C.dataset_meta(ds)
    eps = 0.1 if meta["channels"] == 1 else 0.031
    C.set_seed(p["seed"])
    Xtr, Ytr, Xte, Yte = C.load_dataset(ds, n_train=6000, n_eval=1200, seed=p["seed"])
    net = MultiTaskNet(meta, width=32).to(C.DEVICE)

    aux_X = aux_Y = None
    if p["aug"] == "illusion_aux":
        # human-labelled illusion auxiliary task, resized to the main dataset shape
        k = C.ILLUSION_KINDS[p["seed"] % len(C.ILLUSION_KINDS)]
        aX, _, aHUM, _ = C.make_illusion_dataset(k, n=3000, label_scheme="human",
                                                 sz=meta["size"], seed=p["seed"])
        if meta["channels"] == 3:
            aX = aX.repeat(1, 3, 1, 1)
        aux_X, aux_Y = aX, aHUM

    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
    n = Xtr.size(0)
    net.train()
    for ep in range(6):
        perm = torch.randperm(n, device=C.DEVICE)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = F.cross_entropy(net(Xtr[idx]), Ytr[idx])
            if aux_X is not None:
                ai = torch.randint(0, aux_X.size(0), (128,), device=C.DEVICE)
                loss = loss + 0.3 * F.cross_entropy(net.aux_forward(aux_X[ai]), aux_Y[ai])
            loss.backward()
            opt.step()
        sched.step()
    net.eval()

    lg, acc = C.logits_and_acc(net, Xte, Yte)
    r = C.attack_success(net, Xte, Yte, attack="pgd", eps=eps, steps=7)
    margin = C.margin_of(lg, Yte)
    auroc = C.safe_auroc(r["flips"][r["correct"]].astype(int), -margin[r["correct"]])
    metrics = {"sub": "C3", "dataset": ds, "aug": p["aug"], "seed": p["seed"], "eps": eps,
               "clean_acc": acc, "pgd_asr": r["asr"], "margin_auroc": auroc}
    lines = [
        f"Theme C/C3 | illusion-auxiliary multitask robustness | dataset={ds} "
        f"aug={p['aug']} seed={p['seed']}",
        "",
        f"clean_acc    = {acc:.4f}",
        f"pgd_asr      = {r['asr']:.4f}",
        f"margin_auroc = {auroc:.4f}",
        "",
        "Compare aug=baseline vs aug=illusion_aux: lower pgd_asr with the human",
        "illusion auxiliary => making the machine share human confusion regularises",
        "toward robustness.",
    ]
    return "\n".join(lines), metrics


def _run_c4(p):
    model = _train_illusion_model(p["train_kind"], "physical", "cnn", p["seed"])
    phys_acc, human_agree, _, pgd_asr, amb = _illusion_eval(model, p["test_kind"], p["seed"])
    metrics = {"sub": "C4", "train_kind": p["train_kind"], "test_kind": p["test_kind"],
               "seed": p["seed"], "transfer_physical_acc": phys_acc,
               "transfer_human_agreement": human_agree, "transfer_pgd_asr": pgd_asr}
    lines = [
        f"Theme C/C4 | cross-illusion generalisation | train={p['train_kind']} "
        f"test={p['test_kind']}",
        "",
        f"transfer physical_acc    = {phys_acc:.4f}",
        f"transfer human_agreement = {human_agree:.4f}",
        f"transfer pgd_asr         = {pgd_asr:.4f}",
        "",
        "Does susceptibility to one illusion transfer to another (shared 'human-like'",
        "perceptual bias) or is each illusion learned idiosyncratically?",
    ]
    return "\n".join(lines), metrics


def run(spec):
    p = spec["params"]
    C.set_seed(p["seed"])
    sub = p["sub"]
    if sub == "C12":
        return _run_c12(p)
    if sub == "C3":
        return _run_c3(p)
    if sub == "C4":
        return _run_c4(p)
    raise ValueError(sub)
