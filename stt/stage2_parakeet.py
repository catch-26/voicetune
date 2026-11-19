import json, math, os, sys, torch, jiwer, soundfile as sf
import lightning.pytorch as pl
from collections import defaultdict
from nemo.collections.asr.models import ASRModel
from huggingface_hub import hf_hub_download
from hf_pins import rev
from common import DEV, BASE, split, fleurs, replay, norm
from tdt_loss import TDTLossTorch
BS = 8
os.makedirs("models/parakeet", exist_ok=True)
train, test = split()
replay_train, replay_eval = replay()
fleurs_train, fleurs_eval = fleurs()
if os.environ["MIX_PUBLIC"] != "none": train = train + fleurs_train
train = train + replay_train
print(f"train {len(train)} (fleurs {len(fleurs_train) if os.environ['MIX_PUBLIC'] != 'none' else 0}, replay {len(replay_train)}) own-test {len(test)} other-speaker-check {len(fleurs_eval)} forget-check {len(replay_eval)}")

def manifest(rs, path):
    with open(path, "w", encoding="utf8") as f:
        for r in rs: f.write(json.dumps({"audio_filepath": os.path.abspath(r["wav"]), "duration": sf.info(r["wav"]).duration, "text": r["text"]}, ensure_ascii=False) + "\n")
    return path

def wer_by_lang(model, rs):
    if not rs: return {}
    out = model.transcribe([r["wav"] for r in rs], batch_size=BS, verbose=False)
    out = out[0] if isinstance(out, tuple) else out
    hyp, ref = defaultdict(list), defaultdict(list)
    for r, h in zip(rs, out):
        hyp[r["lang"]].append(" ".join(norm(getattr(h, "text", h)))); ref[r["lang"]].append(" ".join(norm(r["text"])))
    return {k: jiwer.wer(ref[k], hyp[k]) for k in ref}

src = BASE if BASE.endswith(".nemo") else hf_hub_download(BASE, BASE.rsplit("/", 1)[-1] + ".nemo", revision=rev(BASE))
model = ASRModel.restore_from(src, map_location="cpu").to(DEV)
nemo_loss = getattr(getattr(model, "loss", None), "_loss", None)
if type(nemo_loss).__name__ == "TDTLossNumba" and not (DEV == "cuda" and torch.version.cuda):
    model.loss._loss = TDTLossTorch(nemo_loss.blank, nemo_loss.durations, nemo_loss.sigma)
evaluate = lambda: {"own": wer_by_lang(model, test), "other_speakers": wer_by_lang(model, fleurs_eval), "other_langs": wer_by_lang(model, replay_eval)}
before = evaluate(); print("baseline", before)

gpu = DEV == "cuda"
trainer = pl.Trainer(accelerator="gpu" if gpu else "cpu", devices=1, max_epochs=int(os.environ["FT_EPOCHS"]),
    precision="bf16-mixed" if gpu else "32-true", logger=False, enable_checkpointing=False, limit_val_batches=0,
    gradient_clip_val=1.0, log_every_n_steps=10)
model.set_trainer(trainer)
model.setup_training_data({"manifest_filepath": manifest(train, "models/parakeet/train.json"), "sample_rate": 16000,
    "batch_size": BS, "shuffle": True, "num_workers": 2 if sys.platform == "linux" else 0, "pin_memory": gpu, "use_lhotse": False, "max_duration": 30.0, "trim_silence": False})
steps = math.ceil(len(train) / BS) * int(os.environ["FT_EPOCHS"])
model.setup_optimization({"name": "adamw", "lr": 2e-5, "weight_decay": 1e-3,
    "sched": {"name": "CosineAnnealing", "warmup_steps": min(50, steps // 10), "min_lr": 1e-6, "max_steps": steps}})
trainer.fit(model)
model = model.to(DEV)

after = evaluate(); print("finetuned", after)
json.dump({"baseline": before, "finetuned": after}, open("models/eval.json", "w"), indent=1)
mean = lambda d: sum(d.values()) / max(1, len(d))
tol = 1 + float(os.environ["FORGET_TOL"])
worse = lambda grp: [k for k in after[grp] if after[grp][k] > before[grp][k] * tol]
problems = ([] if not test or mean(after["own"]) < mean(before["own"]) else ["own-voice WER did not improve"]) \
    + [f"other speakers worse in {k}" for k in worse("other_speakers")] + [f"forgetting in {k}" for k in worse("other_langs")]
if problems and not os.environ.get("FORCE"):
    sys.exit("stopped: " + "; ".join(problems) + ". see models/eval.json; FORCE=1 ./run.sh to continue anyway")
model.save_to("models/parakeet/ft.nemo")
