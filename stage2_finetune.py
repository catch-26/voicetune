import json, os, sys, torch, jiwer
from collections import defaultdict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from common import DEV, split, fleurs, replay, audio, norm
BASE = "openai/whisper-base"
proc = WhisperProcessor.from_pretrained(BASE)
train, test = split()
replay_train, replay_eval = replay()
fleurs_train, fleurs_eval = fleurs()
if os.environ["MIX_PUBLIC"] != "none": train = train + fleurs_train
train = train + replay_train
print(f"train {len(train)} (fleurs {len(fleurs_train) if os.environ['MIX_PUBLIC'] != 'none' else 0}, replay {len(replay_train)}) own-test {len(test)} other-speaker-check {len(fleurs_eval)} forget-check {len(replay_eval)}")

def label_ids(r):
    proc.tokenizer.set_prefix_tokens(language=r["lang"], task="transcribe")
    return proc.tokenizer(r["text"]).input_ids

class DS(torch.utils.data.Dataset):
    def __init__(s, r): s.r = r
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        a, sr = audio(s.r[i])
        return {"input_features": proc.feature_extractor(a, sampling_rate=sr).input_features[0], "labels": label_ids(s.r[i])}

def collate(b):
    f = proc.feature_extractor.pad([{"input_features": x["input_features"]} for x in b], return_tensors="pt")
    l = proc.tokenizer.pad([{"input_ids": x["labels"]} for x in b], return_tensors="pt")
    labels = l["input_ids"].masked_fill(l["attention_mask"].ne(1), -100)
    if (labels[:, 0] == proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all(): labels = labels[:, 1:]
    return {"input_features": f["input_features"], "labels": labels}

def wer_by_lang(model, rs):
    model.eval(); hyp, ref = defaultdict(list), defaultdict(list)
    for r in rs:
        a, sr = audio(r)
        f = proc(a, sampling_rate=sr, return_tensors="pt").input_features.to(model.device)
        with torch.no_grad(): ids = model.generate(f, language=r["lang"], task="transcribe")
        hyp[r["lang"]].append(" ".join(norm(proc.batch_decode(ids, skip_special_tokens=True)[0]))); ref[r["lang"]].append(" ".join(norm(r["text"])))
    return {k: jiwer.wer(ref[k], hyp[k]) for k in ref}

model = WhisperForConditionalGeneration.from_pretrained(BASE).to(DEV)
model.config.forced_decoder_ids = None; model.generation_config.forced_decoder_ids = None
evaluate = lambda: {"own": wer_by_lang(model, test), "other_speakers": wer_by_lang(model, fleurs_eval), "other_langs": wer_by_lang(model, replay_eval)}
before = evaluate(); print("baseline", before)

args = Seq2SeqTrainingArguments(output_dir="models/ft_ckpt", per_device_train_batch_size=8, learning_rate=1e-5,
    num_train_epochs=float(os.environ["FT_EPOCHS"]), warmup_steps=20, logging_steps=10, save_strategy="no", report_to="none")
Seq2SeqTrainer(model=model, args=args, train_dataset=DS(train), data_collator=collate).train()

after = evaluate(); print("finetuned", after)
json.dump({"baseline": before, "finetuned": after}, open("models/eval.json", "w"), indent=1)
mean = lambda d: sum(d.values()) / max(1, len(d))
tol = 1 + float(os.environ["FORGET_TOL"])
worse = lambda grp: [k for k in after[grp] if after[grp][k] > before[grp][k] * tol]
problems = ([] if not test or mean(after["own"]) < mean(before["own"]) else ["own-voice WER did not improve"]) \
    + [f"other speakers worse in {k}" for k in worse("other_speakers")] + [f"forgetting in {k}" for k in worse("other_langs")]
if problems and not os.environ.get("FORCE"):
    sys.exit("stopped: " + "; ".join(problems) + ". see models/eval.json; FORCE=1 ./run.sh to continue anyway")
model.save_pretrained("models/ft"); proc.save_pretrained("models/ft")
