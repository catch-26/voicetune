import copy, os, random, torch
from torch.nn import functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from common import DEV, split, fleurs, replay, audio
EPOCHS = int(os.environ["ACFT_EPOCHS"])
proc = WhisperProcessor.from_pretrained("models/ft")
full = WhisperForConditionalGeneration.from_pretrained("models/ft")
ref = full.model.to(DEV).eval()
tgt = copy.deepcopy(full.model).to(DEV).train()
train, _ = split()
train = train + replay()[0]
if os.environ["ACFT_PUBLIC"] == "y": train = train + fleurs()[0]
print(f"acft clips {len(train)}")

def enc(m, x, n_ctx):
    d = 2 * n_ctx - x.shape[2]
    x = F.pad(x, [0, d]) if d > 0 else x[:, :, :x.shape[2] + d]
    e = m.encoder
    if n_ctx == 1500: return e(x).last_hidden_state
    h = F.gelu(e.conv1(x)); h = F.gelu(e.conv2(h)).permute(0, 2, 1)
    h = F.dropout(h + e.embed_positions.weight[:n_ctx], p=e.dropout, training=e.training)
    for layer in e.layers: h = layer(h, None, layer_head_mask=None)[0]
    return e.layer_norm(h)

def step(r):
    a, sr = audio(r); length = len(a) / sr
    if length > 29: return None
    x = proc.feature_extractor(a, sampling_rate=sr, return_tensors="pt").input_features.to(DEV)
    proc.tokenizer.set_prefix_tokens(language=r["lang"], task="transcribe")
    ids = torch.tensor([proc.tokenizer(r["text"]).input_ids]).to(DEV)
    n = int(round(1500 / 30 * length)); j = min(64, n // 3)
    n += random.randint(-j, j)
    out_t = tgt.decoder(input_ids=ids, encoder_hidden_states=enc(tgt, x, n), output_hidden_states=True)
    with torch.no_grad():
        out_r = ref.decoder(input_ids=ids, encoder_hidden_states=enc(ref, x, 1500), output_hidden_states=True)
    loss = F.mse_loss(torch.cat(out_t.hidden_states, 0), torch.cat(out_r.hidden_states, 0))
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

opt = torch.optim.Adam(tgt.parameters(), lr=1e-6)
for ep in range(EPOCHS):
    random.Random(ep).shuffle(train); losses = [l for r in train if (l := step(r)) is not None]
    print(f"epoch {ep} loss {sum(losses)/len(losses):.5f}")
full.model = tgt.eval().cpu()
full.save_pretrained("models/acft"); proc.save_pretrained("models/acft")
