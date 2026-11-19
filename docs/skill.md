---
name: unveil
description: Installs UNVEIL, which answers a short phrase with a map of that concept from SAM 3's semantic path, into a new virtual environment, runs a first prediction on the bundled example image and draws the maps. Use it when a user asks to install, set up, try or demo UNVEIL, or wants a map for a phrase such as "shadow", "mirror", "edges" or "salient object" on their own images.
---

# UNVEIL: install and first prediction

UNVEIL answers a short phrase with a map of the concept it names, or with "none". It reads the maps from a trained
copy of SAM 3's semantic path and decides presence with a separate decoder. One set of weights covers light (shadows,
transparent objects, mirrors), context (salient and camouflaged objects), structure (edges, skeletons) and object
categories. These steps install it, run a first prediction, draw a figure and point to the rest.

## 1. Get the repository

If the current folder holds `scripts/predict.py` and `unveil/prompts.py`, it is the repository root, so skip the
clone. Otherwise run the line below. Run every later command from the repository root.

```bash
git clone https://github.com/yuhuan-wu/UNVEIL.git && cd UNVEIL
```

## 2. Check the machine

Run these checks and report the results to the user before you install anything. Stop if one fails.

```bash
python3 --version
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h .
```

- Python 3.10 or newer is required.
- UNVEIL needs a CUDA GPU. Prediction does not run on the CPU.
- The install takes PyTorch wheels built for CUDA 12.8, which bring their own CUDA libraries. Report the driver
  version from the `nvidia-smi` header. Step 4 checks that PyTorch sees the GPU.
- One query peaks at 4.2 GiB of allocated GPU memory, and eight prompts on one image at 6.6 GiB.
- The environment, SAM 3's checkpoint and the 138 MB UNVEIL checkpoint need several gigabytes of disk.

## 3. Ask the user for access to SAM 3's checkpoint, and do not do it yourself

SAM 3's checkpoint `facebook/sam3` on Hugging Face is gated. Ask the user to accept its licence on
https://huggingface.co/facebook/sam3 and to run `hf auth login` in their own terminal, with the environment of
step 4 active. Never ask for a token, never type, print or store one, and never run `hf auth token`. After the
login, `hf auth whoami` shows the account name. Without access to Hugging Face, the user needs a local `sam3.pt`.
`checkpoints/README.md` gives a download from ModelScope into `checkpoints/sam3.pt`, where the code finds it, and
a file at another place is passed as `--sam3-checkpoint`.

## 4. Install into a new virtual environment

Create a new environment inside the repository. Do not install into an environment the user already uses unless the
user asks for it.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install "git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40"
pip install -e ".[eval,data,test]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

These are the commands of the README's Quick start and two checks. The `python -c` line must print `True`. If it
prints `False`, PyTorch does not see the GPU, most often because the NVIDIA driver is too old for these wheels, so
stop and tell the user. `pytest -q` runs the CPU tests. Prediction and drawing need none of the three extras. If the
`eval` extra fails to build, run `pip install -e ".[data,test]"` instead and say so.

## 5. Download and check the UNVEIL checkpoint

```bash
wget -P checkpoints https://github.com/yuhuan-wu/UNVEIL/releases/download/v1.0/unveil_v1.safetensors
sha256sum checkpoints/unveil_v1.safetensors
cat checkpoints/unveil_v1.safetensors.sha256
```

The file is 138 MB. Skip the download if `checkpoints/unveil_v1.safetensors` is already there. The last two lines
must show `ebff5143f0d198e3a8a59d195b3c1e2e6a81a810bc152d4657f83276f414f42f`, as in `checkpoints/README.md`.
`unveil.load()` makes the same check against the `.sha256` file and stops on a mismatch.

## 6. First prediction

```bash
python scripts/predict.py assets/examples/coco_000000228144.jpg --prompts mirror edges "salient object" person --overlay --out outputs/example
```

Without a local `sam3.pt`, the first run downloads SAM 3's checkpoint into the Hugging Face cache. The script prints
one JSON line per prompt: `presence` is the presence probability, `threshold` the prompt's threshold, `accepted`
whether presence reaches it, and `mean` the mean of the map. In `outputs/example/` it writes
`coco_000000228144__<prompt>.png` per prompt, an 8-bit grey map at the image size (white = 1), all black when the
prompt is rejected. Spaces in a prompt become `_`. `--overlay` adds `coco_000000228144__<prompt>__overlay.jpg`, and
`predictions.json` holds the printed records. On this image `mirror`, `edges` and `salient object` are accepted.
`person` is rejected, with a presence of about 0.02 against a threshold of 0.28, so its map is all black.

The same in Python:

```python
import unveil

model = unveil.load()
image = "assets/examples/coco_000000228144.jpg"
mask, presence = model.predict(image, "mirror")
masks, presences = model.predict(image, ["edges", "salient object", "person"])
```

`mask` is a float32 array of shape (H, W) with values in [0, 1], here (426, 640). `presence` is a float in [0, 1].
A map is all zero when its presence is below `model.threshold(prompt)`, and `gate=False` returns it anyway. A list of
prompts passes the image through the ViT once.

## 7. Prompts

The task prompts, as defined in `unveil/prompts.py`:

- Light, region maps: `"shadow"`, `"transparent object"`, `"mirror"`.
- Context, region maps: `"salient object"`, `"camouflaged object"`.
- Structure, soft line maps: `"edges"`, `"skeleton"`, `"symmetry axis"`.

Matching ignores case and extra spaces. Any other phrase is free text, such as `person` or `dog`. Category names work
through SAM 3's vocabulary, and every free-text phrase uses the pooled presence threshold. The presence decoder saw
verified negatives for `"salient object"`, `"camouflaged object"`, category names and, on 87 LVIS images,
`"mirror"`. For these prompts an empty answer is reliable. The other task prompts may return a map on an image
without the concept.

The joint query takes exactly the two Context prompts: `--joint` in `scripts/predict.py`, or `joint=True` in
`model.predict`. It splits a scene between the two concepts, and each member gets its own presence decision. With
`--overlay` it also writes `<image>__joint.jpg`. On the example image both members come back empty. `--path sam3`,
or `path="sam3"` in Python, reads SAM 3's own semantic path for comparison, with no presence decision.

## 8. Draw a figure

```bash
python scripts/visualize.py panel assets/examples/coco_000000228144.jpg --prompts mirror edges "salient object" person --joint --out outputs/panel.png
```

It writes `outputs/panel.png`, the image and one overlay per prompt. Each title gives the presence probability, and
"(none)" marks a rejected prompt. The panel draws every map without the presence decision, so a rejected prompt
still shows its map. `--joint` also writes `outputs/panel_joint.jpg`. Open the image and show it to the user.

## 9. Where to go next

- Evaluation on PCM-Bench: the README section "Evaluate". Maps of another model: "Score another model".
- Training: the README section "Train", with the recipe in `configs/unveil.yaml`.
- Data preparation: the README section "Data" and `datasets/README.md`.

## 10. Troubleshooting

- `GatedRepoError`, or a 401 or 403 error for `facebook/sam3`: the user has not accepted the licence or is not
  logged in. Ask for both (step 3), then run the command again. On a connection error, use a local `sam3.pt`.
- `FileNotFoundError` that points to `checkpoints/README.md`: the UNVEIL checkpoint is not in `checkpoints/`. Run
  the download of step 5.
- `sha256 mismatch`: the checkpoint is incomplete or another version. Remove the file and download it again.
- CUDA out of memory: free the GPU or send fewer prompts per call. One query needs about 4.2 GiB.
- `dtype must be torch.bfloat16 ... or torch.float16`: `unveil.load(dtype=...)` takes only these two. Keep the
  default bfloat16.
- `UNVEIL needs a CUDA GPU`: PyTorch sees no CUDA GPU. Without one, only the CPU tests, data preparation and the
  scoring of stored maps run.

## 11. Report back to the user

- the repository path, the environment path (`.venv` in the repository), and the Python, PyTorch and GPU found;
- whether the install included the `eval` extra, and the test result;
- the result of the checkpoint check;
- how to repeat a prediction: `source .venv/bin/activate`, then the `scripts/predict.py` line of step 6;
- the output folder `outputs/example/` and the figure `outputs/panel.png`, which you showed;
- anything that failed, and what the user has to do next.
