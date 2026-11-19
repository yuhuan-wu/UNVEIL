# Checkpoints

| File | Size | sha256 |
| --- | --- | --- |
| `unveil_v1.safetensors` | 138 MB | `ebff5143f0d198e3a8a59d195b3c1e2e6a81a810bc152d4657f83276f414f42f` |

Download the file into this folder, from the repository root:

```bash
wget -P checkpoints https://github.com/yuhuan-wu/UNVEIL/releases/download/v1.0/unveil_v1.safetensors
```

`unveil.load()` and the scripts read `checkpoints/unveil_v1.safetensors` by default and check it against the `.sha256`
file next to it.

The file holds the trained tensors only, 36.1M parameters in fp32.

| Group | Parameters | Content |
| --- | --- | --- |
| `heads.` | 22,359,602 | fusion encoder, semantic path and presence decoder, trained copies of SAM 3's modules |
| `lora.` | 2,260,992 | LoRA of rank 16 and scaling factor 32 on the attention and MLP layers of ViT blocks 24 to 31 |
| `joint.decoder.` | 11,441,969 | presence decoder of the joint query {salient object, camouflaged object} |
| `joint.convention` | 512 | the two convention vectors, 2 x 256 |

The LoRA factors are merged into copies of the blocks when the model loads. The metadata of the file holds two kinds
of threshold. The presence thresholds were fitted on validation pairs from the training splits, and the binarization
thresholds of the region metrics on training images. No test image was used for either.

| Threshold | Value |
| --- | --- |
| presence, "salient object" | 0.500 |
| presence, "camouflaged object" | 0.125 |
| presence, other task prompts and free text | 0.280 |
| presence, joint query, salient / camouflaged | 0.861 / 0.177 |
| binarization of the 8-bit map, shadow / transparent object / mirror | 45 / 126 / 132 |

## SAM 3's checkpoint

The rest of the model, such as the ViT, the text and geometry encoders and SAM 3's own heads, comes from SAM 3's
checkpoint `sam3.pt`. By default the first run downloads it from `facebook/sam3` on Hugging Face into the Hugging
Face cache. That repository is gated, so accept its licence on the model page and run `hf auth login` once.

A local `sam3.pt` needs no access to Hugging Face. The code looks for one in this order:

1. `--sam3-checkpoint /path/to/sam3.pt` in the scripts, or `unveil.load(sam3_checkpoint="/path/to/sam3.pt")`;
2. the environment variable `UNVEIL_SAM3`;
3. `checkpoints/sam3.pt` in this repository.

ModelScope hosts a copy of the file. From the repository root:

```bash
wget -P checkpoints https://www.modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt
```

| File | Size | sha256 |
| --- | --- | --- |
| `sam3.pt` | 3.2 GB | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |

## Metadata

`python -c "import json, unveil.checkpoint as c; print(json.dumps(c.read_meta('checkpoints/unveil_v1.safetensors'), indent=1))"`
prints the metadata. `scripts/train.py` writes its checkpoint in the same format.
