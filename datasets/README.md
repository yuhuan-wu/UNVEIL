# Datasets

UNVEIL trains on the official training splits of 11 public datasets plus LVIS v1 val, and is evaluated on
PCM-Bench (15 test sets, a rejection split and LVIS retention pairs) and 6 extra test sets. This folder holds the
lists that define every split (`manifests/`, tracked) and tells you where to get the images. Nothing here
redistributes data: you download each archive from its source and `scripts/prepare_data.py` builds the data root.

## Quick start

```bash
export UNVEIL_DATA=/data/unveil                         # default: datasets/data inside the repository
python scripts/prepare_data.py --datasets all --download            # fetch, check, unpack, convert, verify
python scripts/prepare_data.py --datasets all --raw ~/downloads     # or: use archives you downloaded yourself
python scripts/prepare_data.py --datasets test --verify-only        # re-check an existing root
python scripts/prepare_data.py --list                                # keys, roles and archive URLs
```

`--datasets` takes dataset keys (table below), roles (`train`, `test`, `extra`) or `all`. A role selects only
the archives and files it needs (for example, `test` skips the 4.5 GB Trans10K training archive). `--raw` folders
(repeatable) are searched for each archive under the file name in the archive table (or the name in its URL) and
must hold it unmodified. Every archive is checked against its SHA-256 before it is unpacked. The run ends with a
check of every file the manifests reference against `manifests/checksums/`, and exits with status 1 if a file is
missing or differs.

Requirements: Python 3.10+, numpy, Pillow, scipy. `pycocotools` for the LVIS masks, `pyarrow` for GSD and PMD,
and `bsdtar` (libarchive), `unrar` or `7z` for the ISTD `.rar`. About 23 GB of disk for the unpacked data root
with every role, plus the archives (about 21 GB, or 41 GB if you place the COCO 2017 zips in `--raw`).

## Downloads

Every archive has a direct link, so `--download` can fetch all of them. Google Drive files are fetched through
`drive.usercontent.google.com`. Drive may refuse a busy file ("quota exceeded") or return a sign-in page, and the
script then stops with a message. Download that file in a browser and pass its folder with `--raw`.

"Mirror" marks a copy linked from a widely used repository, used because the authors' link needs a login, an
application form or is gone.

| Key | Dataset | Used for (images) | Source | Terms (as stated by the source) |
| --- | --- | --- | --- | --- |
| `duts_tr` | DUTS-TR | salient, train (10,553) | [saliencydetection.net/duts](https://saliencydetection.net/duts/) | No licence. "All rights reserved by the original authors of DUTS Image Dataset" (GT). |
| `duts_te` | DUTS-TE | salient, test (5,019) | same | same |
| `ecssd` | ECSSD | salient, test (1,000) | [CUHK](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) | No licence or terms stated. |
| `hkuis` | HKU-IS | salient, test (4,447) | mirror, Drive file from [InSPyReNet](https://github.com/plemeri/InSPyReNet) ([project](https://i.cs.hku.hk/~gbli/deep_saliency.html)) | No licence stated. |
| `dutomron` | DUT-OMRON | salient, test (5,168) | [saliencydetection.net/dut-omron](https://saliencydetection.net/dut-omron/) | No licence. "All rights reserved by the original authors of DUT-OMRON Image Dataset" (GT). |
| `pascals` | PASCAL-S | salient, test (850) | mirror, Drive file from InSPyReNet ([project](https://ccvl.jhu.edu/datasets/)) | No licence stated; PASCAL VOC images. |
| `camo` | CAMO | camouflaged, train (1,000) and test (250) | [project page](https://sites.google.com/view/ltnghia/research/camo) | CC BY-NC-SA 4.0. |
| `cod10k` | COD10K-v3 | camouflaged, train (3,040) and test (2,026); "camouflaged object" negatives (2,655 train); rejection (784 test background) | [project page](https://dengpingfan.github.io/pages/COD.html) | Non-commercial only. |
| `nc4k` | NC4K | camouflaged, test (4,121) | [COD-Rank-Localize-and-Segment](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment) | No licence stated. |
| `sbu` | SBU | shadow, train (4,085) and test (638) | [Stony Brook](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html) | Non-commercial. |
| `istd` | ISTD | shadow, train (1,330) and test (540) | [ST-CGAN](https://github.com/DeepInsight-PCALab/ST-CGAN) | Research and non-commercial only. |
| `trans10k` | Trans10K v1 | transparent, train (5,000) and test (4,428) | [TransLab](https://xieenze.github.io/projects/TransLAB/TransLAB.html) | No dataset licence; code Apache-2.0 for academic use, commercial use by request. |
| `msd` | MSD | mirror, train (3,063) and test (955) | mirror, Drive file from [Mirror-Glass-Detection](https://github.com/Charmve/Mirror-Glass-Detection) ([project](https://mhaiyang.github.io/ICCV2019_MirrorNet/)) | No licence stated; official access by application. |
| `bsds500` | BSDS500 | edges, test (200); train / val for the edge calibration | [Berkeley](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) | No licence stated; cite Arbelaez et al., TPAMI 2011. |
| `hed_bsds` | HED-BSDS | edges, train (28,800 = 300 BSDS500 images x 96 augmentations) | [HED](https://github.com/s9xie/hed) | No licence stated; cite HED. |
| `sklarge` | SK-LARGE | skeleton, train (746) and test (745) | mirror, Drive file from [DeepFlux](https://github.com/YukangWang/DeepFlux) | No dataset licence; MS-COCO crops. |
| `sympascal` | SYM-PASCAL | symmetry axis, train (647) and test (788) | mirror, Drive file from DeepFlux ([SRN](https://github.com/KevinKecc/SRN)) | No dataset licence; PASCAL VOC 2011 images. |
| `usc12k` | USC12K | salient / camouflaged switching (2,100 train scenes), negatives (1,884), joint presence (8,540 items), validation, rejection (3,600 test) | [USCNet](https://github.com/ssecv/USCNet) | No dataset licence stated. |
| `lvis` | LVIS v1 val + COCO 2017 images | concept replay (10,302 images), presence negatives (10,273), retention (2,294 pairs), rejection (400) | [LVIS](https://www.lvisdataset.org/dataset), [COCO](https://cocodataset.org) | Annotations CC BY 4.0; images under the Flickr terms (COCO terms of use). |
| `ucf` | UCF | shadow, extra test (110) | Drive file from [MTMT](https://github.com/eraserNut/MTMT) | No licence stated. |
| `gsd` | GSD | glass ("transparent object"), extra test (813) | the authors' Hugging Face upload [garrying/GSD](https://huggingface.co/datasets/garrying/GSD) | CC BY-NC 4.0. |
| `pmd` | PMD | mirror, extra test (571) | the authors' Hugging Face upload [garrying/PMD](https://huggingface.co/datasets/garrying/PMD) | CC BY-NC 4.0. |
| `chameleon` | CHAMELEON | camouflaged, extra test (76) | Drive file from [SINet-V2](https://github.com/GewelsJI/SINet-V2) | No licence stated. |
| `biped2` | BIPEDv2 | edges, extra test (50) | Drive folder from [MBIPED](https://github.com/xavysp/MBIPED) | MIT per the MBIPED repository. |
| `whsymmax` | WH-SYMMAX | skeleton, extra test (100) | Drive file from [BlumNet](https://github.com/cong-yang/BlumNet) | No dataset licence; Weizmann Horse images. |

Most of these datasets allow research use only and several forbid commercial use. Check each source's terms
before you use the data.

Mirror and source notes:
- HKU-IS: the authors' Drive link asks for a Google login.
- PASCAL-S: the authors' server is unreachable. The masks are graded (k of 12 annotators); GT = mask >= 129.
- MSD: official access is an application form.
- SK-LARGE and SYM-PASCAL: the official links are down or need a OneDrive login. SYM-PASCAL holds 788 test and 647
  train images (the SRN paper quotes 787 and 648).
- Trans10K: v1, which has the same images as v2.
- LVIS: only the COCO 2017 images that the lists use (14,366 files) are needed. Put `train2017.zip` and
  `val2017.zip` (http://images.cocodataset.org/zips/) in a `--raw` folder, or let `--download` fetch the images
  one by one from `images.cocodataset.org`.
- GSD, PMD: the parquet files of the authors' uploads at fixed revisions (URLs in `--list`). Both are named
  `test-00000-of-00001.parquet` upstream; the script tells them apart by size, or name them as in the table.

### Archives

| Key | File | Bytes | SHA-256 |
| --- | --- | --- | --- |
| duts_tr | `DUTS-TR.zip` | 270,997,309 | `ce61e023c8f59d022b4d46981cf16813b83d089242e6489a45630d83962ea058` |
| duts_te | `DUTS-TE.zip` | 139,799,089 | `d9905c33307e1d35c397604a701ed87f019acf060033076eec633f13a2eb8104` |
| ecssd | `images.zip` | 67,766,979 | `7d5dce1a21c6d82e2f9617c00ad7eae2de5ded9320e8c59a01b4c731d210ea47` |
| ecssd | `ground_truth_mask.zip` | 1,571,109 | `dc0a9f11f2adee2737d95b28f9567a0a5de33f78743447c097ceb4d42f281a31` |
| hkuis | `HKU-IS.zip` | 944,572,407 | `c2c15b80a57bd022b68257ed52074406ebf8d991334b6744416d752e9cd0f441` |
| dutomron | `DUT-OMRON-image.zip` | 114,809,842 | `217ac4ed3893a318c28639a1212fd7086f71c56f70dcf0d033c36d97b001f789` |
| dutomron | `DUT-OMRON-gt-pixelwise.zip.zip` | 6,600,590 | `8307bb97db7ab3b2eb2ea8004404d67a15d9e4086079e150db6811563d2ec8e3` |
| pascals | `PASCAL-S.zip` | 102,894,015 | `5d22955df6448bcb864d318161c6da9279c6bfd9676a862915a994de60bae861` |
| camo | `CAMO-V.1.0.zip` | 286,249,800 | `510eaeef5bbcb626016b1a13f61e269002f60b8b05234cd7d51a9a7399410cc5` |
| cod10k | `COD10K-v3.zip` | 2,416,808,676 | `3bd100640dde83b05f9af2932ae633d4894c899092fcbda712f7ff447a97b1e4` |
| nc4k | `NC4K.zip` | 600,648,391 | `1e9c446fade4647c38952b496eed355686b90c408bbf17474368b1d8b4fd5695` |
| sbu | `SBU-shadow.zip` | 299,916,461 | `8ac703bc4ec0cf3f57e128a6e7a8ed73680d9f68e4e22d08fc83a3f60c612109` |
| istd | `ISTD_Dataset.rar` | 2,139,405,436 | `46debc393465044f25d682c2d500cee2c0a9063253040c35b4a10f79b90bc389` |
| trans10k | `train.zip` | 4,533,524,305 | `5c388786af167c640f472c455a7fd1851d591632f1d419873cf4a8070eb99869` |
| trans10k | `test.zip` | 3,997,412,881 | `514d3788091058e41ad226a9d3a99ca29523150403f9e5a1efc99b23e1bd88bb` |
| msd | `MSD.zip` | 167,173,069 | `bdea8d846d0aff30c2021b085c197459d34e105d61e1165a4e03473774ab2843` |
| bsds500 | `BSR_bsds500.tgz` | 70,763,455 | `97e49d31764f3912f0c4122707d53062ac9e783ba0f095e447a4d53c1a41af8e` |
| hed_bsds | `HED-BSDS.tar` | 1,303,992,320 | `0db652f768291e316281ae41a5d0b7cde1145ddd3d3c4b959ae87820b5146a40` |
| sklarge | `SK-LARGE.zip` | 32,259,568 | `f298e1e6ab3b35dc07437c77d1bd553c34ad3d70282a98632336bcb22fe7b41f` |
| sympascal | `SymPASCAL-by-KZ.zip` | 77,458,583 | `ae9ab56415f1e15d8e8ffe58d3dcba8a98933190bcaf1115cae90331fb7fbd00` |
| usc12k | `VOC-USC12K.zip` | 2,310,013,741 | `41e0573439aa42a565a16679ba2aaf0f8f042a1ecb28d26902285acf4e1a067c` |
| lvis | `lvis_v1_val.json.zip` | 64,026,968 | `5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5` |
| ucf | `UCF.zip` | 16,577,419 | `05ed1d5edeb9b6b3511a98f184abd750f725e4f4c4e98595edfcc196a22b77b7` |
| gsd | `GSD-test-00000-of-00001.parquet` | 33,686,132 | `c1ea301305f04d9d05ef09dbf2982e4ce188fb8d779c85095017c54c3a688963` |
| pmd | `PMD-test-00000-of-00001.parquet` | 169,600,933 | `59381ad0a10f79b72f56bae4f37437682bd366d0c9410d8eff2fc1683d5f9ad8` |
| chameleon | `COD-TestDataset.zip` | 1,310,372,023 | `5d1a4244d513f78682b42cb89b759d44cc462ce9d3ae78d18974b2790e3aa310` |
| biped2 | `BIPEDv2.zip` | 35,139,513 | `5fdcd9ed7e8d0c6614858a63d80f1bef523e6c550ac0133299bc962648dab162` |
| whsymmax | `WH-SYMMAX.tar.xz` | 5,347,136 | `7bcf79479f0d4633b9f06122eae17990c6dfe55f244960ae53dfef28d24541ba` |

## Layout

Each archive is unpacked as released, so the folders are the archives' own. Archives without a top folder go
into a folder named after the dataset (`ECSSD/`, `DUT-OMRON/`, `NC4K/`, `Trans10K/`, `lvis/`). `derived/` holds
the converter outputs; `.unveil/` holds the unpack markers that make a rerun skip finished archives.

```
$UNVEIL_DATA/
  DUTS-TR/  DUTS-TE/  ECSSD/  HKU-IS/  DUT-OMRON/  PASCAL-S/       salient objects
  CAMO-V.1.0-CVIU2019/  COD10K-v3/  NC4K/                          camouflaged objects
  SBU-shadow/  ISTD_Dataset/  Trans10K/{train,test}/  MSD/          shadows, transparent objects, mirrors
  BSR/BSDS500/data/  HED-BSDS/  SK-LARGE/  SymPASCAL-by-KZ/          edges, skeletons
  VOC-USC12K/                                                      USC12K
  lvis/lvis_v1_val.json  coco/{train2017,val2017}/                  LVIS (only the images the lists use)
  UCF/  GSD/test/  PMD/test/  COD-TestDataset/CHAMELEON/  BIPEDv2/  WH-SYMMAX/      extra test sets
  derived/
    sbu_gt/train/  trans10k_gt/train/  sklarge_gt/train/  sympascal_gt/train/
    usc12k_gt_train/{sal,cam}/  lvis_masks/
```

Converters (`unveil/data/prepare/convert.py`, `lvis.py`) write {0, 255} 8-bit gray PNGs:
- `sbu_gt/train`: SBU training masks are graded (JPEG-like noise); foreground = any colour channel >= 128.
- `trans10k_gt/train`: Trans10K RGBA masks, red "things" and white "stuff"; foreground = any RGB channel >= 128.
- `sklarge_gt/train`, `sympascal_gt/train`: the `.mat` skeleton map (`symmetry`, or `skeleton` in SYM-PASCAL
  train) > 0.
- `usc12k_gt_train/{sal,cam}`: USC12K class-index PNGs; salient = index 1 (palette red), camouflaged = index 2
  (palette green), decoded by palette colour.
- `lvis_masks/<12-digit image id>_<category id>.png`: the union of every instance of the category on the
  image, each instance the union of its polygons (pycocotools, as the LVIS API).

GSD and PMD images and masks are written from the parquet files with their original bytes. Test ground truth is
read from the released files (GT modes below), so it needs no converter.

## Manifests

All paths are relative to the data root; `unveil.data.catalog` joins them with the root. Files larger than
2 MB are gzipped (`.jsonl.gz`); the catalog reads either form.

```
manifests/
  train/<stream>.jsonl[.gz]   13 training streams, in training order
  val/single.jsonl, val/joint.jsonl     presence validation pairs (threshold fitting)
  test/<set>.jsonl            21 test sets + bsds500_train / bsds500_val
  rejection.jsonl             4,784 queries
  retention.jsonl             LVIS pairs, split "heldout" (2,294) and "seen" (6,255)
  prompts.json                task prompts, replay noun phrases, negative-only noun phrases
  checksums/<key>.tsv.gz      kind, digest, path of every referenced file
  summary.json                list sizes and the training sampler's groups
```

### Training streams (`catalog.train_index`)

| Stream | Items | Prompt id | Content |
| --- | --- | --- | --- |
| `sod` | 10,553 | 0 | DUTS-TR |
| `cod` | 4,040 | 1 | COD10K-TR camouflaged (3,040) + CAMO-TR (1,000) |
| `edge` | 28,800 | 2 | HED-BSDS augmented BSDS500 train + val |
| `shadow` | 5,415 | 3 | SBU train (4,085) + ISTD train (1,330) |
| `transparent` | 5,000 | 4 | Trans10K v1 train |
| `mirror` | 3,063 | 5 | MSD train |
| `skeleton` | 1,393 | 6, or 7 when `src` is `sympascal` | SK-LARGE train (746) + SYM-PASCAL train (647) |
| `switch` | 2,100 | 0 on `gt`, 1 on `gt2` | USC12K train scenes with a salient and a camouflaged object |
| `neg_cam` | 4,539 | 1 (presence 0) | COD10K-TR background and non-camouflaged (2,655) + USC12K "none" (1,884) |
| `neg_sal` | 1,884 | 0 (presence 0) | USC12K train "none" scenes |
| `replay` | 10,302 | 8 + k | LVIS images, replay categories |
| `replay_neg` | 10,273 | ids in `negs` (presence 0) | LVIS images with verified-absent categories |
| `joint` | 8,540 | 0 and 1 together | joint presence items: both 1,920, single 2,081, none 1,884, bg 2,655 |

Fields:
- Dense streams and the negatives: `src` (source key: `duts_tr`, `cod10k_tr`, `camo_tr`, `hed_bsds`, `sbu`,
  `istd`, `trans10k`, `msd`, `sklarge`, `sympascal`, `usc12k`), `id` (file stem; absent in `sod`, `cod`,
  `edge`), `image`, `gt` (null for images without an object), `gt2` (`switch`: the camouflaged mask; else null),
  `w`, `h` (image size), `downscale` (only when true; see below). Region GT: foreground = gray >= 128. `edge` GT is
  HED's soft annotator mean (8-bit; >= 127.5 edge, 0 non-edge, values between are ignored by the loss). Skeleton GT
  is {0, 255}. The COD10K-TR non-camouflaged negatives carry their object mask in `gt`; it is opened but not used.
- `replay`: `src` ("lvis"), `id` (LVIS image id), `image`, `w`, `h`, `cats` = [[k, mask], ...]; the replay
  noun phrase `prompt_list()["replay"][k]` has prompt id 8 + k, `mask` is its union mask.
- `replay_neg`: `id`, `image`, `negs` (prompt ids of verified-absent categories: replay noun phrases, 5 for
  "mirror", 582 + j for `prompt_list()["negative_only"][j]`), `task` ([5] when "mirror" is among them, else []).
- `joint`: `kind` plus the item of its source stream. `both` = a `switch` item (both members present);
  `single` = a USC12K single-kind scene with `prompt` (its own member, 0 or 1), `presence` 1 and `gt` (own mask),
  the other member absent; `none` = a `neg_sal` item (both absent); `bg` = a COD10K-TR `neg_cam` item (the
  camouflaged member absent).

`downscale: true` marks images whose longer side exceeds 1008 in the dense streams and `switch` (7,938 items;
the `joint` "both" items are copies of `switch` items and carry it too). The loader resizes them before
augmentation: m = max(w, h), size = (max(1, round(w * 1008 / m)),
max(1, round(h * 1008 / m))) in this order of operations; image = `Image.open(p).convert("RGB").resize(size,
Image.BILINEAR)`; each GT = `(np.asarray(Image.open(g).convert("L").resize(size, Image.BILINEAR)) >= 128) * 255`,
resized straight to the image's size even when its aspect differs (the 18 rotated Trans10K masks). Items without
the flag are read at their native size, also when they are larger (negatives, `joint` single items, LVIS).

### Validation pairs (`catalog.val_pairs`)

Records `{id, image, prompts: [p], presence, kind}`, never trained on. `single` (4,243): 10% of every
negative pool held out by md5("<src>/<id>") % 10 == 0 (COD10K-TR, USC12K "none"), and the USC12K train
single-kind scenes that duplicate no test image, as positives of their own prompt. `joint` (4,628): `single`
without the single-kind scenes moved to joint training, plus the cross-concept negative of every remaining
single-kind scene (`kind` "usc12k_xcon") and both prompts of the "both" scenes held out from training by the same
md5 rule (`kind` "usc12k_both").

### Test sets (`catalog.test_records`)

Records `{id, image, gt, gt_mode}` (+ `gt_raw` for BSDS500), in file-name order. `catalog.load_gt` applies the mode:

| Mode | GT array | Sets |
| --- | --- | --- |
| `L` | the mask as 8-bit gray | DUTS-TE, ECSSD, HKU-IS, DUT-OMRON, CAMO, COD10K, NC4K, ISTD, MSD, UCF, CHAMELEON, WH-SYMMAX |
| `thr129` | (gray >= 129) x 255, more than half of PASCAL-S's 12 annotators | PASCAL-S |
| `thr128` | (gray >= 128) x 255 | GSD, PMD, BIPEDv2 |
| `any128` | (any of the first three channels >= 128) x 255 | SBU (graded masks), Trans10K (things and stuff) |
| `skel_mat` | (`.mat` symmetry map > 0) x 255 | SK-LARGE, SYM-PASCAL |
| `bsds_mat` | one Boundaries map per annotator (4 to 8), `catalog.load_bsds_gts` | BSDS500 |

COD10K counts its 2,026 camouflaged test images; Trans10K its easy (3,491) and hard (937) test images.

Rejection queries `{qid, set, source, k, cluster, id, image, prompt, w, h}` (+ `category_id`, `aq_qid` for LVIS):
COD10K test background x "camouflaged object" (784); USC12K test scenes without the concept (3,600: "none" x both
prompts, salient-only x "camouflaged object", camouflaged-only x "salient object"); 400 LVIS images x a category
verified absent. Retention pairs `{image_id, image, w, h, np, cid, mask, area, split}`: LVIS held-out images x
categories left out of the replay (`heldout`), or x replayed categories (`seen`).

## How the lists were made

The lists are fixed; `prepare_data.py` only fetches and converts files. They come from these rules:
- Official training splits, used unchanged.
- USC12K training scenes are typed by their GT content: "both" scenes are switching pairs, "none" scenes are
  negatives of both prompts, single-kind scenes are validation positives; half of them (md5 split) train the
  joint presence module.
- LVIS v1 val: 25% of the images and 30% of the eligible categories (fixed permutations, seed 0) are held out;
  replay = remaining images x remaining categories, union masks covering at least 1% of the image, categories
  whose names or definitions mention mirrors or glass blocked; replay images that duplicate a test image or a
  held-out LVIS image (16 x 16 thumbnail match) are dropped.
