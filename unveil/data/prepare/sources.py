"""Where every dataset comes from and where it lands under the data root.

Each `Dataset` lists its archives (file name, URL, SHA-256, size, the folder under the data root it is unpacked
into, and the members kept), the roles it serves ("train", "test", "extra") and its terms. Archives are unpacked as
released, so the folders under the data root are the archives' own folders (README "Layout"). Google Drive files
are fetched through drive.usercontent.google.com; "mirror" marks a copy linked from a widely used repository when
the authors' link needs a login, a form or is gone (the README says which copy).
"""

from __future__ import annotations

from dataclasses import dataclass

GD = "https://drive.usercontent.google.com/download?export=download&confirm=t&id={}"
HF = "https://huggingface.co/datasets/{}/resolve/{}/{}"
SALIENCY = "https://saliencydetection.net/{}/download/{}"


@dataclass(frozen=True)
class Archive:
    file: str                       # expected file name (what --raw is searched for)
    url: str
    sha256: str
    size: int
    dest: str = "."                 # folder under the data root it is unpacked into
    keep: tuple[str, ...] = ()      # member prefixes kept (empty: all); __MACOSX and ._* are always skipped
    roles: tuple[str, ...] = ("train", "test", "extra")
    unpack: bool = True             # False: read by a converter (parquet), never unpacked


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    roles: tuple[str, ...]
    top: tuple[str, ...]            # folders it owns under the data root
    page: str
    terms: str
    archives: tuple[Archive, ...] = ()
    derived: tuple[str, ...] = ()   # folders under derived/ written by its converters
    note: str = ""


def _ds(key, title, roles, top, page, terms, archives, derived=(), note=""):
    return Dataset(key, title, tuple(roles), tuple(top), page, terms, tuple(archives), tuple(derived), note)


DATASETS = {d.key: d for d in (
    _ds("duts_tr", "DUTS-TR (salient objects, train)", ["train"], ["DUTS-TR"], "https://saliencydetection.net/duts/",
        "No licence; GT annotation: 'All rights reserved by the original authors of DUTS Image Dataset'.",
        [Archive("DUTS-TR.zip", SALIENCY.format("duts", "DUTS-TR.zip"),
                 "ce61e023c8f59d022b4d46981cf16813b83d089242e6489a45630d83962ea058", 270997309)]),
    _ds("duts_te", "DUTS-TE (salient objects, test)", ["test"], ["DUTS-TE"], "https://saliencydetection.net/duts/",
        "As DUTS-TR.",
        [Archive("DUTS-TE.zip", SALIENCY.format("duts", "DUTS-TE.zip"),
                 "d9905c33307e1d35c397604a701ed87f019acf060033076eec633f13a2eb8104", 139799089)]),
    _ds("ecssd", "ECSSD (salient objects, test)", ["test"], ["ECSSD"],
        "https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html", "No licence or terms stated.",
        [Archive("images.zip", "https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip",
                 "7d5dce1a21c6d82e2f9617c00ad7eae2de5ded9320e8c59a01b4c731d210ea47", 67766979, "ECSSD"),
         Archive("ground_truth_mask.zip",
                 "https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip",
                 "dc0a9f11f2adee2737d95b28f9567a0a5de33f78743447c097ceb4d42f281a31", 1571109, "ECSSD")]),
    _ds("hkuis", "HKU-IS (salient objects, test)", ["test"], ["HKU-IS"],
        "https://i.cs.hku.hk/~gbli/deep_saliency.html (mirror: Drive file linked from github.com/plemeri/InSPyReNet)",
        "No licence stated.",
        [Archive("HKU-IS.zip", GD.format("1H3szJYbr5_CRCzrYfhPHThTgszkKd1EU"),
                 "c2c15b80a57bd022b68257ed52074406ebf8d991334b6744416d752e9cd0f441", 944572407)],
        note="mirror: the authors' Drive link asks for a Google login"),
    _ds("dutomron", "DUT-OMRON (salient objects, test)", ["test"], ["DUT-OMRON"],
        "https://saliencydetection.net/dut-omron/",
        "No licence; GT annotation: 'All rights reserved by the original authors of DUT-OMRON Image Dataset'.",
        [Archive("DUT-OMRON-image.zip", SALIENCY.format("dut-omron", "DUT-OMRON-image.zip"),
                 "217ac4ed3893a318c28639a1212fd7086f71c56f70dcf0d033c36d97b001f789", 114809842, "DUT-OMRON"),
         Archive("DUT-OMRON-gt-pixelwise.zip.zip", SALIENCY.format("dut-omron", "DUT-OMRON-gt-pixelwise.zip.zip"),
                 "8307bb97db7ab3b2eb2ea8004404d67a15d9e4086079e150db6811563d2ec8e3", 6600590, "DUT-OMRON")]),
    _ds("pascals", "PASCAL-S (salient objects, test)", ["test"], ["PASCAL-S"],
        "https://ccvl.jhu.edu/datasets/ (mirror: Drive file linked from github.com/plemeri/InSPyReNet)",
        "No licence stated; images are a PASCAL VOC subset.",
        [Archive("PASCAL-S.zip", GD.format("1h0IE2DlUt0HHZcvzMV5FCxtZqQqh9Ztf"),
                 "5d22955df6448bcb864d318161c6da9279c6bfd9676a862915a994de60bae861", 102894015)],
        note="mirror: the authors' server is unreachable"),
    _ds("camo", "CAMO (camouflaged objects, train 1,000 + test 250)", ["train", "test"], ["CAMO-V.1.0-CVIU2019"],
        "https://sites.google.com/view/ltnghia/research/camo",
        "CC BY-NC-SA 4.0 ('You must not use this work for commercial purposes').",
        [Archive("CAMO-V.1.0.zip", GD.format("1lLDZwQ0JiUM9FxTPGUGNQJhzBEkgm7x4"),
                 "510eaeef5bbcb626016b1a13f61e269002f60b8b05234cd7d51a9a7399410cc5", 286249800)]),
    _ds("cod10k", "COD10K-v3 (camouflaged objects, train + test)", ["train", "test"], ["COD10K-v3"],
        "https://dengpingfan.github.io/pages/COD.html",
        "Non-commercial only (COD10K-v3/Readme.txt: 'Our dataset is free for non-commercial usage').",
        [Archive("COD10K-v3.zip", GD.format("1vRYAie0JcNStcSwagmCq55eirGyMYGm5"),
                 "3bd100640dde83b05f9af2932ae633d4894c899092fcbda712f7ff447a97b1e4", 2416808676)]),
    _ds("nc4k", "NC4K (camouflaged objects, test)", ["test"], ["NC4K"],
        "https://github.com/JingZhang617/COD-Rank-Localize-and-Segment", "No licence stated.",
        [Archive("NC4K.zip", GD.format("1kzpX_U3gbgO9MuwZIWTuRVpiB7V6yrAQ"),
                 "1e9c446fade4647c38952b496eed355686b90c408bbf17474368b1d8b4fd5695", 600648391, "NC4K")]),
    _ds("sbu", "SBU (shadows, train 4,085 + test 638)", ["train", "test"], ["SBU-shadow"],
        "https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html",
        "Non-commercial (SBU-shadow/README: 'Feel free to use this dataset for non-commercial purposes').",
        [Archive("SBU-shadow.zip", "https://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip",
                 "8ac703bc4ec0cf3f57e128a6e7a8ed73680d9f68e4e22d08fc83a3f60c612109", 299916461)],
        derived=["sbu_gt"]),
    _ds("istd", "ISTD (shadows, train 1,330 + test 540)", ["train", "test"], ["ISTD_Dataset"],
        "https://github.com/DeepInsight-PCALab/ST-CGAN",
        "Research and non-commercial only (ST-CGAN README); copyright 2018 PCALab, NJUST.",
        [Archive("ISTD_Dataset.rar", GD.format("1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY"),
                 "46debc393465044f25d682c2d500cee2c0a9063253040c35b4a10f79b90bc389", 2139405436)],
        note="RAR archive: needs bsdtar (libarchive), unrar or 7z"),
    _ds("trans10k", "Trans10K v1 (transparent objects, train 5,000 + test 4,428)", ["train", "test"], ["Trans10K"],
        "https://xieenze.github.io/projects/TransLAB/TransLAB.html",
        "No dataset licence stated; the code repository is Apache-2.0 for academic use, commercial use by request.",
        [Archive("train.zip", GD.format("1ZNTTbGjvPUHuLj3r3wEg8m12et7tDluz"),
                 "5c388786af167c640f472c455a7fd1851d591632f1d419873cf4a8070eb99869", 4533524305, "Trans10K",
                 roles=("train",)),
         Archive("test.zip", GD.format("1A1mLgpOjioSQ1pzSUkU0Ce_3W_YozqgY"),
                 "514d3788091058e41ad226a9d3a99ca29523150403f9e5a1efc99b23e1bd88bb", 3997412881, "Trans10K",
                 roles=("test",))],
        derived=["trans10k_gt"], note="v1, which has the same images as v2"),
    _ds("msd", "MSD (mirrors, train 3,063 + test 955)", ["train", "test"], ["MSD"],
        "https://mhaiyang.github.io/ICCV2019_MirrorNet/ (mirror: Drive file linked from "
        "github.com/Charmve/Mirror-Glass-Detection)",
        "No licence stated; official access is by application.",
        [Archive("MSD.zip", GD.format("1Znw92fO6lCKfXejjSSyMyL1qtFepgjPI"),
                 "bdea8d846d0aff30c2021b085c197459d34e105d61e1165a4e03473774ab2843", 167173069)],
        note="mirror"),
    _ds("bsds500", "BSDS500 (edges, test 200)", ["test"], ["BSR"],
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html",
        "No licence stated; cite Arbelaez et al., TPAMI 2011.",
        [Archive("BSR_bsds500.tgz",
                 "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
                 "97e49d31764f3912f0c4122707d53062ac9e783ba0f095e447a4d53c1a41af8e", 70763455,
                 keep=("BSR/BSDS500/data/",))]),
    _ds("hed_bsds", "HED-BSDS (edges, train 28,800 augmented)", ["train"], ["HED-BSDS"],
        "https://github.com/s9xie/hed", "No licence stated for the data; the README asks to cite HED.",
        [Archive("HED-BSDS.tar", "https://vcl.ucsd.edu/hed/HED-BSDS.tar",
                 "0db652f768291e316281ae41a5d0b7cde1145ddd3d3c4b959ae87820b5146a40", 1303992320)]),
    _ds("sklarge", "SK-LARGE (skeletons, train 746 + test 745)", ["train", "test"], ["SK-LARGE"],
        "http://kaiz.xyz/sk-large (down; mirror: Drive file linked from github.com/YukangWang/DeepFlux)",
        "No dataset licence stated; images are MS-COCO crops.",
        [Archive("SK-LARGE.zip", GD.format("1eBIjpzU0kttcKEesRJ2y29_yMcc03sbh"),
                 "f298e1e6ab3b35dc07437c77d1bd553c34ad3d70282a98632336bcb22fe7b41f", 32259568)],
        derived=["sklarge_gt"], note="mirror"),
    _ds("sympascal", "SYM-PASCAL (symmetry axes, train 647 + test 788)", ["train", "test"], ["SymPASCAL-by-KZ"],
        "https://github.com/KevinKecc/SRN (OneDrive needs a login; mirror: Drive file linked from DeepFlux)",
        "No dataset licence stated; images are from PASCAL VOC 2011.",
        [Archive("SymPASCAL-by-KZ.zip", GD.format("1PSfksp7X9fhF0xZ9jOaMb0f4eUs8ED9j"),
                 "ae9ab56415f1e15d8e8ffe58d3dcba8a98933190bcaf1115cae90331fb7fbd00", 77458583)],
        derived=["sympascal_gt"], note="mirror"),
    _ds("usc12k", "USC12K (salient / camouflaged scenes, train 8,400 + test 3,600)", ["train", "test"],
        ["VOC-USC12K"], "https://github.com/ssecv/USCNet",
        "No dataset licence stated (the README shows an MIT badge; the repository has no LICENSE file).",
        [Archive("VOC-USC12K.zip", GD.format("1MIVCH7sLOzFwrzEDjKs2PSba7UpzLG7I"),
                 "41e0573439aa42a565a16679ba2aaf0f8f042a1ecb28d26902285acf4e1a067c", 2310013741)],
        derived=["usc12k_gt_train"]),
    _ds("lvis", "LVIS v1 val (concept replay, presence negatives, retention, rejection)", ["train", "test"],
        ["lvis", "coco"], "https://www.lvisdataset.org/dataset (images: https://cocodataset.org)",
        "LVIS annotations: CC BY 4.0; COCO images: Flickr terms of use (see cocodataset.org/#termsofuse).",
        [Archive("lvis_v1_val.json.zip", "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
                 "5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5", 64026968, "lvis")],
        derived=["lvis_masks"],
        note="images: only the COCO 2017 images the manifests use, taken from train2017.zip / val2017.zip "
             "in --raw or downloaded one by one from images.cocodataset.org"),
    _ds("ucf", "UCF (shadows, test 110)", ["extra"], ["UCF"],
        "https://github.com/eraserNut/MTMT (Drive link in the README)", "No licence stated.",
        [Archive("UCF.zip", GD.format("12DOmMVmE-oNuJVXmkBJrkfBvuDd0O70N"),
                 "05ed1d5edeb9b6b3511a98f184abd750f725e4f4c4e98595edfcc196a22b77b7", 16577419)]),
    _ds("gsd", "GSD (glass, test 813)", ["extra"], ["GSD"],
        "https://jiaying.link/cvpr2021-gsd/ (the authors' Hugging Face upload garrying/GSD)", "CC BY-NC 4.0.",
        [Archive("GSD-test-00000-of-00001.parquet",
                 HF.format("garrying/GSD", "f8897574bbc2632a440efa1de021fd2b02f89f51", "data/test-00000-of-00001.parquet"),
                 "c1ea301305f04d9d05ef09dbf2982e4ce188fb8d779c85095017c54c3a688963", 33686132, unpack=False)],
        note="parquet: needs pyarrow; images and masks are written with their original bytes"),
    _ds("pmd", "PMD (mirrors, test 571)", ["extra"], ["PMD"],
        "https://jiaying.link/cvpr2020-pgd/ (the authors' Hugging Face upload garrying/PMD)", "CC BY-NC 4.0.",
        [Archive("PMD-test-00000-of-00001.parquet",
                 HF.format("garrying/PMD", "73c8ae81846070f410dd16399ff629104173100d", "data/test-00000-of-00001.parquet"),
                 "59381ad0a10f79b72f56bae4f37437682bd366d0c9410d8eff2fc1683d5f9ad8", 169600933, unpack=False)],
        note="parquet: needs pyarrow; images and masks are written with their original bytes"),
    _ds("chameleon", "CHAMELEON (camouflaged objects, test 76)", ["extra"], ["COD-TestDataset"],
        "https://github.com/GewelsJI/SINet-V2 (testing-set Drive link in the README)",
        "No licence stated; images from the web.",
        [Archive("COD-TestDataset.zip", GD.format("1V0iSEdYJrT0Y_DHZfVGMg6TySFRNTy4o"),
                 "5d1a4244d513f78682b42cb89b759d44cc462ce9d3ae78d18974b2790e3aa310", 1310372023,
                 keep=("COD-TestDataset/CHAMELEON/",))]),
    _ds("biped2", "BIPEDv2 (edges, test 50)", ["extra"], ["BIPEDv2"],
        "https://github.com/xavysp/MBIPED (Drive folder in the README)", "MIT per the MBIPED repository.",
        [Archive("BIPEDv2.zip", GD.format("1vvts9m82PpgMz-nRPPI800DuE9svOfsY"),
                 "5fdcd9ed7e8d0c6614858a63d80f1bef523e6c550ac0133299bc962648dab162", 35139513)]),
    _ds("whsymmax", "WH-SYMMAX (skeletons, test 100)", ["extra"], ["WH-SYMMAX"],
        "https://github.com/cong-yang/BlumNet (Drive link in the README)",
        "No dataset licence stated; images are Weizmann Horses.",
        [Archive("WH-SYMMAX.tar.xz", GD.format("1tMRuLLMm_T1qokkjXz_moocTPZAfeXcr"),
                 "7bcf79479f0d4633b9f06122eae17990c6dfe55f244960ae53dfef28d24541ba", 5347136)]),
)}

ROLES = ("train", "test", "extra")
COCO_URL = "http://images.cocodataset.org/{}"           # <split>/<file>, the LVIS json's coco_url
COCO_ZIPS = {"train2017": "train2017.zip", "val2017": "val2017.zip"}   # http://images.cocodataset.org/zips/<file>


def dataset_of(rel: str) -> str:
    """Dataset key owning a path relative to the data root (its top folder, or derived/<folder>)."""
    parts = rel.split("/")
    if parts[0] == "derived":
        for d in DATASETS.values():
            if parts[1] in d.derived:
                return d.key
    else:
        for d in DATASETS.values():
            if parts[0] in d.top:
                return d.key
    raise KeyError(f"no dataset owns {rel}")


def select(names) -> list[str]:
    """Dataset keys from names: keys, roles ("train", "test", "extra") or "all"."""
    out = []
    for n in names:
        if n == "all":
            out += list(DATASETS)
        elif n in ROLES:
            out += [k for k, d in DATASETS.items() if n in d.roles]
        elif n in DATASETS:
            out.append(n)
        else:
            raise KeyError(f"unknown dataset {n!r}; use a key ({', '.join(DATASETS)}), a role or 'all'")
    return list(dict.fromkeys(out))
