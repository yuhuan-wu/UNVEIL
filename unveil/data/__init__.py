"""Datasets of UNVEIL: the catalog (`unveil.data.catalog`) and the preparation of the data root
(`unveil.data.prepare`, run by `scripts/prepare_data.py`)."""

from unveil.data.catalog import (AUX_SETS, TASK_PROMPTS, TEST_SETS, TRAIN_STREAMS, aux_records, calibration_records,
                                 data_root, edge_gts, gt_hw, load_bsds_gts, load_gt, prompt_list, rejection_records,
                                 retention_records, test_records, train_index, val_pairs)

__all__ = ["AUX_SETS", "TASK_PROMPTS", "TEST_SETS", "TRAIN_STREAMS", "aux_records", "calibration_records",
           "data_root", "edge_gts", "gt_hw", "load_bsds_gts", "load_gt", "prompt_list", "rejection_records",
           "retention_records", "test_records", "train_index", "val_pairs"]
