import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from scdiff.utils.data import get_candidate_conditions


def test_get_candidate_conditions_grid(subtests):
    batch_enc, celltype_enc = LabelEncoder(), LabelEncoder()
    batch_enc.classes_ = np.array(["b1", "b2", "b3"])
    celltype_enc.classes_ = np.array(["ct1", "ct2"])

    with subtests.test(mode="grid"):
        cfg = DictConfig({"mode": "grid"})
        cond = get_candidate_conditions(cfg, batch_enc, celltype_enc)
        assert cond.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]

    with subtests.test(mode="select"):
        cfg = DictConfig({"mode": "select", "options": [["b1", "ct1"], ["b2", "ct2"]]})
        cond = get_candidate_conditions(cfg, batch_enc, celltype_enc)
        assert cond.tolist() == [[0, 0], [1, 1]]

    with subtests.test(mode="partialgrid"):
        cfg = DictConfig({"mode": "partialgrid", "options": {"cond0": ["b1", "b2"], "cond1": ["ct1"]}})
        cond = get_candidate_conditions(cfg, batch_enc, celltype_enc)
        assert cond.tolist() == [[0, 0], [1, 0]]

        cfg = DictConfig({"mode": "partialgrid", "options": {"cond1": ["ct1"]}})
        cond = get_candidate_conditions(cfg, batch_enc, celltype_enc)
        assert cond.tolist() == [[0, 0], [1, 0], [2, 0]]
