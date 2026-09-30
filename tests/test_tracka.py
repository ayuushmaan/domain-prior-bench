import numpy as np
import torch
from domainprior.models.nanotfm import build_model, count_params, prep_task, N_TR, N_TE
from domainprior.priors.mixture import sample_mixture


def test_small_is_about_2M():
    p = count_params(build_model("SMALL", 0))
    assert 1_000_000 < p < 4_000_000, p


def test_prep_task_exact_shapes():
    rng = np.random.default_rng(0)
    for _ in range(5):
        t = sample_mixture(rng, 0.5)
        Xtr, ytr, Xte, yte = prep_task(t.X_train, t.y_train, t.X_test, t.y_test, rng)
        assert Xtr.shape == (N_TR, Xtr.shape[1]) and Xte.shape == (N_TE, Xtr.shape[1])
        assert len(ytr) == N_TR and len(yte) == N_TE
        assert not np.isnan(Xtr).any() and not np.isnan(Xte).any()
        assert set(np.unique(ytr)) == {0, 1}
        assert Xtr.shape[1] <= 64


def test_forward_cpu():
    rng = np.random.default_rng(1)
    t = sample_mixture(rng, 0.0)
    Xtr, ytr, Xte, yte = prep_task(t.X_train, t.y_train, t.X_test, t.y_test, rng)
    m = build_model("SMALL", 0).eval()
    x = torch.from_numpy(np.vstack([Xtr, Xte])).unsqueeze(0)
    y = torch.from_numpy(np.concatenate([ytr, yte])).unsqueeze(0).float()
    with torch.no_grad():
        out = m((x, y[:, :N_TR]), train_test_split_index=N_TR)
    assert out.shape == (1, N_TE, 2)
