import torch as th


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32


class MultiHot(Transform):
    """Concatenated one-hots for MultiDiscrete actions.

    Input:  (..., n_dims)  int — one categorical index per dimension
    Output: (..., sum(nvec)) float — concatenated per-dim one-hots
    """
    def __init__(self, nvec):
        self.nvec = list(nvec)
        self.out_dim = sum(self.nvec)

    def transform(self, tensor):
        parts = []
        for i, n in enumerate(self.nvec):
            idx = tensor[..., i:i + 1].long()
            onehot = tensor.new_zeros(*tensor.shape[:-1], n).float()
            onehot.scatter_(-1, idx, 1)
            parts.append(onehot)
        return th.cat(parts, dim=-1)

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32