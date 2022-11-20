import torch
import torch.nn as nn
from mmcv.parallel import MODULE_WRAPPERS, MMDistributedDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from torch.cuda._utils import _get_device_index


@MODULE_WRAPPERS.register_module()
class DistributedDataParallelWrapper(nn.Module):
    

    def __init__(self,
                 module,
                 device_ids,
                 dim=0,
                 broadcast_buffers=False,
                 find_unused_parameters=False,
                 **kwargs):
        super(DistributedDataParallelWrapper, self).__init__()
        assert len(device_ids) == 1, (
            'Currently, DistributedDataParallelWrapper only supports one'
            'single CUDA device for each process.'
            f'The length of device_ids must be 1, but got {len(device_ids)}.')
        self.module = module
        self.dim = dim
        self.to_ddp(
            device_ids=device_ids,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            **kwargs)
        self.output_device = _get_device_index(device_ids[0], True)

    def to_ddp(self, device_ids, dim, broadcast_buffers,
               find_unused_parameters, **kwargs):
        """Wrap models with separate MMDistributedDataParallel.

        It only wraps the modules with parameters.
        """
        for name, module in self.module._modules.items():
            if next(module.parameters(), None) is None:
                module = module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                module = module.cuda()
            else:
                module = MMDistributedDataParallel(
                    module.cuda(),
                    device_ids=device_ids,
                    dim=dim,
                    broadcast_buffers=broadcast_buffers,
                    find_unused_parameters=find_unused_parameters,
                    **kwargs)
            self.module._modules[name] = module

    def scatter(self, inputs, kwargs, device_ids):
        """Scatter function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
            device_ids (int): Device id.
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        """Forward function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])

    def train_step(self, *inputs, **kwargs):
        """Train step function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output

    def val_step(self, *inputs, **kwargs):
        """Validation step function.

        Args:
            inputs (tuple): Input data.
            kwargs (dict): Args for ``scatter_kwargs``.
        """
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output
