from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import ProxyAnchorLoss

from ..engine import Engine
import torch


class ProxyAwareEngine(Engine):
    r"""Proxy Aware loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ProxyAwareSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        margin=0.1,
        alpha=32,
        logger=None
    ):
        super(ProxyAwareEngine, self).__init__(datamanager, use_gpu, logger)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = use_gpu
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = ProxyAnchorLoss(
            num_classes=self.datamanager.num_train_pids,
            embedding_dim=self.model.feature_dim,
            use_gpu=self.use_gpu,
            margin=margin,
            alpha=alpha
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs, pids)
        loss = self.compute_loss(self.criterion, features, pids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }
        torch.nn.utils.clip_grad_value_(self.criterion.parameters(), 10)

        return loss_summary
