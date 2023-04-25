from torchmetrics import Metric
import torch



class LogMetric(Metric):
    def __init__(self, init):
        super().__init__()
        self.add_state("val", default=init, dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, val, num):
        self.val += val
        self.total += num

    def compute(self):
        return self.val.float() / self.total