import time
import datetime


class EarlyStopMonitor:

  def __init__(self, patience):
    self.patience = patience
    self.cnt = 0
    self.cur_best = float('inf')

  def update(self, loss):
    """

    :param loss:
    :return:
        return True if patience exceeded
    """
    if loss < self.cur_best:
      self.cnt = 0
      self.cur_best = loss
    else:
      self.cnt += 1

    if self.cnt >= self.patience:
      return True
    else:
      return False

  def reset(self):
    self.cnt = 0
    self.cur_best = float('inf')


def iterline(fpath):
  with open(fpath) as f:

    for line in f:

      line = line.strip()
      if line == '':
        continue

      yield line


def get_time_stamp():
  return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def get_lr(optimizer):
  return optimizer.state_dict()['param_groups'][0]['lr']


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
