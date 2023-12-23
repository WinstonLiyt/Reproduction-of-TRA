import copy
import torch
import numpy as np
import pandas as pd

from qlib.data.dataset import DatasetH

device = "cuda" if torch.cuda.is_available() else "cpu"


# 将数据转换为torch.Tensor类型
def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)
    return x


# 根据pandas索引创建时间序列切片
def _create_ts_slices(index, seq_len):
    # 确保索引是单调递增的
    assert index.is_monotonic_increasing

    # 按代码统计每个日期的样本数
    sample_count_by_codes = pd.Series(0, index=index).groupby(level=0).size().values

    # 计算每个代码的起始索引
    start_index_of_codes = np.roll(np.cumsum(sample_count_by_codes), 1)
    start_index_of_codes[0] = 0

    # 创建所有[start, stop)的索引，这些特征用于预测“stop - 1”处的标签
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_codes, sample_count_by_codes):
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices)

    return slices


# 获取日期解析函数
def _get_date_parse_fn(target):
    """
    Example:
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    """
    if isinstance(target, pd.Timestamp):
        _fn = lambda x: pd.Timestamp(x)  # Timestamp('2020-01-01')
    elif isinstance(target, str) and len(target) == 8:
        _fn = lambda x: str(x).replace("-", "")[:8]  # '20200201'
    elif isinstance(target, int):
        _fn = lambda x: int(str(x).replace("-", "")[:8])  # 20200201
    else:
        _fn = lambda x: x
    return _fn


# 带有记忆增强的时间序列数据集类
class MTSDatasetH(DatasetH):
    """
    Args:
        handler (DataHandler): 数据处理器
        segments (dict): 数据分段
        seq_len (int): 时间序列长度
        horizon (int): 标签视野（用于TRA的历史损失掩码）
        num_states (int): 要添加的记忆状态数量（用于TRA）
        batch_size (int): 批量大小（<0表示每日批量）
        shuffle (bool): 是否打乱数据
        pin_memory (bool): 是否将数据固定到GPU内存
        drop_last (bool): 是否丢弃小于批量大小的最后一批
    """

    def __init__(
        self,
        handler,
        segments,
        seq_len=20,
        horizon=10,
        num_states=1,
        batch_size=-1,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        **kwargs,
    ):
        # 确保'horizon'参数被指定，以避免数据泄露
        assert horizon > 0, "please specify `horizon` to avoid data leakage"

        # 初始化类实例变量
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_states = num_states
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.params = (batch_size, drop_last, shuffle)

        super().__init__(handler, segments, **kwargs)

    # 数据准备函数
    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        super().setup_data()

        # 将索引更改为<code, date>格式，并就地排序以减少内存使用
        df = self.handler._data
        df.index = df.index.swaplevel()
        df.sort_index(inplace=True)

        self._data = df["feature"].values.astype("float32")
        self._label = df["label"].squeeze().astype("float32")
        self._index = df.index

        # 添加记忆单元到特征中，用于TRA模型的状态记录
        self._data = np.c_[
            self._data, np.zeros((len(self._data), self.num_states), dtype=np.float32)
        ]

        # 创建一个零填充张量，用于填充不足序列长度的数据
        self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32)

        # 如果启用了pin_memory，则将数据转移到GPU上
        if self.pin_memory:
            self._data = _to_tensor(self._data)
            self._label = _to_tensor(self._label)
            self.zeros = _to_tensor(self.zeros)

        # 根据索引创建时间序列切片
        self.batch_slices = _create_ts_slices(self._index, self.seq_len)

        # 创建按日切片的索引，用于每日批量处
        index = [slc.stop - 1 for slc in self.batch_slices]
        act_index = self.restore_index(index)
        daily_slices = {date: [] for date in sorted(act_index.unique(level=1))}
        for i, (code, date) in enumerate(act_index):
            daily_slices[date].append(self.batch_slices[i])
        self.daily_slices = list(daily_slices.values())

    # 根据给定的时间范围准备数据段
    def _prepare_seg(self, slc, **kwargs):
        # 根据索引中的日期格式获取日期解析函数
        fn = _get_date_parse_fn(self._index[0][1])

        # 解析切片或元组形式的日期范围
        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            start, stop = slc
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # 将日期字符串转换为相应的日期格式
        start_date = fn(start)
        end_date = fn(stop)

        # 创建当前对象的浅拷贝
        obj = copy.copy(self)
        # 手动赋值数据，因为原始数据不会被自动复制
        obj._data = self._data
        obj._label = self._label
        obj._index = self._index
        new_batch_slices = []

        # 筛选在指定日期范围内的时间序列切片
        for batch_slc in self.batch_slices:
            date = self._index[batch_slc.stop - 1][1]
            if start_date <= date <= end_date:
                new_batch_slices.append(batch_slc)
        obj.batch_slices = np.array(new_batch_slices)

        # 创建每日切片的索引
        new_daily_slices = []
        for daily_slc in self.daily_slices:
            date = self._index[daily_slc[0].stop - 1][1]
            if start_date <= date <= end_date:
                new_daily_slices.append(daily_slc)
        obj.daily_slices = new_daily_slices
        return obj

    # 根据索引恢复原始数据的索引
    def restore_index(self, index):
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        return self._index[index]

    # 给特定的索引赋值数据
    def assign_data(self, index, vals):
        if isinstance(self._data, torch.Tensor):
            vals = _to_tensor(vals)
        elif isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
            index = index.detach().cpu().numpy()
        self._data[index, -self.num_states :] = vals

    # 清除记忆状态
    def clear_memory(self):
        self._data[:, -self.num_states :] = 0

    # TODO: better train/eval mode design
    def train(self):
        """启用训练模式"""
        self.batch_size, self.drop_last, self.shuffle = self.params

    def eval(self):
        """启用评估模式"""
        self.batch_size = -1
        self.drop_last = False
        self.shuffle = False

    # 获取切片列表和批量大小
    def _get_slices(self):
        if self.batch_size < 0:
            slices = self.daily_slices.copy()
            batch_size = -1 * self.batch_size
        else:
            slices = self.batch_slices.copy()
            batch_size = self.batch_size
        return slices, batch_size

    # 获取迭代器长度
    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    # 迭代器，用于按批次生成数据
    def __iter__(self):
        slices, batch_size = self._get_slices()
        if self.shuffle:
            np.random.shuffle(slices)

        for i in range(len(slices))[::batch_size]:
            if self.drop_last and i + batch_size > len(slices):
                break
            # get slices for this batch
            slices_subset = slices[i : i + batch_size]
            if self.batch_size < 0:
                slices_subset = np.concatenate(slices_subset)
            # collect data
            data = []
            label = []
            index = []
            for slc in slices_subset:
                _data = (
                    self._data[slc].clone()
                    if self.pin_memory
                    else self._data[slc].copy()
                )
                if len(_data) != self.seq_len:
                    if self.pin_memory:
                        _data = torch.cat(
                            [self.zeros[: self.seq_len - len(_data)], _data], axis=0
                        )
                    else:
                        _data = np.concatenate(
                            [self.zeros[: self.seq_len - len(_data)], _data], axis=0
                        )
                if self.num_states > 0:
                    _data[-self.horizon :, -self.num_states :] = 0
                data.append(_data)
                label.append(self._label[slc.stop - 1])
                index.append(slc.stop - 1)

            # concate
            index = torch.tensor(index, device=device)
            if isinstance(data[0], torch.Tensor):
                data = torch.stack(data)
                label = torch.stack(label)
            else:
                data = _to_tensor(np.stack(data))
                label = _to_tensor(np.stack(label))
            # yield -> generator

            yield {"data": data, "label": label, "index": index}
