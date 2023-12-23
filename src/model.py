import os
import copy
import math
import json
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from qlib.model.base import Model

device = "cuda" if torch.cuda.is_available() else "cpu"


class TRAModel(Model):
    def __init__(
        self,
        model_config,
        tra_config,
        model_type="LSTM",
        lr=1e-3,
        n_epochs=500,
        early_stop=50,
        smooth_steps=5,
        max_steps_per_epoch=None,
        freeze_model=False,
        model_init_state=None,
        lamb=0.0,
        rho=0.99,
        seed=None,
        logdir=None,
        eval_train=True,
        eval_test=False,
        avg_params=True,
        **kwargs,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.logger = get_module_logger("TRA")
        self.logger.info("TRA Model...")

        # 根据提供的模型类型和配置初始化模型
        self.model = eval(model_type)(**model_config).to(device)
        # 如果提供了模型的初始状态，则加载这个状态
        if model_init_state:
            self.model.load_state_dict(
                torch.load(model_init_state, map_location="cpu")["model"]
            )
        # 根据参数确定是否冻结模型的参数
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)
        else:  # 计算模型参数的总数并记录
            self.logger.info(
                "# model params: %d" % sum([p.numel() for p in self.model.parameters()])
            )

        # 初始化TRA模块并将其移动到相应的设备上
        self.tra = TRA(self.model.output_size, **tra_config).to(device)
        # 计算TRA模块参数的总数并记录
        self.logger.info(
            "# tra params: %d" % sum([p.numel() for p in self.tra.parameters()])
        )

        # 初始化优化器
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.tra.parameters()), lr=lr
        )

        # 将传入的参数保存为类属性
        self.model_config = model_config
        self.tra_config = tra_config
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.smooth_steps = smooth_steps
        self.max_steps_per_epoch = max_steps_per_epoch
        self.lamb = lamb
        self.rho = rho
        self.seed = seed
        self.logdir = logdir
        self.eval_train = eval_train
        self.eval_test = eval_test
        self.avg_params = avg_params

        # 如果使用TRA的状态数大于1，并且不在训练集上进行评估，则发出警告
        if self.tra.num_states > 1 and not self.eval_train:
            self.logger.warn("`eval_train` will be ignored when using TRA")

        # 如果提供了日志目录，则确保目录存在
        if self.logdir is not None:
            if os.path.exists(self.logdir):
                self.logger.warn(f"logdir {self.logdir} is not empty")
            os.makedirs(self.logdir, exist_ok=True)

        self.fitted = False
        self.global_step = -1

    def train_epoch(self, data_set):
        # 设置模型和TRA模块为训练模式
        self.model.train()
        self.tra.train()

        # 设置数据集为训练模式
        data_set.train()

        # 确定每个周期内的最大步数
        max_steps = self.n_epochs
        if self.max_steps_per_epoch is not None:
            max_steps = min(self.max_steps_per_epoch, self.n_epochs)

        # 初始化一些用于追踪训练进度的变量
        count = 0
        total_loss = 0
        total_count = 0

        # 进行批次训练
        for batch in tqdm(data_set, total=max_steps):
            count += 1
            if count > max_steps:
                break

            self.global_step += 1

            # 从批次中提取数据、标签和索引
            data, label, index = batch["data"], batch["label"], batch["index"]

            # 分离特征和历史损失
            feature = data[:, :, : -self.tra.num_states]
            hist_loss = data[:, : -data_set.horizon, -self.tra.num_states :]

            # 使用模型对特征进行预测
            hidden = self.model(feature)
            # 使用TRA模块进一步处理模型输出
            pred, all_preds, prob = self.tra(hidden, hist_loss)

            # 计算损失
            loss = (pred - label).pow(2).mean()

            # 计算所有预测结果的损失，并进行归一化
            L = (all_preds.detach() - label[:, None]).pow(2)
            L -= L.min(dim=-1, keepdim=True).values

            # 将损失保存到数据集的内存中
            data_set.assign_data(index, L)

            # 如果存在概率值，则使用sinkhorn算法计算样本分配矩阵，并进行正则化
            if prob is not None:
                P = sinkhorn(-L, epsilon=0.01)
                lamb = self.lamb * (self.rho**self.global_step)
                reg = prob.log().mul(P).sum(dim=-1).mean()
                loss = loss - lamb * reg

            # 反向传播、更新优化器、梯度置零
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 累计损失和样本数
            total_loss += loss.item()
            total_count += len(pred)

        # 计算平均损失
        total_loss /= total_count

        return total_loss

    def test_epoch(self, data_set, return_pred=False):
        # 设置模型为评估模式
        self.model.eval()
        self.tra.eval()
        # 设置数据集为评估模式
        data_set.eval()

        # 初始化预测结果和评价指标的列表
        preds = []
        metrics = []

        # 对数据集中的每个批次进行迭代
        for batch in tqdm(data_set):
            # 从批次中提取数据、标签和索引
            data, label, index = batch["data"], batch["label"], batch["index"]

            # 分离特征和历史损失
            feature = data[:, :, : -self.tra.num_states]
            hist_loss = data[:, : -data_set.horizon, -self.tra.num_states :]

            with torch.no_grad():
                hidden = self.model(feature)
                pred, all_preds, prob = self.tra(hidden, hist_loss)

            L = (all_preds - label[:, None]).pow(2)

            L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

            data_set.assign_data(index, L)  # save loss to memory

            # 准备保存预测结果和标签的数组
            X = np.c_[
                pred.cpu().numpy(),
                label.cpu().numpy(),
            ]
            columns = ["score", "label"]
            if prob is not None:
                X = np.c_[X, all_preds.cpu().numpy(), prob.cpu().numpy()]
                columns += ["score_%d" % d for d in range(all_preds.shape[1])] + [
                    "prob_%d" % d for d in range(all_preds.shape[1])
                ]

            pred = pd.DataFrame(X, index=index.cpu().numpy(), columns=columns)

            metrics.append(evaluate(pred))

            if return_pred:
                preds.append(pred)

        # 汇总所有批次的评价指标
        metrics = pd.DataFrame(metrics)
        metrics = {
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
            "IC": metrics.IC.mean(),
            "ICIR": metrics.IC.mean() / metrics.IC.std(),
        }

        # 如果需要返回预测结果，则合并所有批次的预测
        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)
        print("preds")
        print(preds)
        return metrics, preds

    # 模型拟合函数，用于训练和评估模型
    def fit(self, dataset, evals_result=dict()):
        # 准备训练、验证和测试数据集
        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])

        # 初始化最佳评分、最佳轮次、停止轮次以及最佳参数
        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {
            "model": copy.deepcopy(self.model.state_dict()),
            "tra": copy.deepcopy(self.tra.state_dict()),
        }
        params_list = {
            "model": collections.deque(maxlen=self.smooth_steps),
            "tra": collections.deque(maxlen=self.smooth_steps),
        }
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        # 标记模型已开始拟合
        self.fitted = True
        self.global_step = -1

        # 如果TRA的状态数量大于1，则初始化内存
        if self.tra.num_states > 1:
            self.logger.info("init memory...")
            self.test_epoch(train_set)

        # 对每个训练周期进行迭代
        for epoch in range(self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            # 训练
            self.logger.info("training...")
            self.train_epoch(train_set)

            # 评估
            self.logger.info("evaluating...")
            # 对模型参数进行平均，用于推理
            params_list["model"].append(copy.deepcopy(self.model.state_dict()))
            params_list["tra"].append(copy.deepcopy(self.tra.state_dict()))
            self.model.load_state_dict(average_params(params_list["model"]))
            self.tra.load_state_dict(average_params(params_list["tra"]))

            # NOTE: during evaluating, the whole memory will be refreshed
            if self.tra.num_states > 1 or self.eval_train:
                train_set.clear_memory()  # NOTE: clear the shared memory
                train_metrics = self.test_epoch(train_set)[0]
                evals_result["train"].append(train_metrics)
                self.logger.info("\ttrain metrics: %s" % train_metrics)

            # 在验证集上评估模型
            valid_metrics = self.test_epoch(valid_set)[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("\tvalid metrics: %s" % valid_metrics)

            # 如果需要，在测试集上评估模型
            if self.eval_test:
                test_metrics = self.test_epoch(test_set)[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("\ttest metrics: %s" % test_metrics)

            # 如果当前验证集的评分是最佳的，则更新最佳评分和最佳参数
            if valid_metrics["IC"] > best_score:
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                    "tra": copy.deepcopy(self.tra.state_dict()),
                }
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

            # 恢复到上一个周期的参数
            self.model.load_state_dict(params_list["model"][-1])
            self.tra.load_state_dict(params_list["tra"][-1])

        # 记录最佳评分和对应的周期
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        # 加载最佳参数
        self.model.load_state_dict(best_params["model"])
        self.tra.load_state_dict(best_params["tra"])

        # 在测试集上评估模型，并返回评估指标和预测结果
        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)

        # 如果提供了日志目录，则保存模型、预测和配置信息
        if self.logdir:
            self.logger.info("save model & pred to local directory")

            pd.concat(
                {name: pd.DataFrame(evals_result[name]) for name in evals_result},
                axis=1,
            ).to_csv(self.logdir + "/logs.csv", index=False)

            torch.save(best_params, self.logdir + "/model.bin")

            preds.to_pickle(self.logdir + "/pred.pkl")

            info = {
                "config": {
                    "model_config": self.model_config,
                    "tra_config": self.tra_config,
                    "lr": self.lr,
                    "n_epochs": self.n_epochs,
                    "early_stop": self.early_stop,
                    "smooth_steps": self.smooth_steps,
                    "max_steps_per_epoch": self.max_steps_per_epoch,
                    "lamb": self.lamb,
                    "rho": self.rho,
                    "seed": self.seed,
                    "logdir": self.logdir,
                },
                "best_eval_metric": -best_score,  # NOTE: minux -1 for minimize
                "metric": metrics,
            }
            with open(self.logdir + "/info.json", "w") as f:
                json.dump(info, f)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_set = dataset.prepare(segment)

        metrics, preds = self.test_epoch(test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)
        return preds


class LSTM(nn.Module):

    """LSTM Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of hidden layers
        use_attn (bool): whether use attention layer.
            we use concat attention as https://github.com/fulifeng/Adv-ALSTM/
        dropout (float): dropout rate
        input_drop (float): input dropout for data augmentation
        noise_level (float): add gaussian noise to input for data augmentation
    """

    def __init__(
        self,
        input_size=14,
        hidden_size=64,
        num_layers=2,
        use_attn=True,
        dropout=0.0,
        input_drop=0.0,
        noise_level=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attn = use_attn
        self.noise_level = noise_level

        self.input_drop = nn.Dropout(input_drop)

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size

    def forward(self, x):
        x = self.input_drop(x)
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x).to(x)
            x = x + noise * self.noise_level

        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1]

        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1).squeeze()
            last_out = torch.cat([last_out, att_out], dim=1)
        return last_out


class CNN(nn.Module):
    def __init__(
        self,
        input_size=14,
        hidden_size=140,
        dropout=0.0,
        input_drop=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = hidden_size

        self.input_drop = nn.Dropout(input_drop)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
        )
        self.layer3 = nn.Sequential(nn.Conv2d(8, 32, kernel_size=1), nn.LeakyReLU(0.3))
        self.layer4 = nn.Sequential(nn.Conv2d(32, 2, kernel_size=1), nn.LeakyReLU(0.3))
        self.pl = nn.AvgPool2d((2, 2))
        # self.sm = nn.Softmax()

    def forward(self, x):
        # print(x.size())
        x = self.input_drop(x)
        x = x.unsqueeze(1)  # 在第一维增加

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pl(out)
        out = out.view(out.size(0), -1)
        # out = self.sm(out)
        return out


class PositionalEncoding(nn.Module):
    # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):

    """Transformer Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of transformer layers
        num_heads (int): number of heads in transformer
        dropout (float): dropout rate
        input_drop (float): input dropout for data augmentation
        noise_level (float): add gaussian noise to input for data augmentation
    """

    def __init__(
        self,
        input_size=13,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        input_drop=0.0,
        noise_level=0.0,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.noise_level = noise_level

        self.input_drop = nn.Dropout(input_drop)

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.pe = PositionalEncoding(input_size, dropout)
        layer = nn.TransformerEncoderLayer(
            nhead=num_heads,
            dropout=dropout,
            d_model=hidden_size,
            dim_feedforward=hidden_size * 4,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output_size = hidden_size

    def forward(self, x):
        x = self.input_drop(x)

        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x).to(x)
            x = x + noise * self.noise_level

        x = x.permute(1, 0, 2).contiguous()  # the first dim need to be sequence
        x = self.pe(x)

        x = self.input_proj(x)
        out = self.encoder(x)

        return out[-1]


class TRA(nn.Module):

    """
    TRA结合历史预测误差和潜在表示作为输入，然后将输入样本路由到特定预测器进行训练和推断。

    Args:
        input_size (int): 输入大小（RNN/Transformer的隐藏层大小）
        num_states (int): 潜在状态（即交易模式）的数量
                         如果`num_states=1`，则TRA退化为传统方法
        hidden_size (int): 路由器的隐藏层大小
        tau (float): Gumbel Softmax的温度参数
    """

    def __init__(
        self, input_size, num_states=1, hidden_size=8, tau=1.0, src_info="LR_TPE"
    ):
        super().__init__()

        self.num_states = num_states
        self.tau = tau
        self.src_info = src_info

        # 当状态数大于1时，初始化路由器和全连接层
        if num_states > 1:
            self.router = nn.LSTM(
                input_size=num_states,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size + input_size, num_states)

        # 初始化预测器
        self.predictors = nn.Linear(input_size, num_states)

    def forward(self, hidden, hist_loss):
        # 通过预测器获得预测结果
        preds = self.predictors(hidden)

        # 如果状态数为1，退化为传统方法
        if self.num_states == 1:
            return preds.squeeze(-1), preds, None

        # 根据信息类型，从历史损失中获取路由器输出
        router_out, _ = self.router(hist_loss)
        if "LR" in self.src_info:
            latent_representation = hidden
        else:
            latent_representation = torch.randn(hidden.shape).to(hidden)
        if "TPE" in self.src_info:
            temporal_pred_error = router_out[:, -1]
        else:
            temporal_pred_error = torch.randn(router_out[:, -1].shape).to(hidden)

        # 将时序预测误差和潜在表示结合起来，通过全连接层获得输出
        out = self.fc(torch.cat([temporal_pred_error, latent_representation], dim=-1))
        # 使用Gumbel Softmax计算概率
        prob = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=False)

        # 在训练模式下，使用概率加权预测结果；在推断模式下，选择概率最高的预测结果
        if self.training:
            final_pred = (preds * prob).sum(dim=-1)
        else:
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)]
        return final_pred, preds, prob


def evaluate(pred):
    """
    评估函数，用于计算模型预测的多个性能指标。

    Args:
        pred (DataFrame): 包含模型预测得分和真实标签的DataFrame。

    Returns:
        dict: 包含MSE、MAE和IC的字典。
    """
    # 将预测得分转换为百分位数
    pred = pred.rank(pct=True)

    # 从DataFrame中提取预测得分和真实标签
    score = pred.score
    label = pred.label

    # 计算预测得分和真实标签之间的差异
    diff = score - label

    # 计算均方误差（MSE）
    MSE = (diff**2).mean()

    # 计算平均绝对误差（MAE）
    MAE = (diff.abs()).mean()

    # 计算信息系数（IC），即预测得分和真实标签之间的相关性
    IC = score.corr(label)

    # 返回包含这些指标的字典
    return {"MSE": MSE, "MAE": MAE, "IC": IC}


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError("the %d-th model has different params" % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def shoot_infs(inp_tensor):
    """
    将输入张量中的无穷大值替换为张量的最大值。

    Args:
        inp_tensor (Tensor): 输入的张量。
    """
    # 创建一个掩码，标记出所有无穷大值的位置
    mask_inf = torch.isinf(inp_tensor)
    # 获取包含无穷大值位置的索引
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)

    # 如果存在无穷大值，则将其替换为张量的最大值
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.01):
    """
    Args:
        Q (Tensor): 输入的成本矩阵或者对数似然矩阵
        n_iters (int): Sinkhorn算法的迭代次数
        epsilon (float): 用于调整Q值规模的参数
    """
    with torch.no_grad():
        # 处理Q矩阵中的无穷大值
        Q = shoot_infs(Q)
        # 应用Softmax变换，epsilon用于调整Softmax函数的平滑度
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            # 按列归一化
            Q /= Q.sum(dim=0, keepdim=True)
            # 按行归一化
            Q /= Q.sum(dim=1, keepdim=True)
    return Q


class RNN(nn.Module):
    def __init__(
        self,
        input_size=14,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        input_drop=0.0,
        noise_level=0.0,
        *args,
        **kwargs,
    ):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_drop = nn.Dropout(input_drop)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.output_size = hidden_size

    def forward(self, x):
        x = self.input_drop(x)
        r_out, _ = self.rnn(x)
        last_out = r_out[:, -1]

        return last_out


class LSTM_HA(nn.Module):
    def __init__(
        self,
        input_size=14,
        window_len=20,
        hidden_size=64,
        dropout=0.0,
        input_drop=0.0,
        noise_level=0.0,
        *args,
        **kwargs,
    ):
        """self,
        input_size=16,
        hidden_size=64,
        num_layers=2,
        use_attn=True,
        dropout=0.0,
        input_drop=0.0,
        noise_level=0.0,
        *args,
        **kwargs,


        :param in_features: 单个股票特征数
        :param window_len: 时间窗长度
        :param hidden_dim: 隐藏层大小
        :param output_dim: 输出层大小"""

        super(LSTM_HA, self).__init__()
        # self.device = get_device(device)

        self.input_size = input_size
        self.window_len = window_len
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.input_drop = nn.Dropout(input_drop)

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.attn1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.attn2 = nn.Linear(hidden_size, 1, bias=False)

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.input_drop(x)
        """
        :X: [batch_size(B), num_stocks, window_len(L), in_features(I)]
        :return: Parameters: [batch, 2]
        """

        outputs, (h_n, c_n) = self.lstm(x)

        # 最后一个状态h_n与之前的所有状态outputs做attention
        H_n = h_n.repeat((self.window_len, 1, 1)).transpose(1, 0)

        scores = self.attn2(torch.tanh(self.attn1(torch.cat([outputs, H_n], dim=2))))
        scores = scores.squeeze(2).transpose(1, 0)
        attn_weights = torch.softmax(scores, dim=1).transpose(1, 0)
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))
        return embed
