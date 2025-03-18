import numpy as np
import simpy as sp
import prettytable as pt
import traceback
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from . import extendSimpy as esp
from .utils import A1_location as a1loc
from .utils import A3_time_on_air as a3toa
from .utils import A4_propagation as a4pro

import pickle

from os.path import basename as osfun
from sys import _getframe as sysfun
import os
from types import SimpleNamespace
from collections import Counter

from tqdm import tqdm
import re
import gc
import torch as th

second = 1000
minute = 60 * second
hour = 60 * minute
day = 24 * hour


result_dir_name = 'result'

import torch as th

import shutil


def db2abs(x):
    return np.power(10, x / 10)


def abs2db(x):
    return 10 * np.log10(x)


def hex_to_rgba(hex_color, alpha):
    # 移除可能存在的 "#" 字符
    hex_color = hex_color.lstrip("#")
    # 将hex颜色字符串转换为RGBA值
    rgba = 'rgba({},{},{},{})'.format(
        int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), alpha
    )
    return rgba


import palettable.cartocolors.sequential as cartocolors
import palettable.cmocean.sequential as cmocean


def getColor(name, number, alpha):
    if name == 'blue' or name == 0:
        if number == -1:
            return 'rgba(31,119,180,' + str(alpha) + ')'
        else:
            return hex_to_rgba(cmocean.Ice_20.hex_colors[number], alpha)
    elif name == 'orange' or name == 1:
        if number == -1:
            return 'rgba(255, 127, 14,' + str(alpha) + ')'
        else:
            return hex_to_rgba(cmocean.Solar_20.hex_colors[number], alpha)
    elif name == 'green' or name == 2:
        if number == -1:
            return 'rgba(44, 160, 44,' + str(alpha) + ')'
        else:
            return hex_to_rgba(cmocean.Algae_20.hex_colors[number], alpha)
    elif name == 'red' or name == 3:
        if number == -1:
            return 'rgba(214, 39, 40,' + str(alpha) + ')'
        else:
            return hex_to_rgba(cmocean.Amp_20.hex_colors[number], alpha)
    elif name == 'purple' or name == 4:
        return 'rgba(148, 103, 189,' + str(alpha) + ')'
    elif name == 'brown' or name == 5:
        return 'rgba(140, 86, 75,' + str(alpha) + ')'
    elif name == 'pink' or name == 6:
        if number == -1:
            return 'rgba(227, 119, 194,' + str(alpha) + ')'
        else:
            return hex_to_rgba(cartocolors.BurgYl_7.hex_colors[number], alpha)

    elif name == 'gray' or name == 7:
        return 'rgba(127, 127, 127,' + str(alpha) + ')'
    elif name == 'grass green' or name == 8:
        return 'rgba(188, 189, 34,' + str(alpha) + ')'
    elif name == 'baby blue' or name == 9:
        return 'rgba(23, 190, 207,' + str(alpha) + ')'


class Packet:
    def __init__(self, sender, receiver, env: sp.Environment, loraEnv):
        self.sender = sender  # 发送者
        self.receiver = receiver  # 目标接收者
        self.env = env
        self.loraEnv = loraEnv

        # 接收机噪声,干扰,sinr
        self.noiseToGWs = np.zeros(loraEnv.GW_num)
        self.interferenceToGWs = np.zeros(loraEnv.GW_num)  # 初始化置为0
        self.sinrToGWs = np.zeros(loraEnv.GW_num)

        self.conditionsToGWs = np.empty(
            (4, loraEnv.GW_num), dtype=np.bool_
        )  # 初始化时置为true

        self.c3_strongestToGWs = np.empty(
            loraEnv.GW_num, dtype=np.bool_
        )  # 初始化时置为true
        self.c3_early1ToGWs = np.empty(
            loraEnv.GW_num, dtype=np.bool_
        )  # 初始化时置为true
        self.c3_early2ToGWs = np.empty(
            loraEnv.GW_num, dtype=np.bool_
        )  # 初始化时置为true
        self.c3_freeDemToGWs = np.empty(
            loraEnv.GW_num, dtype=np.bool_
        )  # 初始化时置为true

    def updateRX(self, rx):
        self.rx_target = rx
        if rx == 'rx1':
            self.tx_sf = self.tx_sf_rx1
            self.tx_channel = self.tx_channel_rx1
            self.timepoint_tx = self.timepoint_tx_rx1

        else:
            self.tx_sf = self.tx_sf_rx2
            self.tx_channel = self.tx_channel_rx2
            self.timepoint_tx = self.timepoint_tx_rx2
        self.TOA = a3toa.TOA(self.tx_sf, self.payload)  # 空中时间
        self.TOA_4SymbolTime = a3toa.symbolTime(self.tx_sf, 4)  # 计算前4个symbol的时间
        self.TOA_rest = self.TOA - self.TOA_4SymbolTime  # 剩余TOA时间


def updateC0(pac: Packet, loraEnv):
    if isinstance(pac.sender, GW):
        if (
            loraEnv.rssi[pac.sender._id][pac.receiver._id]
            >= loraEnv.RSSI_EN_threshold[pac.tx_sf]
        ):
            pac.conditionsToGWs[0][pac.sender.id] = True
        else:
            pac.conditionsToGWs[0][pac.sender.id] = False
    elif pac.receiver is None:
        for gw in loraEnv.GWs:
            if (
                loraEnv.rssi[pac.sender._id][gw._id]
                >= loraEnv.RSSI_GW_threshold[pac.tx_sf]
            ):
                pac.conditionsToGWs[0][gw.id] = True
            else:
                pac.conditionsToGWs[0][gw.id] = False
    else:
        pass


is_same_channel = False


def updateInterference(pac: Packet, loraEnv):
    if isinstance(pac.sender, GW):
        interference = 0
        for sf, pacs in enumerate(loraEnv.sending_packets[pac.tx_channel]):
            for other in pacs:
                t = isinstance(other.sender, EN)
                if (other is pac) or (t and other.sender.id == pac.receiver.id):
                    continue

                if not is_same_channel and type(pac.sender) != type(other.sender):
                    continue

                interference += (
                    loraEnv.interCoefficient[other.tx_sf - 7][sf - 7]
                    * other.tx_pow
                    * other.channelGain
                    * loraEnv.pathloss[other.sender._id][pac.receiver._id]
                )

        if interference > pac.interferenceToGWs[pac.sender.id]:
            pac.interferenceToGWs[pac.sender.id] = interference
    elif pac.receiver is None:
        for gw in loraEnv.GWs:
            interference = 0
            for sf, pacs in enumerate(loraEnv.sending_packets[pac.tx_channel]):
                for other in pacs:
                    t = isinstance(other.sender, GW)
                    if (other is pac) or (t and other.sender.id == gw.id):
                        continue

                    if not is_same_channel and type(pac.sender) != type(other.sender):
                        continue

                    interference += (
                        loraEnv.interCoefficient[other.tx_sf - 7][sf - 7]
                        * loraEnv.rssi[other.sender._id][gw._id]
                    )
            if interference > pac.interferenceToGWs[gw.id]:
                pac.interferenceToGWs[gw.id] = interference
    else:
        pass


# 获得接收机noise,计算SINR并判断SINR是否大于阈值
def updateC1(pac: Packet, loraEnv):
    if isinstance(pac.sender, GW):
        pac.noiseToGWs[pac.sender.id] = pac.receiver.receiver_noise
        if pac.noiseToGWs[pac.sender.id] + pac.interferenceToGWs[pac.sender.id] == 0:
            pac.sinrToGWs[pac.sender.id] = np.inf
        else:
            pac.sinrToGWs[pac.sender.id] = loraEnv.rssi[pac.sender._id][
                pac.receiver._id
            ] / (pac.noiseToGWs[pac.sender.id] + pac.interferenceToGWs[pac.sender.id])
        if pac.sinrToGWs[pac.sender.id] >= loraEnv.SINR_threshold[pac.tx_sf]:
            pac.conditionsToGWs[1][pac.sender.id] = True
        else:
            pac.conditionsToGWs[1][pac.sender.id] = False
    elif pac.receiver is None:
        for i in range(loraEnv.GW_num):
            pac.noiseToGWs[i] = loraEnv.GWs[i].receiver_noise
            if pac.noiseToGWs[i] + pac.interferenceToGWs[i] == 0:
                pac.sinrToGWs[i] = np.inf
            else:
                pac.sinrToGWs[i] = loraEnv.rssi[pac.sender._id][loraEnv.GWs[i]._id] / (
                    pac.noiseToGWs[i] + pac.interferenceToGWs[i]
                )
            if pac.sinrToGWs[i] >= loraEnv.SINR_threshold[pac.tx_sf]:
                pac.conditionsToGWs[1][i] = True
            else:
                pac.conditionsToGWs[1][i] = False
    else:
        pass


# 判断正在发射的所有同(channel,tx_sf)信号,是否在整个发射期间为最强且4倍的信号
# 有信号开始发射时,进行更新
def updateC2(pac: Packet, loraEnv):

    if is_same_channel or pac.receiver is None:
        for gw in loraEnv.GWs:
            one = None
            two = None
            for p in loraEnv.sending_packets[pac.tx_channel][pac.tx_sf]:

                if isinstance(p.sender, GW) and p.sender.id == gw.id:
                    continue

                if not is_same_channel and type(pac.sender) != type(p.sender):
                    continue

                if one is None:
                    one = p
                elif two is None:
                    if (
                        loraEnv.rssi[p.sender._id][gw._id]
                        > loraEnv.rssi[one.sender._id][gw._id]
                    ):
                        one.conditionsToGWs[2][gw.id] = False
                        two = one
                        one = p
                    else:
                        p.conditionsToGWs[2][gw.id] = False
                        two = p
                else:
                    if (
                        loraEnv.rssi[p.sender._id][gw._id]
                        > loraEnv.rssi[one.sender._id][gw._id]
                    ):
                        one.conditionsToGWs[2][gw.id] = False
                        two = one
                        one = p
                    elif (
                        loraEnv.rssi[p.sender._id][gw._id]
                        > loraEnv.rssi[two.sender._id][gw._id]
                    ):
                        p.conditionsToGWs[2][gw.id] = False
                        two = p
                    else:
                        p.conditionsToGWs[2][gw.id] = False

            if (
                one is not None
                and two is not None
                and loraEnv.rssi[one.sender._id][gw._id]
                < 4 * loraEnv.rssi[two.sender._id][gw._id]
            ):
                one.conditionsToGWs[2][gw.id] = False

    if is_same_channel or isinstance(pac.receiver, EN):
        for p in loraEnv.sending_packets[pac.tx_channel][pac.tx_sf]:
            if isinstance(p.sender, GW):
                for other in loraEnv.sending_packets[pac.tx_channel][pac.tx_sf]:
                    if (other is p) or (
                        isinstance(other.sender, EN)
                        and other.sender.id == p.receiver.id
                    ):
                        continue

                    if not is_same_channel and type(pac.sender) != type(other.sender):
                        continue

                    if (
                        loraEnv.rssi[p.sender._id][p.receiver._id]
                        < 4
                        * other.tx_pow
                        * other.channelGain
                        * loraEnv.pathloss[other.sender._id][p.receiver._id]
                    ):
                        p.conditionsToGWs[2][p.sender.id] = False
                        break
            elif isinstance(p.sender, EN) and p.receiver is not None:
                pass


def updateStrongest(pac: Packet, loraEnv):
    if is_same_channel or pac.receiver is None:
        for gw in loraEnv.GWs:
            # 按照rssi降序排列
            loraEnv.sending_packets_4symbol[pac.tx_channel][pac.tx_sf].sort(
                key=lambda pac: loraEnv.rssi[pac.sender._id][gw._id], reverse=True
            )

            target = 0
            for p in loraEnv.sending_packets_4symbol[pac.tx_channel][pac.tx_sf]:
                if isinstance(p.sender, GW) and p.sender.id == gw.id:
                    continue

                if not is_same_channel and type(pac.sender) != type(p.sender):
                    continue

                if target > 0:
                    p.c3_strongestToGWs[gw.id] = False
                target += 1

    if is_same_channel or isinstance(pac.receiver, EN):
        for p in loraEnv.sending_packets_4symbol[pac.tx_channel][pac.tx_sf]:
            # 判断GW->EN的情况
            if isinstance(p.sender, GW):
                for other in loraEnv.sending_packets_4symbol[pac.tx_channel][pac.tx_sf]:
                    if (other is p) or (
                        isinstance(other.sender, EN)
                        and other.sender.id == p.receiver.id
                    ):
                        continue

                    if not is_same_channel and type(pac.sender) != type(p.sender):
                        continue

                    if (
                        loraEnv.rssi[p.sender._id][p.receiver._id]
                        < other.tx_pow
                        * other.channelGain
                        * loraEnv.pathloss[other.sender._id][p.receiver._id]
                    ):
                        p.c3_strongestToGWs[p.sender.id] = False
                        break
            elif isinstance(p.sender, EN) and p.receiver is not None:
                pass


# 发射的前处理
def send_pre(pac: Packet, loraEnv):
    # 计算两个时间点
    pac.timepoint_tx = loraEnv.env.now
    pac.timepoint_4symbol = loraEnv.env.now + pac.TOA_4SymbolTime

    # 加入正在发射的集合
    loraEnv.sending_packets[pac.tx_channel][pac.tx_sf].append(pac)
    loraEnv.sending_packets_4symbol[pac.tx_channel][pac.tx_sf].append(pac)
    loraEnv.sending_packets_one.append(pac)
    loraEnv.sending_packets_channel[pac.tx_channel].append(pac)

    updateC0(pac, loraEnv)

    # 更新自己在发射时受到的干扰
    updateInterference(pac, loraEnv)
    for pacs in loraEnv.sending_packets[pac.tx_channel]:
        for other in pacs:
            updateInterference(other, loraEnv)

    updateC2(pac, loraEnv)
    updateStrongest(pac, loraEnv)


# 发射后处理
def send_post(pac: Packet, loraEnv):
    # 从正在发射的集合移除
    loraEnv.sending_packets[pac.tx_channel][pac.tx_sf].remove(pac)
    loraEnv.sending_packets_one.remove(pac)
    loraEnv.sending_packets_channel[pac.tx_channel].remove(pac)

    updateC1(pac, loraEnv)


def createFile(file_path):
    if not os.path.exists(file_path):
        # 如果文件不存在，创建文件及其文件夹
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


def openFile(file_path, t):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 如果文件不存在，创建文件及其文件夹
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return open(file_path, t)


class EN:
    # rho:节点发射频率,次/分钟
    def __init__(self, id, x, y, rho: float, env: sp.Environment, loraEnv):
        self.id = id
        self._id = id  # 此_id用来检索距离表,,GW和EN的_id不会出现相同的情况
        self.x = x
        self.y = y
        self.distance0 = np.sqrt(np.power(self.x, 2) + np.power(self.y, 2))
        self.rho = rho / (60.0 * 1000.0)  # 次/分钟 ->  次/毫秒
        self.env = env
        self.loraEnv: LoRaEnv = loraEnv

        self.rng_base = np.random.default_rng(self._id + loraEnv.param['seed'])
        self.rng_channel = np.random.default_rng(self.rng_base.integers(0, 2**63))
        self.rng_nosie = np.random.default_rng(self.rng_base.integers(0, 2**63))
        self.rng_pl = np.random.default_rng(self.rng_base.integers(0, 2**63))
        self.rng_sleep = np.random.default_rng(self.rng_base.integers(0, 2**63))
        self.rng_channelGain = np.random.default_rng(self.rng_base.integers(0, 2**63))
        self.rng_sf = np.random.default_rng(self.rng_base.integers(0, 2**63))

        self.tx_sf = None  # 遵循tx_sf_gw_d
        self.tx_pow_lv = None
        self.tx_pow = None  # 遵循tx_pow_gw_d
        self.tx_channel = None  # 遵循tx_channel_range_gw_d

        self.rx1_sf = None  # 遵循rx1_sf_offset_gw_f
        self.rx1_channel = None  # 同tx_channel
        self.rx2_sf = None  # 遵循rx2_sf_gw_f
        self.rx2_channel = None  # 遵循rx2_channel_gw_f

        self.rx1_delay = None  # 遵循rx1_delay_gw_f
        self.rx2_delay = 1000  # 不变,从rx1_delay结束后开始计算
        self.rx_size = 500

        self.payload = None  # 负载

        self.strategy = loraEnv.mainStrategy()

        # EN状态
        # 0:tx  1:rx_wait  2:rx  3:sleep
        self.status = 3

        # ACK的状态
        # 0:收到正确的ack
        # 1:收到乱码
        # 2:未收到ACK
        self.ACK_Status = 0

        self.event_ACK_4symbol = esp.extendEvent(
            env,
            'EN{}'.format(self.id),
            'receive ACK 4symbol end',
            '{}({})'.format(osfun(__file__), sysfun().f_lineno),
        )

        self.event_ACK_end = esp.extendEvent(
            env,
            'EN{}'.format(self.id),
            'receive ACK end',
            '{}({})'.format(osfun(__file__), sysfun().f_lineno),
        )

        self.up_sentNum = 0  # 总发射数量
        self.up_demSuccessToGWs = np.zeros(
            loraEnv.GW_num, dtype=int
        )  # 成功被该GW解调的报文数量
        self.up_demFailToGWs = np.zeros(
            loraEnv.GW_num, dtype=int
        )  # 占用到了解调器,但未被该GW解调的报文数量
        self.up_noDemToGWs = np.zeros(
            loraEnv.GW_num, dtype=int
        )  # 未占用到该GW的解调器的报文数量
        self.up_demSuccess = 0
        self.up_demFail = 0
        self.up_noDem = 0


        # up_demSuccessToGWs[i] = ..Ack..[i] + ..up_demFail..[i] + ..up_noDem..[i] + ..NotMe..[i] + ..NoAck..[i]
        self.ack_demSuccessToGWs = np.zeros(
            loraEnv.GW_num, dtype=int
        )  # 收到该网关正确ack的报文数量
        self.ack_demFailToGWs = np.zeros(
            loraEnv.GW_num, dtype=int
        )  # 收到了该网关乱码的报文数量
        self.ack_noDemToGWs = np.zeros(loraEnv.GW_num, dtype=int)  # ack未占用EN的解调器
        self.ack_notMeToGWs = np.zeros(loraEnv.GW_num, dtype=int)  # 不是本网关发送ack
        self.ack_noAckToGWs = np.zeros(loraEnv.GW_num, dtype=int)  # ns未发送ack

        # up_demSuccess = ack_demSuccess + ack_demFail + ack_noAck + ack_noDem
        self.ack_demSuccess = 0  # 收到ACK,ack_demSuccess++
        self.ack_demFail = 0  # ACK解调失败
        self.ack_noDem = 0  # ACK未占用该EN的解调器
        self.ack_noAck = 0  # NS未发送ACK

        # up_demFailToGWs[i] <= ..C1ToGWs[i] + ..C2ToGWs[i]
        self.up_demFailC1ToGWs = np.zeros(loraEnv.GW_num, dtype=int)  # 解调失败的原因
        self.up_demFailC2ToGWs = np.zeros(loraEnv.GW_num, dtype=int)
        # up_demFail <= up_demFailC1 + up_demFailC2
        self.up_demFailC1 = 0
        self.up_demFailC2 = 0

        # up_noDemToGWs[i] = ..C0ToGWs[i]+..c3_early1ToGWs[i]+..c3_early2ToGWs[i]+..c3_freeDemToGWs[i]
        self.up_noDemStrongestToGWs = np.zeros(
            loraEnv.GW_num, dtype=int
        )  # 未占用解调器的原因
        self.up_noDemEarly1ToGWs = np.zeros(
            loraEnv.GW_num, dtype=int
        )  # 本报文为最强报文,发射时间晚于弱信号的4符号结束时间
        self.up_noDemEarly2ToGWs = np.zeros(
            loraEnv.GW_num, dtype=int
        )  # 本报文为弱报文,4符号结束时间晚于强信号的发射时间
        self.up_noDemFreeDemToGWs = np.zeros(
            loraEnv.GW_num, dtype=int
        )  # 网关处没有空闲解调器
        # up_noDem = up_noDemStrongest + up_noDemEarly1 + up_noDemEarly2 + up_noDemFreeDem
        self.up_noDemStrongest = 0
        self.up_noDemEarly1 = 0
        self.up_noDemEarly2 = 0
        self.up_noDemFreeDem = 0

        # ack_demFailToGWs[i] <= ack_demFailC1ToGWs[i] + ack_demFailC2ToGWs[i]
        self.ack_demFailC1ToGWs = np.zeros(loraEnv.GW_num, dtype=int)
        self.ack_demFailC2ToGWs = np.zeros(loraEnv.GW_num, dtype=int)
        self.ack_demFailC1 = 0
        self.ack_demFailC2 = 0

        # ack_noDemToGWs[i] = ack_noDemStrongestToGWs[i]+ack_noDemEarly1ToGWs[i]+ack_noDemEarly2ToGWs[i]+ack_noDemFreeDemToGWs[i]
        self.ack_noDemStrongestToGWs = np.zeros(loraEnv.GW_num, dtype=int)
        self.ack_noDemEarly1ToGWs = np.zeros(loraEnv.GW_num, dtype=int)
        self.ack_noDemEarly2ToGWs = np.zeros(loraEnv.GW_num, dtype=int)
        self.ack_noDemFreeDemToGWs = np.zeros(loraEnv.GW_num, dtype=int)
        self.ack_noDemStrongest = 0
        self.ack_noDemEarly1 = 0
        self.ack_noDemEarly2 = 0
        self.ack_noDemFreeDem = 0

        self.data_separate = np.zeros(shape=(self.loraEnv.GW_num, 6), dtype=int)
        self.data_separate_ch = np.zeros(shape=(self.loraEnv.GW_num, 8, 6), dtype=int)
        self.data_separate_sf = np.zeros(shape=(self.loraEnv.GW_num, 6, 6), dtype=int)

        # 占用的解调器
        self.occupiedDemodulator = []

        # 解调器的数量
        self.demodulatorNum = 1
        self.demodulatingPacs = []  # 正在被解调的报文

        # 日志目录
        self.logTable = pt.PrettyTable()
        self.logTable.set_style(pt.SINGLE_BORDER)
        self.logFilename = self.loraEnv.root + 'log_EN/EN{}.txt'.format(id)


        self.pac = Packet(self, None, env, loraEnv)

        self.receiver_noise = 0

        # 噪声为随机变量
        # env.process(self.updateNoise(env))

        # 噪声为取方差
        # 方差为 -174 + 6 + 10*log10(125000) dbm --> np.power(10, (-174 + 6) / 10) * 125000 mw
        self.receiver_noise = np.power(10, (-174 + 6) / 10) * 125000

        # 使用相互独立的信道增益
        self.channelGain = 1
        self.process_channelGain = env.process(self.updateChannelGain(env))

        self.process = env.process(self.working(env))

    def working(self, env: sp.Environment):
        try:
            log1 = []
            log2 = []

            def logAppend(p1, p2):
                if len(log1) > 0 and p1 == log1[len(log1) - 1]:
                    p1 = ''
                log1.append(p1)
                log2.append(p2)

            while not self.loraEnv.stop:
                log1.clear()
                log2.clear()
                ACK = None

                self.strategy.EN_initUplink(self)


                # 开始发射前4个symbol
                t1 = env.now
                self.status = 0
                send_pre(self.pac, self.loraEnv)

                yield esp.extendTimeout(
                    env,
                    self.pac.TOA_4SymbolTime,
                    'EN{}'.format(self.id),
                    'TX4',
                    '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                )

                # 申请解调器
                self.request_demodulator(self.pac)

                # 开始发射剩余部分
                yield esp.extendTimeout(
                    env,
                    self.pac.TOA_rest,
                    'EN{}'.format(self.id),
                    'TX_rest',
                    '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                )
                # 整个报文发送完毕,释放解调器,后处理
                self.release_demodulator(self.pac)
                send_post(self.pac, self.loraEnv)

                # 通知GW发送完毕
                for gw in self.loraEnv.GWs:
                    esp.extendSucceed(env, gw.event_uplink_end, 0, self.pac)
                    gw.event_uplink_end = esp.extendEvent(
                        env,
                        'EN{}->GW{}'.format(self.id, gw.id),
                        'uplink_end',
                        '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                    )

                # 顺便发送给NS
                esp.extendSucceed(env, self.loraEnv.ns.event_uplink_end, 0, self.pac)
                self.loraEnv.ns.event_uplink_end = esp.extendEvent(
                    env,
                    'EN{}->NS'.format(self.id),
                    'uplink_end',
                    '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                )

                # 记录几个阶段的时间点
                timepoint_w1_d2 = env.now + self.rx1_delay + self.rx_size
                timepoint_d2_w2 = timepoint_w1_d2 + self.rx2_delay - self.rx_size
                timepoint_w2_end = timepoint_d2_w2 + self.rx_size

                try:
                    self.status = 3

                    yield esp.extendTimeout(
                        env,
                        self.rx1_delay,
                        'EN{}'.format(self.id),
                        'D1',
                        '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                    )

                    self.status = 1

                    yield esp.extendTimeout(
                        env,
                        self.rx_size,
                        'EN{}'.format(self.id),
                        'W1',
                        '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                    )

                    self.status = 3

                    yield esp.extendTimeout(
                        env,
                        self.rx2_delay - self.rx_size,
                        'EN{}'.format(self.id),
                        'D2',
                        '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                    )

                    self.status = 1

                    yield esp.extendTimeout(
                        env,
                        self.rx_size,
                        'EN{}'.format(self.id),
                        'W2',
                        '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                    )

                    if self.loraEnv.waiting_ACK[self.id] != 0:
                        self.loraEnv.GWs[self.loraEnv.waiting_ACK[self.id]].removeACK(
                            self.id, self.pac.packetNum
                        )


                    self.ACK_Status = 2
                except sp.Interrupt as i:
                    ACK = i.cause

                    self.status = 1
                    yield self.event_ACK_4symbol
                    if not ACK.conditionsToGWs[3][ACK.sender.id]:
                        # 未占用解调器,继续try块中的休眠
                        if env.now < timepoint_w1_d2:
                            # 当前在w1内
                            self.status = 1
                            yield esp.extendTimeout(
                                env,
                                timepoint_w1_d2 - env.now,
                                'EN{}'.format(self.id),
                                'W1_complement',
                                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                            )

                            self.status = 3

                            yield esp.extendTimeout(
                                env,
                                self.rx2_delay - self.rx_size,
                                'EN{}'.format(self.id),
                                'D2',
                                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                            )

                            self.status = 1

                            yield esp.extendTimeout(
                                env,
                                self.rx_size,
                                'EN{}'.format(self.id),
                                'W2',
                                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                            )

                        elif env.now < timepoint_d2_w2:
                            # 当前在d2内
                            self.status = 3
                            yield esp.extendTimeout(
                                env,
                                timepoint_d2_w2 - env.now,
                                'EN{}'.format(self.id),
                                'D2_complement',
                                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                            )

                            self.status = 1

                            yield esp.extendTimeout(
                                env,
                                self.rx_size,
                                'EN{}'.format(self.id),
                                'W2',
                                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                            )
                        elif env.now < timepoint_w2_end:
                            # 当前在w2内
                            self.status = 1
                            yield esp.extendTimeout(
                                env,
                                timepoint_w2_end - env.now,
                                'EN{}'.format(self.id),
                                'W2_complement',
                                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                            )

                        else:
                            # 当前在w2之后
                            pass
                    else:
                        pass

                        self.status = 2
                        yield self.event_ACK_end

                        c1 = ACK.conditionsToGWs[1][ACK.sender.id]
                        c2 = ACK.conditionsToGWs[2][ACK.sender.id]
                        c3 = ACK.conditionsToGWs[3][ACK.sender.id]

                        if c1 and c2 and c3:
                            self.ACK_Status = 0
                        else:
                            self.ACK_Status = 1
                self.status = 3
                t2 = env.now
                rho_p = self.rho / (1 - self.rho * (t2 - t1))
                self.sleeptime = int(self.rng_sleep.exponential(scale=1 / rho_p))

                res = self.strategy.EN_dispatch(self, ACK, self.ACK_Status)

                self.statDetail(self.pac, ACK)  # 统计历史数据
                self.status = 3
                yield esp.extendTimeout(
                    env,
                    self.sleeptime,
                    'EN{}'.format(self.id),
                    'sleep',
                    '{}({})'.format(osfun(__file__), sysfun().f_lineno),
                )
        except Exception as e:
            self.logFile.close()
            raise e

    def judgePrint(self, logType, lc):
        shouldPrint = lc[1][0] != 3 and (
            (lc[1][0] == 0)
            or (lc[1][0] == 1 and self.env.now >= lc[1][1] and self.env.now <= lc[1][2])
            or (
                lc[1][0] == 2
                and self.up_sentNum >= lc[1][1]
                and self.up_sentNum <= lc[1][2]
            )
        )

        if logType == 'TXRX':
            return lc[0][0] and shouldPrint
        elif logType == 'dispatch':
            return lc[0][1] and shouldPrint
        elif logType == 'detail':
            return lc[0][2] and shouldPrint

    def logPrint(self, log1: list, log2: list):
        self.logTable.clear()
        self.logTable.field_names = log1
        self.logTable.add_row(log2)
        self.logFile.write(self.logTable.__str__() + '\n')


    # 统计历史数据
    def statDetail(self, uplink: Packet, ACK: Packet):
        self.up_sentNum += 1

        for i in range(self.loraEnv.GW_num):
            c1 = uplink.conditionsToGWs[1][i]
            c2 = uplink.conditionsToGWs[2][i]
            c3 = uplink.conditionsToGWs[3][i]

            c0 = uplink.conditionsToGWs[0][i]
            t1 = uplink.c3_early1ToGWs[i]
            t3 = uplink.c3_freeDemToGWs[i]
            if c1 and c2 and c0 and t1 and t3:
                self.data_separate[i][0] += 1
                self.data_separate_ch[i][uplink.tx_channel][0] += 1
                self.data_separate_sf[i][uplink.tx_sf - 7][0] += 1

            if c1:
                self.data_separate[i][1] += 1
                self.data_separate_ch[i][uplink.tx_channel][1] += 1
                self.data_separate_sf[i][uplink.tx_sf - 7][1] += 1
            if c2:
                self.data_separate[i][2] += 1
                self.data_separate_ch[i][uplink.tx_channel][2] += 1
                self.data_separate_sf[i][uplink.tx_sf - 7][2] += 1
            if c0:
                self.data_separate[i][3] += 1
                self.data_separate_ch[i][uplink.tx_channel][3] += 1
                self.data_separate_sf[i][uplink.tx_sf - 7][3] += 1
            if t1:
                self.data_separate[i][4] += 1
                self.data_separate_ch[i][uplink.tx_channel][4] += 1
                self.data_separate_sf[i][uplink.tx_sf - 7][4] += 1
            if t3:
                self.data_separate[i][5] += 1
                self.data_separate_ch[i][uplink.tx_channel][5] += 1
                self.data_separate_sf[i][uplink.tx_sf - 7][5] += 1

        # 更新该报文到各个网关的历史数据
        for i in range(self.loraEnv.GW_num):
            if not uplink.conditionsToGWs[3][i]:
                self.up_noDemToGWs[i] += 1  # 未占用到该GW的解调器的报文数量
                if not uplink.conditionsToGWs[0][i]:
                    self.up_noDemStrongestToGWs[i] += 1
                elif not uplink.c3_freeDemToGWs[i]:
                    self.up_noDemFreeDemToGWs[i] += 1
                elif uplink.c3_strongestToGWs[i] and not uplink.c3_early1ToGWs[i]:
                    self.up_noDemEarly1ToGWs[i] += 1
                elif not uplink.c3_strongestToGWs[i] and not uplink.c3_early2ToGWs[i]:
                    self.up_noDemEarly2ToGWs[i] += 1

            elif (not uplink.conditionsToGWs[1][i]) or (
                not uplink.conditionsToGWs[2][i]
            ):
                self.up_demFailToGWs[i] += 1  # 占用到了解调器,但未被该GW解调的报文数量
                if not uplink.conditionsToGWs[1][i]:
                    self.up_demFailC1ToGWs[i] += 1
                if not uplink.conditionsToGWs[2][i]:
                    self.up_demFailC2ToGWs[i] += 1
            else:
                self.up_demSuccessToGWs[i] += 1  # 成功被该GW解调的报文数量
                if ACK is None:
                    self.ack_noAckToGWs[i] += 1 
                else:
                    if ACK.sender.id == i:
                        if not ACK.conditionsToGWs[3][i]:
                            self.ack_noDemToGWs[i] += 1  # ACK未占用该EN的解调器
                            if not ACK.conditionsToGWs[0][i]:
                                self.ack_noDemStrongestToGWs[i] += 1
                            elif not ACK.c3_freeDemToGWs[i]:
                                self.ack_noDemFreeDemToGWs[i] += 1
                            elif ACK.c3_strongestToGWs[i] and not ACK.c3_early1ToGWs[i]:
                                self.ack_noDemEarly1ToGWs[i] += 1
                            elif (
                                not ACK.c3_strongestToGWs[i]
                                and not ACK.c3_early2ToGWs[i]
                            ):
                                self.ack_noDemEarly2ToGWs[i] += 1

                        elif (not ACK.conditionsToGWs[1][ACK.sender.id]) or (
                            not ACK.conditionsToGWs[2][ACK.sender.id]
                        ):
                            self.ack_demFailToGWs[i] += 1  
                            if not ACK.conditionsToGWs[1][ACK.sender.id]:
                                self.ack_demFailC1ToGWs[i] += 1
                            if not ACK.conditionsToGWs[2][ACK.sender.id]:
                                self.ack_demFailC1ToGWs[i] += 1
                        else:
                            self.ack_demSuccessToGWs[
                                i
                            ] += 1  # 收到该网关正确ack的报文数量
                    else:
                        self.ack_notMeToGWs[i] += 1  # 不是该网关发送ack
                        self.loraEnv.GWs[i].ack_notMe += 1

        success = []  # 解调uplink成功的GW
        fail = []  # 解调uplink失败的GW
        up_noDem = []  # uplink占用解调器失败的GW
        for i in range(self.loraEnv.GW_num):
            if not uplink.conditionsToGWs[3][i]:
                up_noDem.append(i)
            elif (not uplink.conditionsToGWs[1][i]) or (
                not uplink.conditionsToGWs[2][i]
            ):
                fail.append(i)
            else:
                success.append(i)

        if len(success) > 0:
            self.up_demSuccess += 1

            if ACK is None:
                self.ack_noAck += 1
            else:
                c0 = ACK.conditionsToGWs[0][ACK.sender.id]
                c1 = ACK.conditionsToGWs[1][ACK.sender.id]
                c2 = ACK.conditionsToGWs[2][ACK.sender.id]
                c3 = ACK.conditionsToGWs[3][ACK.sender.id]

                t0 = ACK.c3_strongestToGWs[ACK.sender.id]
                t1 = ACK.c3_early1ToGWs[ACK.sender.id]
                t2 = ACK.c3_early2ToGWs[ACK.sender.id]
                t3 = ACK.c3_freeDemToGWs[ACK.sender.id]
                if not c3:
                    self.ack_noDem += 1
                    if not c0:
                        self.ack_noDemStrongest += 1
                    elif not t3:
                        self.ack_noDemFreeDem += 1
                    elif t0 and (not t1):
                        self.ack_noDemEarly1 += 1
                    elif (not t0) and (not t2):
                        self.ack_noDemEarly2 += 1
                elif not c1 or not c2:
                    self.ack_demFail += 1
                    if not c1:
                        self.ack_demFailC1 += 1
                    if not c2:
                        self.ack_demFailC2 += 1
                else:
                    self.ack_demSuccess += 1
        else:
            if len(fail) > 0:
                self.up_demFail += 1

                c1 = []
                c2 = []
                for j in fail:
                    if not uplink.conditionsToGWs[1][j]:
                        c1.append(j)
                    if not uplink.conditionsToGWs[2][j]:
                        c2.append(j)
                if len(c1) > 0:
                    self.up_demFailC1 += 1
                if len(c2) > 0:
                    self.up_demFailC2 += 1

            if len(up_noDem) > 0:
                self.up_noDem += 1
                c0 = []
                t1 = []
                t2 = []
                t3 = []
                for j in up_noDem:
                    if not uplink.conditionsToGWs[0][j]:
                        c0.append(j)
                    elif not uplink.c3_freeDemToGWs[j]:
                        t3.append(j)
                    elif uplink.c3_strongestToGWs[j] and not uplink.c3_early1ToGWs[j]:
                        t1.append(j)
                    elif (
                        not uplink.c3_strongestToGWs[j] and not uplink.c3_early2ToGWs[j]
                    ):
                        t2.append(j)
                if len(c0) > 0:
                    self.up_noDemStrongest += 1
                if len(t1) > 0:
                    self.up_noDemEarly1 += 1
                if len(t2) > 0:
                    self.up_noDemEarly2 += 1
                if len(t3) > 0:
                    self.up_noDemFreeDem += 1

    def request_demodulator(self, pac: Packet):
        self.loraEnv.sending_packets_4symbol[pac.tx_channel][pac.tx_sf].remove(pac)

        for id in range(self.loraEnv.GW_num):

            # 计算t1,t2和t3
            for p in self.loraEnv.sending_packets[pac.tx_channel][pac.tx_sf]:
                if (p is pac) or (isinstance(p.sender, GW) and p.sender.id == id):
                    continue

                if not is_same_channel and not isinstance(p.sender, EN):
                    continue

                if pac.timepoint_tx >= p.timepoint_4symbol:
                    pac.c3_early1ToGWs[id] = False

                if pac.timepoint_4symbol >= p.timepoint_tx:
                    pac.c3_early2ToGWs[id] = False

                if (not pac.c3_early1ToGWs[id]) and (not pac.c3_early2ToGWs[id]):
                    break

            if self.loraEnv.GWs[id].demodulatorNum > 0:
                pac.c3_freeDemToGWs[id] = True
            else:
                pac.c3_freeDemToGWs[id] = False

            # 判断是否占用解调器
            if not pac.conditionsToGWs[0][id]:
                pac.conditionsToGWs[3][id] = False
            elif pac.c3_freeDemToGWs[id]:
                if pac.c3_strongestToGWs[id] and pac.c3_early1ToGWs[id]:
                    pac.conditionsToGWs[3][id] = True
                    self.loraEnv.GWs[id].demodulatorNum -= 1
                    self.loraEnv.GWs[id].demodulatingPacs.append(pac)
                    self.occupiedDemodulator.append(id)
                elif (not pac.c3_strongestToGWs[id]) and pac.c3_early2ToGWs[id]:
                    pac.conditionsToGWs[3][id] = True
                    self.loraEnv.GWs[id].demodulatorNum -= 1
                    self.loraEnv.GWs[id].demodulatingPacs.append(pac)
                    self.occupiedDemodulator.append(id)
                else:
                    pac.conditionsToGWs[3][id] = False
            else:
                pac.conditionsToGWs[3][id] = False

    def release_demodulator(self, pac: Packet):
        for gw in self.loraEnv.GWs:
            if pac.conditionsToGWs[3][gw.id]:
                gw.demodulatorNum += 1
                gw.demodulatingPacs.remove(pac)
                self.occupiedDemodulator.remove(gw.id)

    # 接收机噪声
    def updateNoise(self, env: sp.Environment):

        while not self.loraEnv.stop:
            noise = self.rng_nosie.normal(
                0, np.sqrt(np.power(10, (-174 + 6) / 10) * 125000)
            )
            if noise < 0:
                noise = 0
            self.receiver_noise = noise
            yield esp.extendTimeout(
                env,
                self.loraEnv.ChannelGain_changesfrequency,
                'EN{}'.format(self.id),
                'updateNoise',
                '',
            )

    def updateChannelGain(self, env: sp.Environment):
        while not self.loraEnv.stop:
            channelGain = self.rng_channelGain.exponential(1)
            if channelGain > 1:
                channelGain = 1

            self.channelGain = channelGain
            self.calculateRssi()
            yield esp.extendTimeout(
                env,
                self.loraEnv.ChannelGain_changesfrequency,
                'EN',
                'updateChannelGain',
                '',
            )

    def calculateRssi(self):
        env: LoRaEnv = self.loraEnv

        for i in range(env.EN_num + env.GW_num):
            env.rssi[self._id][i] = (
                self.tx_pow * self.channelGain * env.pathloss[self._id][i]
            )


class GW:
    def __init__(self, id, x, y, env: sp.Environment, loraEnv):
        self.id = id
        self._id = (
            id + loraEnv.EN_num
        )  # 此_id用来检索距离表,GW和EN的_id不会出现相同的情况
        self.x = x
        self.y = y
        self.env = env
        self.loraEnv = loraEnv

        self.rng_base = np.random.default_rng(self._id + loraEnv.param['seed'])
        self.rng_nosie = np.random.default_rng(self.rng_base.integers(0, 2**63))
        self.rng_channelGain = np.random.default_rng(self.rng_base.integers(0, 2**63))

        self.tx_pow = loraEnv.txpower_max

        self.logTable = pt.PrettyTable()
        self.logTable.set_style(pt.SINGLE_BORDER)
        self.logFilename = self.loraEnv.root + 'log_GW/GW{}.txt'.format(id)
        self.logFile = openFile(self.logFilename, 'w')

        # 有节点发送完毕
        self.event_uplink_end = esp.extendEvent(
            env,
            'GW{}'.format(self.id),
            'uplink_end',
            '{}({})'.format(osfun(__file__), sysfun().f_lineno),
        )

        self.event_downlink_start = None

        self.ACKs = []

        self.receiver_noise = 0


        # 噪声为方差
        self.receiver_noise = np.power(10, (-174 + 6) / 10) * 125000

        # 使用相互独立的信道增益
        self.channelGain = 1
        self.process_channelGain = env.process(self.updateChannelGain(env))

        self.process = env.process(self.rxing(env))

        # demWorkCount = up_demSuccess + up_demFail
        self.demWorkCount = 0
        self.up_demSuccess = 0
        self.up_demFail = 0
        self.up_noDem = 0

        # 被解调成功的报文,其ACK的情况
        self.ack_demSuccess = 0
        self.ack_demFail = 0
        self.ack_noDem = 0
        self.ack_notMe = 0
        self.ack_miss = 0
        self.ack_noACK = 0

        self.ack_allocate = 0
        self.ack_sent = 0

        # ack_demFail <= ack_demFailC1 + ack_demFailC2
        self.ack_demFailC1 = 0
        self.ack_demFailC2 = 0

        # ack_noDem = ack_noDemStrongest + ack_noDemEarly1 + ack_noDemEarly2 + ack_noDemFreeDem
        self.ack_noDemStrongest = 0
        self.ack_noDemEarly1 = 0
        self.ack_noDemEarly2 = 0
        self.ack_noDemFreeDem = 0

        # up_demFail <= up_demFailC1 + up_demFailC2
        self.up_demFailC1 = 0
        self.up_demFailC2 = 0

        # up_noDem = up_noDemStrongest + up_noDemEarly1 + up_noDemEarly2 + up_noDemFreeDem
        self.up_noDemStrongest = 0
        self.up_noDemEarly1 = 0
        self.up_noDemEarly2 = 0
        self.up_noDemFreeDem = 0

        # 占用的解调器
        self.occupiedDemodulator = []

        # 解调器的数量
        self.demodulatorNum = 8
        self.demodulatingPacs = []  # 正在被解调的报文

        # 8个解调器的总工作时间
        self.demodulatorWorkTime_free = 0  # 空闲
        self.demodulatorWorkTime_demSuccess = 0  # 解调上行报文成功
        self.demodulatorWorkTime_demFail = 0  # 解调上行报文失败
        self.demodulatorWorkTime_free_ratio = 0
        self.demodulatorWorkTime_demSuccess_ratio = 0
        self.demodulatorWorkTime_demFail_ratio = 0

    def rxing(self, env: sp.Environment):
        try:
            log1 = []
            log2 = []

            def logAppend(p1, p2):
                if len(log1) > 0 and p1 == log1[len(log1) - 1]:
                    p1 = ''
                log1.append(p1)
                log2.append(p2)

            while not self.loraEnv.stop:
                log1.clear()
                log2.clear()
                uplink = yield self.event_uplink_end
                c1 = uplink.conditionsToGWs[1][self.id]
                c2 = uplink.conditionsToGWs[2][self.id]
                c3 = uplink.conditionsToGWs[3][self.id]

                if c3:

                    if c1 and c2:
                        self.demodulatorWorkTime_demSuccess += uplink.TOA
                    else:
                        self.demodulatorWorkTime_demFail += uplink.TOA

                self.statDetail_RX(uplink)
        except Exception as e:
            self.logFile.close()
            raise e

    def sendACK(self, env: sp.Environment):
        try:
            self.ack_sent += 1
            # self.ack_miss = self.ack_allocate - self.ack_sent
            log1 = []
            log2 = []

            def logAppend(p1, p2):
                if len(log1) > 0 and p1 == log1[len(log1) - 1]:
                    p1 = ''
                log1.append(p1)
                log2.append(p2)

            ACK: Packet = self.popACK()
            if ACK.timepoint_tx < env.now:
                if (
                    env.now >= ACK.timepoint_tx_rx1 + ACK.rx_size
                    and env.now < ACK.timepoint_tx_rx2
                ):
                    # 当前时间为D2
                    ACK.updateRX('rx2')
                    self.addACK(ACK)
                    return
                elif env.now >= ACK.timepoint_tx_rx2 + ACK.rx_size:
                    # 当前时间为W2之后
                    self.ack_miss += 1
                    self.updateEvent()
                    return
                else:
                    # 当前时间为W1或w2
                    ACK.timepoint_tx = env.now

            self.calculateRssi(ACK)
            send_pre(ACK, self.loraEnv)


            # 开始前4个符号
            if ACK.conditionsToGWs[0][self.id]:
                esp.extendInterruption(
                    ACK.receiver.process,
                    'GW{}->EN{}'.format(self.id, ACK.receiver.id),
                    'ack_start',
                    ACK,
                )
            yield esp.extendTimeout(
                env,
                ACK.TOA_4SymbolTime,
                'GW{}->EN{}'.format(self.id, ACK.receiver.id),
                'TX4',
                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
            )

            # 申请EN端的解调器
            self.request_demodulator(ACK)
            esp.extendSucceed(env, ACK.receiver.event_ACK_4symbol, 0)
            ACK.receiver.event_ACK_4symbol = esp.extendEvent(
                env,
                'EN{}'.format(ACK.receiver.id),
                'receive ACK 4symbol end',
                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
            )

            yield esp.extendTimeout(
                env,
                ACK.TOA_rest,
                'GW{}->EN{}'.format(self.id, ACK.receiver.id),
                'TX_rest',
                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
            )
            send_post(ACK, self.loraEnv)
            self.release_demodulator(ACK)
            esp.extendSucceed(env, ACK.receiver.event_ACK_end, 0)
            ACK.receiver.event_ACK_end = esp.extendEvent(
                env,
                'EN{}'.format(ACK.receiver.id),
                'receive ACK end',
                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
            )

            self.updateEvent()
            self.statDetail_TX(ACK)  # 更新历史数据
        except Exception as e:
            self.logFile.close()
            raise e

    def logPrint(self, log1: list, log2: list, logFile):
        self.logTable.clear()
        self.logTable.field_names = log1
        self.logTable.add_row(log2)
        logFile.write(self.logTable.__str__() + '\n')


    def judgePrint(self, logType, lc):
        shouldPrint = lc[1][0] != 3 and (
            (lc[1][0] == 0)
            or (lc[1][0] == 1 and self.env.now >= lc[1][1] and self.env.now <= lc[1][2])
            or (
                lc[1][0] == 2
                and self.up_sentNum >= lc[1][1]
                and self.up_sentNum <= lc[1][2]
            )
        )

        if logType == 'TX':
            return lc[0][0] and shouldPrint
        elif logType == 'RX':
            return lc[0][1] and shouldPrint
        elif logType == 'detail':
            return lc[0][2] and shouldPrint

    def request_demodulator(self, pac: Packet):

        self.loraEnv.sending_packets_4symbol[pac.tx_channel][pac.tx_sf].remove(pac)
        if not pac.conditionsToGWs[0][self.id]:
            pac.conditionsToGWs[3][self.id] = False
            return

        # 计算t1,t2和t3
        for p in self.loraEnv.sending_packets[pac.tx_channel][pac.tx_sf]:
            if (p is pac) or (
                isinstance(p.sender, EN) and p.sender.id == pac.receiver.id
            ):
                continue

            if not is_same_channel and not isinstance(p.sender, GW):
                continue

            if pac.timepoint_tx >= p.timepoint_4symbol:
                pac.c3_early1ToGWs[self.id] = False

            if pac.timepoint_4symbol >= p.timepoint_tx:
                pac.c3_early2ToGWs[self.id] = False

            if (not pac.c3_early1ToGWs[self.id]) and (not pac.c3_early2ToGWs[self.id]):
                break

        if pac.receiver.demodulatorNum > 0:
            pac.c3_freeDemToGWs[self.id] = True
        else:
            pac.c3_freeDemToGWs[self.id] = False

        # 判断是否占用解调器
        if pac.c3_freeDemToGWs[self.id]:
            if pac.c3_strongestToGWs[self.id] and pac.c3_early1ToGWs[self.id]:
                pac.conditionsToGWs[3][self.id] = True
                pac.receiver.demodulatorNum -= 1
                pac.receiver.demodulatingPacs.append(pac)
                self.occupiedDemodulator.append(pac.receiver.id)
            elif (not pac.c3_strongestToGWs[self.id]) and pac.c3_early2ToGWs[self.id]:
                pac.conditionsToGWs[3][self.id] = True
                pac.receiver.demodulatorNum -= 1
                pac.receiver.demodulatingPacs.append(pac)
                self.occupiedDemodulator.append(pac.receiver.id)
            else:
                pac.conditionsToGWs[3][self.id] = False
        else:
            pac.conditionsToGWs[3][self.id] = False


    def release_demodulator(self, pac: Packet):
        if pac.conditionsToGWs[3][self.id]:
            pac.receiver.demodulatorNum += 1
            pac.receiver.demodulatingPacs.remove(pac)
            self.occupiedDemodulator.remove(pac.receiver.id)

    # 调用时机:有新ACK加入,有ACK的timepoint_tx变化
    def sortACKs(self):
        # TOA -> timepoint_tx
        def policy1(ACK: Packet):
            return ACK.TOA

        def policy2(ACK: Packet):
            return ACK.timepoint_tx

        self.ACKs.sort(key=policy2)

    def addACK(self, ACK: Packet):
        self.ack_allocate += 1
        self.ACKs.append(ACK)
        self.loraEnv.waiting_ACK[ACK.receiver.id] = self.id
        self.sortACKs()
        self.updateEvent()

    def removeACK(self, en_id, packetNum):
        i = 0
        for ACK in self.ACKs:
            if ACK.sender.id == en_id and ACK.sender.packetNum == packetNum:
                break
            i += 1
        self.ACKs.pop(i)
        self.loraEnv.waiting_ACK[en_id] = 0
        self.ack_miss += 1
        self.sortACKs()
        self.updateEvent()

    def popACK(self):
        res = self.ACKs.pop(0)
        self.loraEnv.waiting_ACK[res.receiver.id] = 0
        return res

    # 当ACKs发生变化时,更新发送ACK的触发事件event_downlink_start
    def updateEvent(self):
        if self.event_downlink_start is not None:
            esp.deleteEvent(self.env, self.event_downlink_start)
            self.event_downlink_start = None
        if len(self.ACKs) > 0:
            self.event_downlink_start = esp.extendTimeout(
                self.env,
                (
                    self.ACKs[0].timepoint_tx - self.env.now
                    if self.ACKs[0].timepoint_tx > self.env.now
                    else 0
                ),
                'GW{}'.format(self.id),
                'ACK_tx_countdown',
                '{}({})'.format(osfun(__file__), sysfun().f_lineno),
            )

            def fun(event):
                event.env.process(self.sendACK(event.env))

            self.event_downlink_start.callbacks.append(fun)

 
    def statDetail_RX(self, uplink):
        c1 = uplink.conditionsToGWs[1][self.id]
        c2 = uplink.conditionsToGWs[2][self.id]
        c3 = uplink.conditionsToGWs[3][self.id]
        if c3:
            self.demWorkCount += 1
            if c1 and c2:
                self.up_demSuccess += 1
            else:
                self.up_demFail += 1
                if not c1:
                    self.up_demFailC1 += 1
                if not c2:
                    self.up_demFailC2 += 1
        else:
            self.up_noDem += 1
            if not uplink.conditionsToGWs[0][self.id]:
                self.up_noDemStrongest += 1
            elif not uplink.c3_freeDemToGWs[self.id]:
                self.up_noDemFreeDem += 1
            elif (
                uplink.c3_strongestToGWs[self.id] and not uplink.c3_early1ToGWs[self.id]
            ):
                self.up_noDemEarly1 += 1
            elif (
                not uplink.c3_strongestToGWs[self.id]
                and not uplink.c3_early2ToGWs[self.id]
            ):
                self.up_noDemEarly2 += 1

    def statDetail_TX(self, ACK: Packet):
        if (
            ACK.conditionsToGWs[1][self.id]
            and ACK.conditionsToGWs[2][self.id]
            and ACK.conditionsToGWs[3][self.id]
        ):
            self.ack_demSuccess += 1
        else:
            if not ACK.conditionsToGWs[3][self.id]:
                self.ack_noDem += 1
                if not ACK.conditionsToGWs[0][self.id]:
                    self.ack_noDemStrongest += 1
                elif not ACK.c3_freeDemToGWs[self.id]:
                    self.ack_noDemFreeDem += 1
                elif ACK.c3_strongestToGWs[self.id] and not ACK.c3_early1ToGWs[self.id]:
                    self.ack_noDemEarly1 += 1
                elif (
                    not ACK.c3_strongestToGWs[self.id]
                    and not ACK.c3_early2ToGWs[self.id]
                ):
                    self.ack_noDemEarly2 += 1
            else:
                self.ack_demFail += 1
                if not ACK.conditionsToGWs[1][self.id]:
                    self.ack_demFailC1 += 1
                if not ACK.conditionsToGWs[2][self.id]:
                    self.ack_demFailC2 += 1


    # 接收机噪声
    def updateNoise(self, env: sp.Environment):
        while not self.loraEnv.stop:
            noise = self.rng_nosie.normal(
                0, np.sqrt(np.power(10, (-174 + 6) / 10) * 125000)
            )
            if noise < 0:
                noise = 0
            self.receiver_noise = noise
            # 每秒更新一次
            yield esp.extendTimeout(
                env,
                self.loraEnv.ChannelGain_changesfrequency,
                'GW{}'.format(self.id),
                'updateNoise',
                '',
            )

    def updateChannelGain(self, env: sp.Environment):
        while not self.loraEnv.stop:
            channelGain = self.rng_channelGain.exponential(1)
            if channelGain > 1:
                channelGain = 1

            self.channelGain = channelGain

            yield esp.extendTimeout(
                env,
                self.loraEnv.ChannelGain_changesfrequency,
                'GW',
                'updateChannelGain',
                '',
            )

    def calculateRssi(self, ACK: Packet):
        ACK.channelGain = self.channelGain  # 信道增益
        for i in range(self.loraEnv.EN_num + self.loraEnv.GW_num):
            self.loraEnv.rssi[self._id][i] = (
                ACK.tx_pow * ACK.channelGain * self.loraEnv.pathloss[self._id][i]
            )


class NS:
    def __init__(self, env: sp.Environment, loraEnv) -> None:
        self.env = env
        self.loraEnv = loraEnv
        self.strategy = loraEnv.mainStrategy()

        self.event_uplink_end = esp.extendEvent(
            env, 'NS', 'uplink_end', '{}({})'.format(osfun(__file__), sysfun().f_lineno)
        )

        env.process(self.working(env))

    def working(self, env: sp.Environment):
        while True:
            uplink = yield self.event_uplink_end
            dispatchResult = self.strategy.NS_dispatch(uplink)
            self.strategy.NS_updateACKs(uplink, dispatchResult)


class LoRaEnv:
    def __init__(
        self,
        radius,
        rho,
        EN_num,
        GW_num,
        strategyName,
        baseDir,
        mainStrategy,
        payload,
        runtime,
        isPaint,
        testID,
        param,
    ) -> None:
        self.env = sp.Environment()
        self.radius = radius  # 半径
        self.rho = rho
        self.EN_num = EN_num
        self.GW_num = GW_num
        self.lam = EN_num / (np.pi * np.square(radius))

        self.ENs: list[EN] = []  # 以id为索引
        self.GWs: list[GW] = []  # 以id为索引

        self.sleepLaunchWindow = 0
        self.initEnv()

        self.is_multiprocess = True  # 是否为多进程环境
        self.group_log: list = []

        self.strategyName = strategyName
        self.mainStrategy = mainStrategy
        self.payload = payload
        self.runtime = runtime
        self.isPaint = isPaint
        self.testID = testID
        self.stop = False

        self.param = param

        from . import strategy as st

        self.ChannelGain_changesfrequency = 1 * minute
        self.ReceiverNoise_changesfrequency = 1 * minute

        self.tqdmBar: tqdm = None


        self.rng_base = np.random.default_rng(EN_num + GW_num + self.param['seed'])
        self.rng_random_action = np.random.default_rng(self.rng_base.integers(0, 2**63))

        self.baseDir = '.{}'.format(baseDir)
        self.root = (
            self.baseDir
            + '/'
            + result_dir_name
            + '/{}{}/'.format(self.testID, self.strategyName)
        )

        # 打印实验参数
        self.parameterFile = openFile(self.root + 'test parameter.txt', 'w')
        print(
            'radius={} km\nrho={} num/min'.format(radius, rho), file=self.parameterFile
        )
        print('EN_num={} \nGW_num={}'.format(EN_num, GW_num), file=self.parameterFile)
        print(
            'lam={} num/km^2\nstrategyName={}'.format(self.lam, strategyName),
            file=self.parameterFile,
        )
        print(
            'payload={} bytes\nsimulated runtime={} min'.format(
                payload, runtime / minute
            ),
            file=self.parameterFile,
        )

        self.printFile = openFile(self.root + 'print.txt', 'w')

        self.successProbabilityFile = openFile(
            self.root + 'successProbability.txt', 'w'
        )

        self.txt_file = [
            self.printFile,
            self.successProbabilityFile,
        ]


        # 创建NS和GW
        self.ns: NS = NS(self.env, self)
        if GW_num == 1:
            self.GWs.append(GW(0, 0, 0, self.env, self))
            self.GW_xs = [0]
            self.GW_ys = [0]
        else:
            pass

        # 创建EN
        xs, ys = a1loc.generateENCoordinates(
            radius=radius, EN_num=EN_num, rng=self.rng_base
        )
        self.EN_xs = xs
        self.EN_ys = ys
        for i in range(EN_num):
            self.ENs.append(EN(i, xs[i], ys[i], rho, self.env, self))

        self.distance = np.zeros((EN_num + GW_num, EN_num + GW_num))
        self.pathloss = np.zeros((EN_num + GW_num, EN_num + GW_num))
        self.rssi = np.zeros((EN_num + GW_num, EN_num + GW_num))
        for en_i in self.ENs:
            for en_j in self.ENs:
                if en_i is en_j:
                    self.distance[en_i._id][en_j._id] = 0
                self.distance[en_i._id][en_j._id] = np.sqrt(
                    np.power(en_i.x - en_j.x, 2) + np.power(en_i.y - en_j.y, 2)
                )
            for gw_j in self.GWs:
                self.distance[en_i._id][gw_j._id] = np.sqrt(
                    np.power(en_i.x - gw_j.x, 2) + np.power(en_i.y - gw_j.y, 2)
                )
        for gw_i in self.GWs:
            for en_j in self.ENs:
                self.distance[gw_i._id][en_j._id] = np.sqrt(
                    np.power(gw_i.x - en_j.x, 2) + np.power(gw_i.y - en_j.y, 2)
                )
            for gw_j in self.GWs:
                if gw_i is gw_j:
                    self.distance[gw_i._id][gw_j._id] = 0
                self.distance[gw_i._id][gw_j._id] = np.sqrt(
                    np.power(gw_i.x - gw_j.x, 2) + np.power(gw_i.y - gw_j.y, 2)
                )
        for i in range(EN_num + GW_num):
            for j in range(EN_num + GW_num):
                if self.distance[i][j] == 0:
                    self.pathloss[i][j] = 1
                else:
                    self.pathloss[i][j] = a4pro.pathloss_calculate(self.distance[i][j])

        self.log_data_init()
        self.data = self.networkData(self)

        self.algorithm_runtime = 0
        self.environment_runtime = 0
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

        # 初始化结点
        for en in self.ENs:
            en.strategy.EN_init(en)

    # 初始化用于debug的日志信息
    def log_data_init(self):
        self.sf_change = {}
        self.record_action = {action: 0 for action in range(0, 7)}
        self.action_taked = {sf: 0 for sf in range(7, 13)}

    def global_action_init(self):
        for en in self.ENs:
            for i in range(self.actions_num):
                self.global_action[en._id, i] = self.actual_to_buffer[i](en)

    def action_to_actual(self, action):

        action_list = []
        space = self.action_space_size
        for size in self.action_space:
            action_list.append(int(action / (space / size)))
            action = action % (space / size)
            space /= size

        return action_list

    def actual_to_action(self, action_list: list):
        action = 0
        space = self.action_space_size
        for i, a in enumerate(action_list):
            action += a * (space / self.action_space[i])
            space /= self.action_space[i]
        return action

    def get_random_action(self, en: EN):
        action_list = []

        # 统计每个属性的节点个数
        tx_sf_num = {tx_sf: 0 for tx_sf in range(6)}
        tx_pow_lv_num = {tx_pow_lv: 0 for tx_pow_lv in range(8)}
        rx_sf_num = {rx_sf: 0 for rx_sf in range(6)}
        rx_size_num = {rx_size: 0 for rx_size in range(8)}

        for e in self.ENs:
            tx_sf_num[self.actual_to_buffer[0](e)] += 1
            tx_pow_lv_num[self.actual_to_buffer[1](e)] += 1
            rx_sf_num[self.actual_to_buffer[2](e)] += 1
            rx_size_num[self.actual_to_buffer[4](e)] += 1

        en_tx_sf = self.actual_to_buffer[0](en)
        en_pow_lv = self.actual_to_buffer[1](en)
        en_rx_sf = self.actual_to_buffer[2](en)
        en_rx_size = self.actual_to_buffer[4](en)

        if self.actions_num >= 1:
            action_list.append(self.random_sf(en_tx_sf, tx_sf_num))

        if self.actions_num >= 2:
            action_list.append(self.random_pow_lv(en_pow_lv, tx_pow_lv_num))

        if self.actions_num >= 3:
            action_list.append(self.random_sf(en_rx_sf, rx_sf_num))

        if self.actions_num >= 4:
            action_list.append(self.random_rx_size(en_rx_size, rx_size_num))

        return th.tensor(action_list, device=self.device, dtype=th.int64)

    def random_sf(self, en_sf, global_sf_dis):
        average = self.EN_num / 6
        s = int(average * 0.1)
        if global_sf_dis[en_sf] > average + s:
            # 将该节点分配到节点数量最少的sf
            min_num = self.EN_num
            min_sf = 0
            for sf, num in global_sf_dis.items():
                if num < min_num:
                    min_num = num
                    min_sf = sf
            return min_sf
        elif global_sf_dis[en_sf] > average - s and global_sf_dis[en_sf] <= average + s:
            # 随机分配到其它<=average + s的sf,包括当前sf
            l = []
            for sf, num in global_sf_dis.items():
                if num <= average + s:
                    l.append(sf)
            return self.rng_random_action.choice(l)
        else:
            return en_sf

    def random_pow_lv(self, pow_lv, dis):
        alpha = 0.9
        prob = np.array([alpha**i for i in range(8)], dtype=np.float64)
        prob = (prob / prob.sum()) * self.EN_num

        s = prob[pow_lv] * 0.1
        if dis[pow_lv] > prob[pow_lv] + s:
            # 将该节点分配到节点数量最少的sf
            min_num = self.EN_num
            min_pow_lv = 0
            for p, num in dis.items():
                if num < min_num:
                    min_num = num
                    min_pow_lv = p
            return min_pow_lv
        elif dis[pow_lv] > prob[pow_lv] - s and dis[pow_lv] <= prob[pow_lv] + s:
            # 随机分配到其它<=average + s的sf,包括当前sf
            l = []
            for p, num in dis.items():
                if num <= prob[pow_lv] + s:
                    l.append(p)
            return self.rng_random_action.choice(l)
        else:
            return pow_lv

    def random_rx_size(self, rx_size, dis):
        alpha = 0.9
        prob = np.array([alpha**i for i in range(8)], dtype=np.float64)
        prob = (prob / prob.sum()) * self.EN_num

        s = prob[rx_size] * 0.1
        if dis[rx_size] > prob[rx_size] + s:
            # 将该节点分配到节点数量最少的sf
            min_num = self.EN_num
            min_rx_size = 0
            for p, num in dis.items():
                if num < min_num:
                    min_num = num
                    min_rx_size = p
            return min_rx_size
        elif dis[rx_size] > prob[rx_size] - s and dis[rx_size] <= prob[rx_size] + s:
            # 随机分配到其它<=average + s的sf,包括当前sf
            l = []
            for p, num in dis.items():
                if num <= prob[rx_size] + s:
                    l.append(p)
            return self.rng_random_action.choice(l)
        else:
            return rx_size

    def initEnv(self):

        # 8x6x0,8信道,6扩频因子,当前正在发送的报文
        self.sending_packets = [[[] for _ in range(6 + 7)] for _ in range(8)]

        # 8x6x0,8信道,6扩频因子,当前正在发送的报文,发送时加入,发送完4symbol后移除
        self.sending_packets_4symbol = [[[] for _ in range(6 + 7)] for _ in range(8)]

        # 当前正在发送的报文,不分信道和扩频因子
        self.sending_packets_one = []

        # 当前正在发送的报文,按照信道划分
        self.sending_packets_channel = [[] for _ in range(8)]

        # waiting_ACK[x] = y 表示GWy有准备发送给ENx的ACK
        # 在ENx关闭最后一个窗口后,将waiting_ACK[x] = 0
        self.waiting_ACK = np.zeros(self.EN_num)

        # LoRa Demodulator SNR
        # db->倍数
        self.SINR_threshold = {
            7: -7.5,
            8: -10.0,
            9: -12.5,
            10: -15.0,
            11: -17.5,
            12: -20.0,
        }
        for key in self.SINR_threshold:
            self.SINR_threshold[key] = pow(10, self.SINR_threshold[key] / 10)

        # sx1302,接收灵敏度(Table 3-6)
        # dbm->mw
        self.RSSI_GW_threshold = {
            7: -127,
            8: -129,
            9: -132.5,
            10: -135.5,
            11: -138,
            12: -141,
        }
        for key in self.RSSI_GW_threshold:
            self.RSSI_GW_threshold[key] = pow(10, self.RSSI_GW_threshold[key] / 10)

        # sx1276,接收灵敏度(page20 RFS_L125_HF)
        # dbm->mw
        self.RSSI_EN_threshold = {
            7: -123,
            8: -126,
            9: -129,
            10: -132,
            11: -133,
            12: -136,
        }
        for key in self.RSSI_EN_threshold:
            self.RSSI_EN_threshold[key] = pow(10, self.RSSI_EN_threshold[key] / 10)

        # 中国最大发射功率17dBm=50mw
        self.txpower_max = 50
        self.txpower_min = 6.25
        self.tx_pow_level = [self.txpower_max * db2abs(-i * 2) for i in range(8)]

        # 干扰系数
        self.interCoefficient = [
            [1, 0.104, 0.062, 0.041, 0.029, 0.021],
            [0.104, 1, 0.073, 0.043, 0.029, 0.020],
            [0.062, 0.073, 1, 0.052, 0.030, 0.020],
            [0.041, 0.043, 0.052, 1, 0.037, 0.021],
            [0.029, 0.029, 0.030, 0.037, 1, 0.026],
            [0.021, 0.020, 0.020, 0.021, 0.026, 1],
        ]

    # 运行时间runtime的单位为ms
    def runEnv(self):
        self.stopEvent = self.env.timeout(self.runtime)

        def stopFun(event):
            self.stop = True
            if hasattr(self.ns.strategy, 'model'):
                self.ns.strategy.model.end_print()

        self.stopEvent.callbacks.append(stopFun)

        self.env.run()
        self.runtime = self.env.now

        if self.have_pbar:
            self.pbar.close()


        if not self.is_multiprocess:
            for log in self.group_log:
                print(log)
            print()


        old_root = self.root[0:-1]

        # 修改文件名，将成功率加入到文件名中
        newname = self.modify_filename(str(self.testID) + self.strategyName)
        newname = newname.format(self.data.enData.up_demSuccess[0][-1])
        # 注意要去掉尾部的'/'
        new_root = self.baseDir + '/' + result_dir_name + '/' + newname

        # 重命名文件夹
        if os.name != 'nt':
            if os.path.exists(new_root):
                shutil.rmtree(new_root)
            os.rename(old_root, new_root)

        self.root = new_root + '/'

        self.close()

    def modify_filename(self, input_string):
        if '/' in input_string:
            return input_string + '_{:.1%}'
        else:
            pattern = re.compile(r'_')
            modified_string = pattern.sub(r'_{:.1%}_', input_string, 1)
        return modified_string

    # 终止运行
    def stopEnv(self):
        self.ns.strategy.model.end_print()

        esp.deleteEvent(self.env, self.stopEvent)


        self.stop = True







    # 关闭资源
    def close(self):
        for en in self.ENs:
            if en.logFile is not None:
                en.logFile.close()

        for gw in self.GWs:
            if gw.logFile is not None:
                gw.logFile.close()

        for f in self.txt_file:
            f.close()
