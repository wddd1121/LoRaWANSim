from typing import Any
from . import LoRaEnvironment as le
import numpy as np
from collections import deque
from . import extendSimpy as esp
from abc import ABC, abstractmethod


class strategyBase:
    # 初始化设备参数
    def EN_init(self, en: le.EN):

        # 按照等宽环法初始化
        bounds = {sf: (sf - 6) / 6 * en.loraEnv.radius for sf in range(7, 13)}
        en.tx_sf = None
        for sf in range(7, 12):
            if en.loraEnv.distance[en._id, en.loraEnv.EN_num] > bounds[sf]:
                en.tx_sf = sf + 1
        if en.tx_sf is None:
            en.tx_sf = 7

        en.tx_pow_lv = 0
        en.tx_pow = en.loraEnv.tx_pow_level[en.tx_pow_lv]

        en.tx_channel = en.rng_channel.integers(0, 8)

        en.rx1_sf = en.tx_sf
        en.rx1_channel = en.tx_channel  # rx1_channe必须等于tx_channel
        en.rx2_sf = en.tx_sf
        en.rx2_channel = en.tx_channel  # rx2_channel不一定等于tx_channel

        en.rx1_delay = 1000
        en.rx2_delay = 1000  # rx2打开延迟(毫秒),此处为在rx1结束后开始计时
        en.rx_size = (
            500  # rx窗口打开大小,(毫秒),最小取决于下行链路的前导码数量,最大为1000毫秒
        )
        e: le.LoRaEnv = en.loraEnv

        e.sf_change[en._id] = [en.tx_sf]


    # 初始化报文参数
    def EN_initUplink(self, en: le.EN):
        pass

    # 调度算法的EN部分
    def EN_dispatch(self, en: le.EN, ACK: le.Packet, ACK_Status):
        pass

    def NS_updateACKs(self, uplink: le.Packet, dispatchResult):

        qualified_gws = []
        for gw_id in range(uplink.loraEnv.GW_num):
            if (
                uplink.conditionsToGWs[1][gw_id]
                and uplink.conditionsToGWs[2][gw_id]
                and uplink.conditionsToGWs[3][gw_id]
            ):
                qualified_gws.append(gw_id)

        if len(qualified_gws) <= 0:
            return
        if dispatchResult is None:
            for gw_id in qualified_gws:
                uplink.loraEnv.GWs[gw_id].ack_noACK += 1
            return

        def fun(g):
            return uplink.sinrToGWs[g]

        gw_id = max(qualified_gws, key=fun)

        gw: le.GW = uplink.loraEnv.GWs[gw_id]

        ACK = le.Packet(gw, uplink.sender, uplink.env, uplink.loraEnv)

        self.NS_initACK(uplink, gw, ACK)

        if dispatchResult is None:
            ACK.payload = 0
            ACK.ADRACK = False
        else:
            ACK.payload = dispatchResult.payload
            ACK.ADRACK = True

        ACK.updateRX('rx1')  # 默认选择rx1
        ACK.dispatchResult = dispatchResult
        gw.addACK(ACK)

    # 调度算法的NS部分
    def NS_dispatch(self, uplink: le.Packet):
        pass

    # 初始化ACK
    def NS_initACK(self, uplink: le.Packet, gw: le.GW, ACK: le.Packet):
        ACK.tx_pow_lv = 0
        ACK.tx_pow = uplink.loraEnv.tx_pow_level[0]
        ACK.packetNum = uplink.packetNum

        ACK.interferenceToGWs.fill(0)
        ACK.conditionsToGWs.fill(True)
        ACK.c3_strongestToGWs.fill(True)
        ACK.c3_early1ToGWs.fill(True)
        ACK.c3_early2ToGWs.fill(True)
        ACK.c3_freeDemToGWs.fill(True)

        ACK.rx_size = uplink.rx_size

        # 在rx1下发
        ACK.tx_sf_rx1 = uplink.rx1_sf
        ACK.tx_channel_rx1 = uplink.rx1_channel
        ACK.timepoint_tx_rx1 = uplink.env.now + uplink.rx1_delay

        # 在rx2下发
        ACK.tx_sf_rx2 = uplink.rx2_sf
        ACK.tx_channel_rx2 = uplink.rx2_channel
        ACK.timepoint_tx_rx2 = uplink.env.now + uplink.rx1_delay + uplink.rx2_delay

    def EN_getPayload(self, en: le.EN):
        return en.loraEnv.payload


class strategyADRx(strategyBase):

    def __init__(self):
        super().__init__()
        self.ADR_data: list[deque] = None
        self.SINR_threshold = {
            7: -7.5,
            8: -10.0,
            9: -12.5,
            10: -15.0,
            11: -17.5,
            12: -20.0,
        }

        self.ADR_ACK_CNT = 0
        self.ADR_ACK_LIMIT = 64
        self.ADR_ACK_DELAY = 32

    # 初始化报文参数
    def EN_initUplink(self, en: le.EN):

        en.pac.tx_sf = en.tx_sf
        en.pac.tx_pow = en.tx_pow
        en.pac.tx_pow_lv = en.tx_pow_lv

        en.pac.tx_channel = en.tx_channel

        en.pac.rx1_sf = en.rx1_sf
        en.pac.rx1_channel = en.rx1_channel
        en.pac.rx2_sf = en.rx2_sf
        en.pac.rx2_channel = en.rx2_channel

        en.pac.rx1_delay = en.rx1_delay

        en.pac.payload = self.EN_getPayload(en)

        en.pac.rx_size = en.rx_size
        en.pac.rx2_delay = en.rx2_delay
        en.pac.receiver = None
        en.pac.packetNum = en.up_sentNum
        en.pac.channelGain = en.channelGain  # 信道增益
        en.pac.TOA = le.a3toa.TOA(en.tx_sf, en.pac.payload)  # 完整的空中时间
        en.pac.TOA_4SymbolTime = le.a3toa.symbolTime(
            en.tx_sf, 4
        )  # 计算前4个symbol的时间
        en.pac.TOA_rest = en.pac.TOA - en.pac.TOA_4SymbolTime  # 剩余TOA时间


        en.pac.interferenceToGWs.fill(0)
        en.pac.conditionsToGWs.fill(True)
        en.pac.c3_strongestToGWs.fill(True)
        en.pac.c3_early1ToGWs.fill(True)
        en.pac.c3_early2ToGWs.fill(True)
        en.pac.c3_freeDemToGWs.fill(True)

        en.pac.ADRACK = False
        self.ADR_ACK_CNT += 1
        if self.ADR_ACK_CNT >= self.ADR_ACK_LIMIT:
            en.pac.ADRACK = True

    # 应用ADR算法的EN部分
    def EN_dispatch(self, en: le.EN, ACK: le.Packet, ACK_Status):

        res = None

        if (
            self.ADR_ACK_CNT >= self.ADR_ACK_LIMIT
            and self.ADR_ACK_CNT < self.ADR_ACK_LIMIT + self.ADR_ACK_DELAY
        ):
            if ACK_Status == 0 and ACK.dispatchResult is not None:
                ACK.dispatchResult(en)
                self.ADR_ACK_CNT = 0
                res = ACK.dispatchResult.logs
        elif self.ADR_ACK_CNT >= self.ADR_ACK_LIMIT + self.ADR_ACK_DELAY:
            self.ADR_ACK_CNT = 0
            if en.tx_sf < 12:
                en.tx_sf += 1
                res = [
                    ['ADR', ''],
                    ['EN part', 'SF{}->SF{}'.format(en.tx_sf - 1, en.tx_sf)],
                ]
            else:
                res = [['ADR', ''], ['EN part', 'do nothing']]

        en.tx_channel = en.rng_channel.integers(0, 8)
        en.rx1_channel = en.tx_channel
        en.rx2_channel = en.tx_channel
        en.rx1_sf = en.tx_sf
        en.rx2_sf = en.tx_sf

        return res

    class ADRresult:
        def __init__(self, tx_sf_gw, tx_pow_gw, payload) -> None:
            self.tx_sf_gw = tx_sf_gw
            self.tx_pow_gw = tx_pow_gw
            self.payload = payload
            self.strategyName = 'ADR'
            self.logs = [[], []]
            self.logs[0].append('ADR')
            self.logs[1].append('GW part')

        def __call__(self, en: le.EN, *args: Any, **kwds: Any) -> Any:
            self.logs[0].append('SF{}->SF{}'.format(en.tx_sf, self.tx_sf_gw))
            self.logs[1].append(
                '{}mw->{:.2f}*{}mw'.format(
                    en.tx_pow, le.db2abs(self.tx_pow_gw), en.tx_pow
                )
            )
            en.tx_sf = self.tx_sf_gw
            en.tx_pow = le.db2abs(self.tx_pow_gw) * en.tx_pow
            if en.tx_pow > en.loraEnv.txpower_max:
                en.tx_pow = en.loraEnv.txpower_max
            elif en.tx_pow < en.loraEnv.txpower_min:
                en.tx_pow = en.loraEnv.txpower_min

        def getDetail(self):
            return self.logs

    def NS_dispatch(self, uplink: le.Packet):
        e: le.LoRaEnv = uplink.loraEnv

        if self.ADR_data is None:
            self.ADR_data: list[deque] = [None] * uplink.loraEnv.EN_num
            for i in range(len(self.ADR_data)):
                self.ADR_data[i] = deque()
            self.margin_dbs = [5 for _ in range(e.EN_num)]
            self.margin_db_update_count = [0 for _ in range(e.EN_num)]
            self.N = e.param['ARDx_N']

        GtwDiversity = 0
        SINRmax = -1
        for gw_id in range(uplink.loraEnv.GW_num):
            if (
                uplink.conditionsToGWs[1][gw_id]
                and uplink.conditionsToGWs[2][gw_id]
                and uplink.conditionsToGWs[3][gw_id]
            ):
                GtwDiversity += 1
                if uplink.sinrToGWs[gw_id] > SINRmax:
                    SINRmax = uplink.sinrToGWs[gw_id]

        if GtwDiversity > 0:
            # sinr转换为db
            self.ADR_data[uplink.sender.id].append(
                [uplink.packetNum, le.abs2db(SINRmax), GtwDiversity]
            )
            if len(self.ADR_data[uplink.sender.id]) > self.N:
                self.ADR_data[uplink.sender.id].popleft()

        if (
            GtwDiversity > 0
            and len(self.ADR_data[uplink.sender.id]) >= self.N
            and uplink.ADRACK
        ):
            en_id = uplink.sender.id
            self.margin_db_update_count[en_id] += 1
            self.update_margin_db(en_id, e)

            SINRm = -np.inf
            for _, sinr, _ in self.ADR_data[uplink.sender.id]:
                if sinr > SINRm:
                    SINRm = sinr

            if SINRm == np.inf:
                NStep = 1
            else:
                SINRmargin = (
                    SINRm - self.SINR_threshold[uplink.tx_sf] - self.margin_dbs[en_id]
                )
                NStep = int(SINRmargin / 3)

            tx_sf_gw = uplink.tx_sf
            tx_pow_gw = 0  # 单位为db

            while NStep != 0:
                if NStep > 0:
                    if tx_sf_gw > 7:
                        tx_sf_gw -= 1
                    else:
                        tx_pow_gw -= 3
                    NStep -= 1
                    if (
                        le.db2abs(tx_pow_gw) * uplink.tx_pow
                        <= uplink.loraEnv.txpower_min
                    ):
                        break
                else:
                    if (
                        le.db2abs(tx_pow_gw) * uplink.tx_pow
                        >= uplink.loraEnv.txpower_max
                    ):
                        break
                    tx_pow_gw += 3
                    NStep += 1

            return self.ADRresult(tx_sf_gw, tx_pow_gw, 5)
        return None

    def update_margin_db(self, en_id, e: le.LoRaEnv):
        if self.margin_db_update_count[en_id] < self.N:
            return
        self.margin_db_update_count[en_id] = 0
        oldest = self.ADR_data[en_id][0]
        newest = self.ADR_data[en_id][-1]
        DER_inst = self.N / (newest[0] - oldest[0])
        DER_ref = e.param['ARDx_DER_ref']
        if DER_inst < DER_ref and self.margin_dbs[en_id] < 30:
            self.margin_dbs[en_id] += 5
        elif DER_inst > 1.15 * DER_ref and self.margin_dbs[en_id] > 5:
            self.margin_dbs[en_id] -= 2.5
