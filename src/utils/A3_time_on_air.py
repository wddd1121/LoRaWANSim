import numpy as np
import matplotlib.pyplot as plt

# MHDR，MIC，FHDR的长度
other = 1 + 4 + 7
# lorawan标准1.0.0,sf对应的最长FRMpayload
sf2maxpayload = {
    7: 250 - 7,
    8: 250 - 7,
    9: 123 - 7,
    10: 59 - 7,
    11: 59 - 7,
    12: 59 - 7,
}

# 使用CN470频段的论文
# AdapLoRa：资源适应以最大化 LoRa 网络中的网络生命周期

# 使用CN470频段，则SF12的有效字节数只有1字节
# 符合发射时间不超过1秒
sf2maxpayload1 = {
    7: 250 - 7,
    8: 250 - 7,
    9: 123 - 7,
    10: 59 - 7,
    11: 34,
    12: 2,
}

# 若使用占空比为1%,则节点发送数据包的频率lam不能超过占空比


# 返回单位为毫秒
def TOA(SF, FRMPayload):
    BW = 125000
    IH = 0
    CR = 1
    CRC = 0
    preamble = 8

    PHYPayload = other + FRMPayload

    Ts = np.power(2, SF) / BW
    preamble_all = preamble + 4.25
    payload = 8 + max(
        np.ceil((8 * PHYPayload - 4 * SF + float(28) + 16 * CRC - 20 * IH) / (4 * SF))
        * (CR + 4),
        0,
    )

    t1 = preamble_all * Ts
    t2 = payload * Ts
    return int((t1 + t2) * 1000)

def symbolTime(SF, symbolNum):
    BW = 125000
    Ts = np.power(2, SF) / BW
    return int(symbolNum * Ts * 1000)


if __name__ == "__main__":
    for sf in range(7, 13):
        # print(TOA(sf, sf2maxpayload1[sf])/1000)
        print(TOA(sf, 30)/1000)

