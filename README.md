# LoRaSim

An open source LoRaWAN simulator. This simulator not only simulates the LoRa link behavior, but also implements uplink and downlink communication, receive windowing, and multi-channel multi-demodulator gateways.The simulator is an implementation of the following paper.



Hong Yang, Haibo Luo*, Chao Feng. "HEAT: Optimizing LoRaWAN via Comprehensive Modeling and A Deep Reinforcement Learning Resource Allocation Strategy."



### Requirements

* Python 3.9.18
* simpy 4.1.1
* kaleido 0.2.1
* prettytable 3.9.0
* palettable 3.3.0
* matplotlib 3.7.2
* numpy 1.26.0
* pandas 2.1.4
  
  
  
  

# HEAT: Optimizing LoRaWAN via Comprehensive Modeling and A Deep Reinforcement Learning Resource Allocation Strategy

### Abstract

Long Range Wide Area Network (LoRaWAN), a Low Power Wide Area Network (LPWAN) technology, is extensively studied and applied, but struggles with bandwidth limitations as device numbers increase. This necessitates optimized resource allocation strategies to enhance network performance. The current methods typically focus solely on the parameters of the uplink, such as the uplink spreading factor and transmission power, while overlooking the significant impact that downlink settings, including the downlink spreading factor and receive window size, have on overall network performance. This paper introduces the History-Enhanced dual-phase Actor-critic algorithm with a unified Transformer (HEAT) to address these challenges. HEAT, a Deep Reinforcement Learning (DRL) algorithm, optimizes both uplink and downlink parameters, employing offline and online learning phases to continually adapt and improve based on both historical data and real-time interactions. To evaluate HEAT's effectiveness, we enhanced existing LoRa/LoRaWAN simulators to create an open-source LoRaWAN simulation tool, LoRaWANSim, which conducts comprehensive simulations of Long Range (LoRa) link behaviors and the core functions of LoRaWAN protocol. Comparative experiments across various device densities and traffic intensities show that HEAT improves Packet Delivery Rate (PDR) and Energy Efficiency Ratio (EER) by at least 12% and 95% respectively, surpassing established methods.



### Requirements

* Python 3.9.18
* simpy 4.1.1
* kaleido 0.2.1
* prettytable 3.9.0
* palettable 3.3.0
* matplotlib 3.7.2
* numpy 1.26.0
* pandas 2.1.4

# 
