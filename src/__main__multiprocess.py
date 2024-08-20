from . import LoRaEnvironment as le
from . import strategy as st
import itertools

import numpy as np
import os
from multiprocessing import Pool, Manager

from .other import train
import traceback

from multiprocessing import Pool, RLock, freeze_support
from tqdm import tqdm
import re


class field:
    def __init__(self, format, data) -> None:
        self.format = format
        self.data = data


train_times = np.array([45], dtype=np.int64)
runtimes = field('{}min', train_times)  # 模拟时间，分钟


ID = 0
strategys = field('{}', [st.strategyADRx])
infos = field('{}', ['adrx'])


ARDx_DER_refs = field('DER_ref({})', [0.5])
ARDx_Ns = field('N({})', [64])


EN_num_list = [512]


EN_num = field('{}EN', EN_num_list)

Seeds = field('Seed({})', [s for s in range(1)])


rhos_list = [2]
rhos = field('rho{:.2f}', rhos_list)  # 每分钟的发射次数


normal = True  # 正常训练
always_average_policy = False  # 一直使用均匀策略
not_update_network = False  # 不更新网络


if ID < 10:
    ID = '0' + str(ID)
else:
    ID = str(ID)


def get_configs(
    always_average_policy,
    is_update_network=True,
):

    config = [
        strategys,
        infos,
        EN_num,
        runtimes,
        rhos,
        ARDx_DER_refs,
        ARDx_Ns,
        Seeds,
    ]

    fields_one = [infos]
    fields_multiple = []

    # 选出config中只有一个参数
    for f in config:
        if f is not strategys:
            if len(f.data) > 1:
                fields_multiple.append(f)
            else:
                # fields_one.append(f)
                pass  # 一个参数的不打印出来

    # 参数相同的部分,生成最后的目录名
    dir_name_shared = ''
    for f in fields_one:
        dir_name_shared += '_' + f.format.format(f.data[0])

    dir_name_list = []
    log_targets = []

    if len(fields_multiple) > 0:
        dir_name_shared += '/'

        # 参数不同的部分的format之和
        format_sum = ''
        for f in fields_multiple:
            format_sum += '_' + f.format

        # 填充format_sum
        fields_multiple_data = [f.data for f in fields_multiple]
        cartesian_product_1 = list(itertools.product(*fields_multiple_data))

        c = len(cartesian_product_1)
        fm = '{:0' + str(len(str(c))) + 'd}'
        for i, items in enumerate(cartesian_product_1):
            p = format_sum.format(*items)
            # 使用正则表达式去除模式 _Seed(*)
            log_target = re.sub(r'Seed\(\d+\)', '', p)
            log_targets.append(log_target)
            dir_name_list.append(dir_name_shared + fm.format(i) + p)
    else:
        log_targets.append('0')
        dir_name_list.append(dir_name_shared + '/0')

    fields_data = [f.data for f in config]
    cartesian_product_2 = list(itertools.product(*fields_data))

    configs = []
    for iterate_count, (
        strategy,
        info,
        en_num,
        ti,
        rho,
        ARDx_DER_ref,
        ARDx_N,
        seed,
    ) in enumerate(cartesian_product_2):
        param = {
            'ID': ID,
            'en_num': en_num,
            'count': dir_name_list[iterate_count],
            'log_target': log_targets[iterate_count],
            'strategy': strategy,
            'runtime': ti * le.minute,
            'always_average_policy': always_average_policy,
            'is_update_network': is_update_network,
            'seed': seed,
            'rho': rho,
            'ARDx_DER_ref': ARDx_DER_ref,
            'ARDx_N': ARDx_N,
        }
        configs.append(param)

    return configs, dir_name_list, dir_name_shared


def pool_run_group(cfgs, train, group_size, cuda_cfgs, final_succes_rate_dict):
    try:
        # 使用Manager来创建一个共享的列表
        manager = Manager()

        # 多进程日志
        group_log = [manager.list() for _ in range(group_size)]

        freeze_support()

        # processes:允许开几个进程,设置进程数等于一个组内的任务数量
        po = Pool(processes=group_size, initializer=tqdm.set_lock, initargs=(RLock(),))
        for position, cfg in enumerate(cfgs):
            po.apply_async(
                train,
                args=(
                    cfg,
                    final_succes_rate_dict,
                    group_log[position],
                    position,
                    cuda_cfgs[position],
                ),
            )
        po.close()  # 关闭进程池入口，此后不能再向进程池中添加任务了
        po.join()  # 阻塞等待，只有进程池中所有任务(即所有cfgs)都完成了才往下执行

        # 打印多进程日志
        for log in group_log:
            for l in log:
                print(l)
            print('----------')
        print()

    except Exception as e:
        print("主进程捕获到异常，终止所有进程")
        po.terminate()  # 终止所有进程
        po.join()  # 确保所有进程终止
        print("所有进程已终止")
        traceback.print_exc()
        raise e


def pool_run(cfgs, dir_name_list, dir_name_shared, train):
    groups = []
    if os.name != 'nt':

        cuda_cfgs = []
        gpu_num = 1
        precess_per_gpu = 1
        memory_ratio = round(1 / precess_per_gpu, 3)
        for device_id in range(gpu_num):
            for _ in range(precess_per_gpu):
                cuda_cfgs.append(
                    {
                        'CUDA_VISIBLE_DEVICES': str(device_id),
                        'per_process_memory_fraction': memory_ratio,
                    }
                )

        # cuda_cfgs = [
        #     {'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'per_process_memory_fraction': 1.0},
        #     {'CUDA_VISIBLE_DEVICES': '4,5,6,7', 'per_process_memory_fraction': 1.0},
        # ]

        group_size = len(cuda_cfgs)

        # liunx cpu
        # group_size = 64
        # cuda_cfgs = [None for _ in range(group_size)]
    else:
        group_size = 1
        cuda_cfgs = [None for _ in range(group_size)]

    for i in range(0, len(cfgs), group_size):
        groups.append(cfgs[i : i + group_size])

    # 使用Manager来创建一个共享的列表
    manager = Manager()
    final_succes_rate_dict = manager.dict()  # 初始化共享列表

    for group in groups:
        pool_run_group(
            group, train, group_size, cuda_cfgs[: len(group)], final_succes_rate_dict
        )

    if len(dir_name_list) > 1:
        baseDir = (
            './'
            + le.result_dir_name
            + '/{}{}/'.format(ID, dir_name_shared.split('/')[0])
        )
        for key, value in final_succes_rate_dict.items():
            follow = 'succ={:.2%} std={:.2%} key={}.txt'.format(
                np.mean(value), np.std(value), key
            )
            le.openFile(baseDir + follow, 'w').close()


def do():
    old_name = infos.data[0]
    if normal:
        pool_run(*get_configs(False, True), train)

    if not_update_network:
        infos.data[0] = 'NU_' + old_name
        pool_run(*get_configs(False, False), train)

    if always_average_policy:
        infos.data[0] = 'AVG_' + old_name
        pool_run(*get_configs(True, True), train)
