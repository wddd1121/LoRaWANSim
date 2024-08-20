import time
from . import LoRaEnvironment as le
import gc
import torch as th
import traceback
import os


def train(
    param: dict,
    final_succes_rate_dict: list,
    group_log: list,
    position: int,
    cuda_cfg: dict,
):
    start = time.perf_counter()
    try:
        if isinstance(cuda_cfg, dict):
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_cfg['CUDA_VISIBLE_DEVICES']
            th.cuda.set_per_process_memory_fraction(
                cuda_cfg['per_process_memory_fraction']
            )

        loraEnv = le.LoRaEnv(
            radius=5,
            rho=param['rho'],
            EN_num=param['en_num'],
            GW_num=1,
            strategyName=param['count'],
            baseDir='',
            mainStrategy=param['strategy'],
            payload=20,
            runtime=param['runtime'],
            isPaint=True,
            testID=param['ID'],
            param=param,
        )
        loraEnv.group_log = group_log
        loraEnv.position = position
        if th.cuda.is_available():
            th.cuda.reset_peak_memory_stats(loraEnv.device)
        loraEnv.runEnv()
    except Exception as e:
        # print('Error: %s' % (province), traceback.print_exc())
        traceback.print_exc()
        raise e
    finally:
        # 记录实际运行时间
        actualRunTime = time.perf_counter() - start  # 单位为秒
        print(
            'actual runtime={:.2} min'.format(actualRunTime / 60),
            file=loraEnv.parameterFile,
        )
        print(
            '    environment runtime={:.2} min'.format(
                (actualRunTime - loraEnv.algorithm_runtime) / 60
            ),
            file=loraEnv.parameterFile,
        )
        print(
            '    algorithm runtime={:.2} min'.format(loraEnv.algorithm_runtime / 60),
            file=loraEnv.parameterFile,
        )
        print(
            param,
            file=loraEnv.parameterFile,
        )

        loraEnv.parameterFile.close()
        if param['log_target'] not in final_succes_rate_dict:
            final_succes_rate_dict[param['log_target']] = []

        t = final_succes_rate_dict[param['log_target']]
        t.append(loraEnv.final_succes_rate)
        final_succes_rate_dict[param['log_target']] = t

        del loraEnv
        gc.collect()
        th.cuda.empty_cache()
