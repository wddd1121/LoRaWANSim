from typing import Any, Optional
import simpy as sp
from simpy.core import Environment
from simpy.events import (
    Process,
    ProcessGenerator,
    Interruption,
    PENDING,
    EventPriority,
    NORMAL,
    URGENT,
)
from heapq import heapify


class extendTimeout(sp.Timeout):
    def __init__(
        self,
        env: sp.Environment,
        delay: sp.core.SimTime,
        device,
        action,
        lineNum,
        value: Optional[Any] = None,
    ):
        super().__init__(env, delay, value)
        self.device = device
        self.action = action
        self.lineNum = lineNum

    def _desc(self) -> str:
        return self.device + ' {}({}) '.format(self.action, self._delay) + self.lineNum


class extendEvent(sp.Event):
    def __init__(self, env: Environment, device, action, lineNum):
        super().__init__(env)
        self.device = device
        self.action = action
        self.lineNum = lineNum

    def _desc(self) -> str:
        return self.device + ' {} '.format(self.action) + self.lineNum


class extendInterruption(Interruption):
    def __init__(self, process: Process, device, action, cause: Optional[Any] = None):
        super().__init__(process, cause)
        self.device = device
        self.action = action

    def _desc(self) -> str:
        return self.device + ' {}'.format(self.action)


def extendSucceed(
    env: sp.Environment,
    event: sp.Event,
    priority: int = 1,
    value: Optional[Any] = None,
) -> sp.Event:
    if event._value is not PENDING:
        raise RuntimeError(f'{event} has already been triggered')

    event._ok = True
    event._value = value
    if priority == 0:
        env.schedule(event, URGENT)
    elif priority == 1:
        env.schedule(event, NORMAL)
    return event

# 从环境中删除某个事件
def deleteEvent(env:sp.Environment, eventToDelete):
    i = 0
    for _, _, _, event in env._queue:
        if event is eventToDelete:
            env._queue.pop(i)
            heapify(env._queue)
            return
        i += 1