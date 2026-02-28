import pickle
import signal
import time
import traceback
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, Optional, Self

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Intracomm

from livn.env import Env
from livn.io import IO
from livn.types import Array, Env as EnvProtocol, Float, Int, Model

if TYPE_CHECKING:
    from livn.stimulus import Stimulus

world_comm = MPI.COMM_WORLD
size = world_comm.size
rank = world_comm.rank
# NEURON subworlds start from 0 so make controller remainder rank
controller_rank = size - 1

is_controller = rank == controller_rank
is_worker = not is_controller
n_workers = size - 1
start_time = time.time()

_state: dict = {}


class DistributedEnv(EnvProtocol):
    """Env wrapper that fans out simulation calls to MPI workers"""

    def __init__(
        self,
        system: str,
        model: Optional[Model] = None,
        io: Optional[IO] = None,
        seed: int | None = 123,
        comm: Optional[MPI.Intracomm] = None,
        subworld_size: int | None = None,
    ):
        self.controller: MPIController | None = None
        self._result_buffer: dict[int, object] = {}
        self._last_probe_time: float = 0.0

        if not isinstance(system, str):
            raise ValueError(
                "System must be a directory path to allow re-initialization on workers"
            )

        self._system_uri = system
        self._model_arg = model
        self._io_arg = io

        self._local_system: _ControllerSystem | None = None
        self._local_model: Model | None = None
        self._local_io: IO | None = None

        self.seed = seed
        self.comm = comm
        self.subworld_size = subworld_size

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_local_system"] = None
        d["_local_model"] = None
        d["_local_io"] = None
        d["controller"] = None
        d["_result_buffer"] = {}
        return d

    @property
    def system(self) -> "_ControllerSystem":
        if self._local_system is None:
            self._local_system = _ControllerSystem(self._system_uri)
        return self._local_system

    @property
    def model(self) -> Model:
        if self._local_model is None:
            if self._model_arg is not None:
                self._local_model = self._model_arg
            else:
                self._local_model = self.system.default_model(comm=False)
        return self._local_model

    @property
    def io(self) -> IO:
        if self._local_io is None:
            if self._io_arg is not None:
                self._local_io = self._io_arg
            else:
                self._local_io = self.system.default_io(comm=False)
        return self._local_io

    def is_root(self):
        return is_controller

    def init(self):
        args = (self,)

        _mpi_run(
            controller_fn=_controller_init,
            worker_fn=_worker_init,
            nprocs_per_worker=self.subworld_size,
            args=args,
            auto_exit=False,
        )

        return self

    def shutdown(self):
        if self.controller is not None:
            expected = self.controller.comm.size - 1
            for _ in range(10):
                self.controller.process()
                if len(self.controller.active_workers) >= expected:
                    break
                time.sleep(0.2)
            self.controller.exit()

    def _broadcast_to_workers(self, method_name: str, args: tuple) -> None:
        if self.controller is None:
            return
        n_workers = self.controller.comm.size - 1
        task_ids = self.controller.submit_multiple(
            _env_config_call,
            args_list=[(method_name, a) for a in [args] * n_workers],
        )
        for _ in range(len(task_ids)):
            self.controller.get_next_result()

    def clear_recordings(self) -> Self:
        self._broadcast_to_workers("clear_recordings", ())
        return self

    def apply_model_defaults(self, weights: bool = True, noise: bool = True) -> Self:
        self._broadcast_to_workers("apply_model_defaults", (weights, noise))
        return self

    def set_weights(self, weights: dict) -> Self:
        self._broadcast_to_workers("set_weights", (weights,))
        return self

    def set_noise(self, noise: dict) -> Self:
        self._broadcast_to_workers("set_noise", (noise,))
        return self

    def enable_plasticity(self, config: dict | None = None) -> Self:
        self._broadcast_to_workers("enable_plasticity", (config,))
        return self

    def disable_plasticity(self) -> Self:
        self._broadcast_to_workers("disable_plasticity", ())
        return self

    def record_spikes(self, population: str | list | tuple | None = None) -> Self:
        self._broadcast_to_workers("record_spikes", (population,))
        return self

    def record_voltage(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ) -> Self:
        self._broadcast_to_workers("record_voltage", (population, dt))
        return self

    def record_membrane_current(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ) -> Self:
        self._broadcast_to_workers("record_membrane_current", (population, dt))
        return self

    def cell_stimulus(
        self,
        channel_inputs: Float[Array, "batch timestep n_channels"],
    ) -> Float[Array, "batch timestep n_gids"]:
        raise NotImplementedError(
            "Please implement the cell stimulus as part of the encoding in __call__"
        )

    def channel_recording(
        self,
        ii: Float[Array, "i"],
        *recordings: Float[Array, "_"],
    ) -> tuple[dict[int, Array], ...]:
        raise NotImplementedError(
            "Please implement the channel recording as part of the decoding in __call__"
        )

    def run(
        self,
        duration,
        stimulus: Optional["Stimulus"] = None,
        dt: float = 0.025,
        **kwargs,
    ) -> tuple[
        Int[Array, "n_spiking_neuron_ids"] | None,
        Float[Array, "n_spiking_neuron_times"] | None,
        Int[Array, "n_voltage_neuron_ids"] | None,
        Float[Array, "neuron_ids voltages"] | None,
        Int[Array, "n_membrane_current_neuron_ids"] | None,
        Float[Array, "neuron_ids membrane_currents"] | None,
    ]:
        raise NotImplementedError("Please use __call__ instead")

    def __call__(self, decoding, inputs=None, encoding=None, **kwargs):
        if self.controller is None:
            return

        if inputs is None:
            inputs = [None]

        num_inputs = 0
        for i in inputs:
            self.submit_call(decoding, i, encoding, **kwargs)
            num_inputs += 1

        recordings = []
        for _ in range(num_inputs):
            response = self.receive_response()
            if len(response) == 1:
                recordings.append(response[0])
            else:
                recordings.append(response)

        return recordings

    def submit_call(self, decoding, inputs=None, encoding=None, **kwargs) -> int:
        """Submit an env call for async execution on a worker."""
        return self.controller.submit_call(
            _envcall,
            args=(decoding, inputs, encoding, kwargs),
        )

    def receive_response(self):
        """Block until the next result is available."""
        _task_id, response = self.controller.get_next_result()
        return response

    def probe_response(self, task_id: int):
        """Non-blocking poll for a specific task's result."""
        if self.controller is None:
            return None

        if task_id not in self._result_buffer:
            now = time.monotonic()
            if now - self._last_probe_time >= 0.001:
                for tid, result in self.controller.probe_all_next_results():
                    self._result_buffer[tid] = result
                self._last_probe_time = now

        if task_id not in self._result_buffer:
            return None

        raw = self._result_buffer.pop(task_id)
        return raw[0] if len(raw) == 1 else raw

    def potential_recording(
        self, membrane_currents: Float[Array, "timestep n_neurons"] | None
    ) -> Float[Array, "timestep n_channels"]:
        if _state["env"] is None:
            return
        _state["env"].potential_recording(membrane_currents)

    def clear(self) -> Self:
        self._broadcast_to_workers("clear", ())
        return self

    def set_params(self, params) -> Self:
        self._broadcast_to_workers("set_params", (params,))
        return self


class MessageTag(IntEnum):
    READY = 0
    DONE = 1
    TASK = 2
    EXIT = 3


class MPIController:
    def __init__(self, comm: Intracomm, time_limit: Any = None) -> None:
        n = comm.size
        self.comm = comm
        self.workers_available = n > 1
        self.start_time = start_time
        self.time_limit = time_limit

        self._next_id = 0
        self._buf = bytearray(1 << 20)

        self.active_workers: set[int] = set()
        self._idle: list[int] = []
        self._idle_data: dict[int, Any] = {}
        self._cost = np.ones(n)
        self._cost[0] = np.inf

        # lifecycle: queued -> dispatched -> finished
        self._queued_ids: list[int] = []
        self._queued: dict[int, tuple[Callable, tuple, dict, int]] = {}
        self._dispatched: dict[int, int] = {}  # task id -> worker rank
        self._per_worker: list[list[int]] = [[] for _ in range(n)]
        self._finished: dict[int, Any] = {}
        self._finished_order: list[int] = []

        self.n_processed = np.zeros(n, dtype=int)
        self.total_time = np.zeros(n, dtype=np.float32)
        self.stats: list[dict] = []

    def process(self, limit: int = 1000, block: bool = False) -> list[int]:
        if not self.workers_available:
            return []

        status = MPI.Status()
        n_recv = 0

        if block:
            self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
            if limit is not None and n_recv >= limit:
                break

            src = status.Get_source()
            tag = status.Get_tag()
            nbytes = status.Get_count(MPI.BYTE)

            if nbytes > len(self._buf):
                self._buf = bytearray(nbytes)

            self.comm.Recv([self._buf, nbytes, MPI.BYTE], source=src, tag=tag)

            if tag == MessageTag.READY:
                if src not in self._idle:
                    data = pickle.loads(self._buf[:nbytes]) if nbytes else None
                    self._idle.append(src)
                    self._idle_data[src] = data
                    self.active_workers.add(src)

            elif tag == MessageTag.DONE:
                task_id, result, stat = pickle.loads(self._buf[:nbytes])
                self._finished[task_id] = result
                self._finished_order.append(task_id)
                self._dispatched.pop(task_id, None)
                self._per_worker[src] = [
                    t for t in self._per_worker[src] if t != task_id
                ]
                if stat:
                    self.stats.append(stat)
                    self.n_processed[src] = stat["n_processed"]
                    self.total_time[src] = stat["total_time"]
                n_recv += 1

            else:
                raise RuntimeError(f"Controller: unexpected message tag {tag}")

        return self._flush_pending()

    def submit_call(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        time_est: int = 1,
        task_id: int | None = None,
    ) -> int:
        if kwargs is None:
            kwargs = {}
        if task_id is None:
            task_id = self._next_id
            self._next_id += 1
        self._assert_unused(task_id)

        self.process()
        if self._idle:
            self._dispatch(task_id, fn, args, kwargs, time_est)
        else:
            self._enqueue(task_id, fn, args, kwargs, time_est)

        return task_id

    def submit_multiple(
        self,
        fn: Callable,
        args_list: list[tuple],
        kwargs_list: list[dict] | None = None,
        time_est: int = 1,
    ) -> list[int]:
        n = len(args_list)
        if kwargs_list is None:
            kwargs_list = [{} for _ in range(n)]

        self.process()
        ids: list[int] = []
        for a, kw in zip(args_list, kwargs_list):
            tid = self._next_id
            self._next_id += 1
            self._assert_unused(tid)
            self._enqueue(tid, fn, a, kw, time_est)
            ids.append(tid)
        return ids

    def get_next_result(self) -> tuple[int, Any] | None:
        self.process()
        if self._finished_order:
            tid = self._finished_order.pop(0)
            return tid, self._finished.pop(tid)
        if self._dispatched or self._queued:
            while not self._finished_order:
                self.process(block=True)
            tid = self._finished_order.pop(0)
            return tid, self._finished.pop(tid)
        return None

    def probe_all_next_results(self) -> list[tuple[int, Any]]:
        self.process()
        out: list[tuple[int, Any]] = []
        while self._finished_order:
            tid = self._finished_order.pop(0)
            out.append((tid, self._finished.pop(tid)))
        return out

    def exit(self) -> None:
        if not self.workers_available:
            return
        while self.get_next_result() is not None:
            pass
        reqs = [
            self.comm.isend(None, dest=w, tag=MessageTag.EXIT)
            for w in self.active_workers
        ]
        MPI.Request.Waitall(reqs)

    def abort(self) -> None:
        traceback.print_exc()
        self.comm.Abort()

    def _assert_unused(self, task_id: int) -> None:
        if task_id in self._finished:
            raise RuntimeError(f"task {task_id}: result already stored")
        if task_id in self._dispatched:
            raise RuntimeError(f"task {task_id}: already dispatched")
        if task_id in self._queued:
            raise RuntimeError(f"task {task_id}: already queued")

    def _enqueue(
        self, tid: int, fn: Callable, args: tuple, kwargs: dict, est: int
    ) -> None:
        self._queued_ids.append(tid)
        self._queued[tid] = (fn, args, kwargs, est)

    def _select_worker(self) -> int:
        costs = np.array([self._cost[w] for w in self._idle])
        return self._idle[int(np.argmin(costs))]

    def _dispatch(
        self, tid: int, fn: Callable, args: tuple, kwargs: dict, est: int
    ) -> None:
        worker = self._select_worker()
        self.comm.isend(
            (fn, args, kwargs, est, tid), dest=worker, tag=MessageTag.TASK
        ).Wait()
        self._idle.remove(worker)
        self._idle_data.pop(worker, None)
        self._dispatched[tid] = worker
        self._per_worker[worker].append(tid)
        self._cost[worker] += est

    def _flush_pending(self) -> list[int]:
        sent: list[int] = []
        reqs: list[MPI.Request] = []
        while self._queued_ids and self._idle:
            tid = self._queued_ids.pop(0)
            fn, args, kwargs, est = self._queued.pop(tid)
            worker = self._select_worker()
            reqs.append(
                self.comm.isend(
                    (fn, args, kwargs, est, tid),
                    dest=worker,
                    tag=MessageTag.TASK,
                )
            )
            self._idle.remove(worker)
            self._idle_data.pop(worker, None)
            self._dispatched[tid] = worker
            self._per_worker[worker].append(tid)
            self._cost[worker] += est
            sent.append(tid)
        if reqs:
            MPI.Request.Waitall(reqs)
        return sent


class _Halt:
    """Sentinel shutdown signal"""


class MPICollectiveBroker:
    def __init__(
        self,
        worker_id: int,
        comm: Intracomm,
        group_comm: Intracomm,
        merged_comm: Intracomm,
        nprocs_per_worker: int,
        is_worker: bool = False,
    ) -> None:
        self.comm = comm
        self.group_comm = group_comm
        self.merged_comm = merged_comm
        self.worker_id = worker_id
        self.nprocs_per_worker = nprocs_per_worker
        self.is_worker = is_worker
        self.is_broker = True

        gs = merged_comm.Get_size()
        gr = merged_comm.Get_rank()
        self.n_processed = np.full(gs, np.nan)
        self.n_processed[gr] = 0
        self.total_time = np.full(gs, np.nan)
        self.total_time[gr] = 0.0

    def serve(self) -> None:
        my_rank = self.merged_comm.Get_rank()

        while True:
            buf = pickle.dumps(None)
            self.comm.Isend([buf, MPI.BYTE], dest=0, tag=MessageTag.READY).Wait()

            tag, payload = self._recv_from_controller()

            if tag == MessageTag.EXIT:
                self._scatter_to_group(_Halt, (), {}, 0, 0)
                break
            elif tag == MessageTag.TASK:
                fn, args, kwargs, est, task_id = payload
            else:
                raise RuntimeError(f"Broker {self.worker_id}: unexpected tag {tag}")

            self._scatter_to_group(fn, args, kwargs, est, task_id)

            if self.is_worker:
                t0 = time.time()
                local_result = fn(*args, **kwargs)
                elapsed = time.time() - t0
                self.n_processed[my_rank] += 1
                local_stat = {
                    "id": task_id,
                    "rank": my_rank,
                    "this_time": elapsed,
                    "time_over_est": elapsed / est if est else 0,
                    "n_processed": self.n_processed[my_rank],
                    "total_time": time.time() - start_time,
                }
            else:
                local_result, local_stat = None, None

            results, stats = self._gather_from_group(local_result, local_stat)

            rep = None
            if stats:
                times = [s["this_time"] for s in stats]
                rep = stats[int(np.argmax(times))]

            done_buf = pickle.dumps((task_id, results, rep))
            self.comm.Isend([done_buf, MPI.BYTE], dest=0, tag=MessageTag.DONE).Wait()

    def _recv_from_controller(self) -> tuple[int, Any]:
        status = MPI.Status()
        self.comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        data = self.comm.recv(source=0, tag=tag)
        return tag, data

    def _scatter_to_group(
        self, fn: Callable, args: tuple, kwargs: dict, est: int, task_id: int
    ) -> None:
        n = self.merged_comm.Get_size()
        root = self.merged_comm.Get_rank()
        self.merged_comm.scatter([(fn, args, kwargs, est, task_id)] * n, root=root)

    def _gather_from_group(
        self, local_result: Any, local_stat: Any
    ) -> tuple[list, list]:
        root = self.merged_comm.Get_rank()
        pairs = self.merged_comm.gather((local_result, local_stat), root=root)
        results = [r for r, _ in pairs if r is not None]
        stats = [s for _, s in pairs if s is not None]
        return results, stats


class MPICollectiveWorker:
    def __init__(
        self,
        local_comm: Intracomm,
        merged_comm: Intracomm,
        worker_id: int,
        n_workers: int,
    ) -> None:
        self.comm = local_comm
        self.merged_comm = merged_comm
        self.worker_id = worker_id
        self.n_workers = n_workers

        gs = merged_comm.Get_size()
        gr = merged_comm.Get_rank()
        self.n_processed = np.full(gs, np.nan)
        self.n_processed[gr] = 0
        self.total_time = np.full(gs, np.nan)
        self.total_time[gr] = 0.0

    def serve(self) -> None:
        my_rank = self.merged_comm.Get_rank()

        while True:
            fn, args, kwargs, est, task_id = self.merged_comm.scatter(None, root=0)

            if fn is _Halt:
                break

            t0 = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0

            self.n_processed[my_rank] += 1
            stat = {
                "id": task_id,
                "rank": my_rank,
                "this_time": elapsed,
                "time_over_est": elapsed / est if est else 0,
                "n_processed": self.n_processed[my_rank],
                "total_time": time.time() - start_time,
            }

            self.merged_comm.gather((result, stat), root=0)


def _mpi_run(
    controller_fn: Callable,
    worker_fn: Callable | None = None,
    nprocs_per_worker: int = 1,
    args: tuple = (),
    time_limit: int | None = None,
    auto_exit: bool = True,
) -> Any:
    assert nprocs_per_worker > 0

    if world_comm.size <= 1:
        worker_group = world_comm
    else:
        worker_group = world_comm.Split(2 if is_controller else 1, key=world_comm.rank)

    if is_controller:
        ctrl_comm = world_comm.Split(1, key=0)
        ctl = MPIController(ctrl_comm, time_limit=time_limit)
        signal.signal(signal.SIGINT, lambda *_: ctl.abort())
        ctrl_comm.Barrier()
        try:
            result = controller_fn(ctl, *args)
        except Exception:
            ctl.abort()
            raise
        if auto_exit:
            ctl.exit()
        return result

    n_groups = worker_group.size // nprocs_per_worker
    worker_id = (worker_group.rank // nprocs_per_worker) + 1
    broker_ranks = {g * nprocs_per_worker for g in range(n_groups)}
    am_broker = worker_group.rank in broker_ranks

    ctrl_comm = world_comm.Split(
        1 if am_broker else 2,
        key=0 if is_controller else 1,
    )

    first = (worker_id - 1) * nprocs_per_worker
    members = set(range(first, first + nprocs_per_worker))
    color = worker_id if worker_group.rank in members else MPI.UNDEFINED
    local_comm = worker_group.Split(color, key=worker_group.rank)
    sub_comm = local_comm.Dup()
    merged_comm = sub_comm.Dup()

    if am_broker:
        local_comm.Free()
        broker = MPICollectiveBroker(
            worker_id,
            ctrl_comm,
            worker_group,
            merged_comm,
            nprocs_per_worker,
            is_worker=True,
        )
        ctrl_comm.Barrier()

        if worker_fn is not None:
            merged_comm.bcast(args, root=0)
            worker_fn(broker, *args)

        broker.serve()
    else:
        worker = MPICollectiveWorker(local_comm, merged_comm, worker_id, n_groups)

        if worker_fn is not None:
            merged_comm.bcast(args, root=0)
            worker_fn(worker, *args)

        worker.serve()

    return None


class _ControllerSystem:
    _UNCACHED = frozenset({"neuron_coordinates", "gids"})

    def __init__(self, uri: str):
        from livn.system import System

        object.__setattr__(self, "_inner", System(uri, comm=MPI.COMM_SELF))

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        result = getattr(self._inner, name)
        if name in self._UNCACHED:
            self._inner._neuron_coordinates = None
        return result


def _envcall(decoding, inputs, encoding, kwargs):
    if kwargs is None:
        kwargs = {}
    return _state["env"](decoding, inputs, encoding, **kwargs)


def _env_config_call(method_name: str, args: tuple):
    env = _state.get("env")
    if env is None:
        return None
    getattr(env, method_name)(*args)


def _worker_init(worker, distributed_env: "DistributedEnv"):
    if distributed_env._system_uri is not None:
        worker_comm = getattr(worker, "merged_comm", worker.comm)

        env = Env(
            distributed_env._system_uri,
            distributed_env._model_arg,
            distributed_env._io_arg,
            distributed_env.seed,
            comm=worker_comm,
            subworld_size=distributed_env.subworld_size,
        )
        env.init()
        _state["env"] = env

    MPI.COMM_WORLD.Barrier()


def _controller_init(controller: MPIController, distributed_env: "DistributedEnv"):
    if distributed_env._system_uri is not None:
        # throw-away env to participate in NEURON's collective h.pc.subworlds
        Env(
            distributed_env._system_uri,
            distributed_env._model_arg,
            distributed_env._io_arg,
            distributed_env.seed,
            subworld_size=distributed_env.subworld_size,
        )
        _state["env"] = None

    MPI.COMM_WORLD.Barrier()

    distributed_env.controller = controller
