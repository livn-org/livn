import pickle

import gymnasium as gym
import numpy as np

from livn.decoding import Pipe
from livn.env.gymnasium import GymStep, ObsAugmentation
from livn.types import Decoding, Encoding


class DummyEncoding(Encoding):
    scale: float = 1.0

    def __call__(self, env, t_end, inputs):
        return None


class DummyDecoding(Decoding):
    duration: int = 100
    alpha: float = 0.5

    def __call__(self, env, it, tt, iv, vv, im, mp):
        return None


class TestEncodingDecodingHash:
    def test_identical_objects_same_hash(self):
        a = DummyEncoding(scale=2.0)
        b = DummyEncoding(scale=2.0)
        assert hash(a) == hash(b)
        assert a == b

    def test_different_fields_different_hash(self):
        a = DummyEncoding(scale=1.0)
        b = DummyEncoding(scale=2.0)
        assert hash(a) != hash(b)
        assert a != b

    def test_decoding_hash_eq(self):
        a = DummyDecoding(duration=100, alpha=0.5)
        b = DummyDecoding(duration=100, alpha=0.5)
        assert hash(a) == hash(b)
        assert a == b

    def test_decoding_different(self):
        a = DummyDecoding(duration=100, alpha=0.5)
        b = DummyDecoding(duration=100, alpha=0.9)
        assert a != b

    def test_cross_type_not_equal(self):
        a = DummyEncoding(scale=1.0)
        b = DummyDecoding(duration=100, alpha=0.5)
        result = a.__eq__(b)
        assert result is NotImplemented

    def test_usable_as_dict_key(self):
        enc = DummyEncoding(scale=3.0)
        dec = DummyDecoding(duration=50, alpha=1.0)
        cache = {}
        cache[(dec, enc)] = 42
        assert (
            cache[(DummyDecoding(duration=50, alpha=1.0), DummyEncoding(scale=3.0))]
            == 42
        )


class TestPipeGetstate:
    def test_state_excluded_from_pickle(self):
        pipe = Pipe(stages=[], duration=100)
        pipe._state["io_action"] = np.array([1.0, 2.0])
        pipe._state["raw_gym_obs"] = np.array([0.5])
        pipe._context["spike_counts"] = {0: 10}

        h1 = hash(pipe)

        pipe2 = Pipe(stages=[], duration=100)
        h2 = hash(pipe2)

        assert h1 == h2, "_state and _context should not affect hash"

    def test_structural_change_invalidates_hash(self):
        pipe1 = Pipe(stages=[], duration=100)
        pipe2 = Pipe(stages=[], duration=200)
        assert hash(pipe1) != hash(pipe2)

    def test_pickle_roundtrip_preserves_structure(self):
        pipe = Pipe(stages=[], duration=100)
        pipe._state["foo"] = "bar"

        data = pickle.dumps(pipe)
        restored = pickle.loads(data)

        assert restored.duration == 100
        # _state should be empty after roundtrip (excluded by __getstate__)
        assert restored._state == {}
        assert restored._context == {}


class TestGymStep:
    def test_same_env_spec_same_hash(self):
        g1 = GymStep(gym.make("CartPole-v1"))
        g2 = GymStep(gym.make("CartPole-v1"))
        assert hash(g1) == hash(g2)
        assert g1 == g2

    def test_different_env_spec_different_hash(self):
        g1 = GymStep(gym.make("CartPole-v1"))
        g2 = GymStep(gym.make("MountainCar-v0"))
        assert hash(g1) != hash(g2)
        assert g1 != g2

    def test_pickle_roundtrip(self):
        g = GymStep(gym.make("CartPole-v1"))

        g.gym_env.reset()
        g.gym_env.step(0)

        data = pickle.dumps(g)
        restored = pickle.loads(data)

        assert hasattr(restored, "gym_env")
        assert restored.gym_env.spec.id == "CartPole-v1"

        assert hash(g) == hash(restored)

    def test_hash_stable_across_steps(self):
        g = GymStep(gym.make("CartPole-v1"))
        g.gym_env.reset()
        h1 = hash(g)
        g.gym_env.step(0)
        h2 = hash(g)
        assert h1 == h2, "GymStep hash should not change with episode state"



class ConcreteAug(ObsAugmentation):
    obs_dim: int = 4

    def _features(self, env, context, state):
        return np.zeros(self.obs_dim, dtype=np.float32)


class TestObsAugmentation:
    def test_same_config_same_hash(self):
        a = ConcreteAug(obs_dim=4)
        b = ConcreteAug(obs_dim=4)
        assert hash(a) == hash(b)
        assert a == b

    def test_different_config_different_hash(self):
        a = ConcreteAug(obs_dim=4)
        b = ConcreteAug(obs_dim=8)
        assert hash(a) != hash(b)
        assert a != b



class TestCultureActivityExtractor:
    def test_same_config_same_hash(self):
        from benchmarks.rl.decoding import CultureActivityExtractor

        a = CultureActivityExtractor(n_channels=16, duration=100)
        b = CultureActivityExtractor(n_channels=16, duration=100)
        assert hash(a) == hash(b)
        assert a == b

    def test_different_config_different_hash(self):
        from benchmarks.rl.decoding import CultureActivityExtractor

        a = CultureActivityExtractor(n_channels=16, duration=100)
        b = CultureActivityExtractor(n_channels=8, duration=100)
        assert hash(a) != hash(b)
        assert a != b


class TestSpikeDecodingGetstate:
    def test_last_spike_counts_excluded(self):
        from benchmarks.rl.decoding import ArgmaxDecoding

        dec = ArgmaxDecoding(duration=100, groups=((0,), (1,)))

        dec._last_spike_counts = {0: 5, 1: 10}

        h1 = hash(dec)

        dec2 = ArgmaxDecoding(duration=100, groups=((0,), (1,)))
        h2 = hash(dec2)

        assert h1 == h2, "_last_spike_counts should not affect hash"




class TestPipeWithGymStep:
    def test_pipe_hash_stable_across_gym_steps(self):
        cart = gym.make("CartPole-v1")
        pipe = Pipe(stages=[GymStep(cart)], duration=100)
        h1 = hash(pipe)

        # mutate gym env internal state
        cart.reset()
        cart.step(0)

        h2 = hash(pipe)
        assert h1 == h2, "Pipe hash should be stable despite gym env state changes"

    def test_different_gym_env_different_pipe_hash(self):
        pipe1 = Pipe(stages=[GymStep(gym.make("CartPole-v1"))], duration=100)
        pipe2 = Pipe(stages=[GymStep(gym.make("MountainCar-v0"))], duration=100)
        assert hash(pipe1) != hash(pipe2)


class TestPipelineCall:
    def test_install_and_step(self):
        from livn.env.distributed import _PipelineResult, _pipeline_call, _state

        call_log = []

        class FakeEnv:
            def __call__(self, dec, inputs, enc, **kwargs):
                call_log.append((dec, inputs, enc, kwargs))
                return "result"

        _state["env"] = FakeEnv()
        _state["pipelines"] = {}

        dec = DummyDecoding(duration=100)
        enc = DummyEncoding(scale=1.0)

        # first call for install
        pr = _pipeline_call(
            0,
            "action_input",
            {},
            {},
            decoding=dec,
            encoding=enc,
        )
        assert isinstance(pr, _PipelineResult)
        assert pr.result == "result"
        assert (0) in _state["pipelines"]
        assert len(call_log) == 1

        # second call for cache hit (no decoding/encoding args)
        pr2 = _pipeline_call(
            0,
            "action_input_2",
            {},
            {},
        )
        assert isinstance(pr2, _PipelineResult)
        assert pr2.result == "result"
        assert len(call_log) == 2

        _state.pop("pipelines", None)
        _state.pop("env", None)

    def test_needs_reset_flag(self):
        from livn.env.distributed import _pipeline_call, _state

        reset_called = []

        class FakeEnv:
            def __call__(self, dec, inputs, enc, **kwargs):
                return "ok"

        class ResetTrackingStage:
            def reset(self, **kwargs):
                reset_called.append(True)
                return None

            def __call__(self, env, *data):
                return None

        pipe = Pipe(stages=[ResetTrackingStage()], duration=100)

        _state["env"] = FakeEnv()
        _state["pipelines"] = {0: (pipe, None)}

        _pipeline_call(0, None, {"_needs_reset": True}, {})
        assert len(reset_called) == 1

        _state.pop("pipelines", None)
        _state.pop("env", None)
