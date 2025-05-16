"""SNN

## License information

This module contains code adapted from the snnax project (https://github.com/cholberg/snnax).
The original copyright notice and the license under which the code was received is set forth below.

NOTICES AND INFORMATION BEGIN HERE
==================================

Copyright 2024 Christian Holberg

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

=========================================
END OF NOTICES AND INFORMATION
"""

import functools as ft
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optimistix as optx
from diffrax import (
    AbstractPath,
    BrownianIncrement,
    SpaceTimeLevyArea,
    VirtualBrownianTree,
)
from diffrax._brownian.tree import _levy_diff, _make_levy_val
from diffrax._custom_types import RealScalarLike, levy_tree_transpose
from diffrax._misc import linear_rescale
from jax._src.ad_util import stop_gradient_p  # pyright: ignore
from jax.interpreters import ad
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree, Real
from typing_extensions import TypeAlias


# Work around JAX issue #22011,
# as well as https://github.com/patrick-kidger/diffrax/pull/387#issuecomment-2174488365
def stop_gradient_transpose(ct, x):
    return (ct,)


ad.primitive_transposes[stop_gradient_p] = stop_gradient_transpose


_Spline: TypeAlias = Literal["sqrt", "quad", "zero"]


class Solution(eqx.Module):
    """Solution of the event-driven model."""

    t1: Real
    ys: Float[Array, "samples spikes neurons times 3"]
    ts: Float[Array, "samples spikes times"]
    spike_times: Float[Array, "samples spikes"]
    spike_marks: Float[Array, "samples spikes neurons"]
    num_spikes: int
    dt: Real


def plottable_paths(
    sol: Solution,
) -> Tuple[Real[Array, "samples times"], Float[Array, "samples neurons times 3"]]:
    """Takes an instance of `Solution` from `SpikingNeuralNet.__call__(...)` and outputs the times
    and values of the internal neuron states in a plottable format.

    **Arguments**:

    - `sol`: An instance of `Solution` as returned from `SpikingNeuralNet.__call__(...)`.

    **Returns**:

    - `ts`: The time axis of the path of shape `(samples, times)`.
    - `ys`: The values of the internal state of the neuron of shape `(samples, neurons, times, 3)`.
    """

    @jax.vmap
    def _plottable_neuron(ts, ys):
        t0 = ts[0, 0]
        t1 = sol.t1
        _, neurons, times, _ = ys.shape
        ys = ys.transpose((1, 0, 2, 3))
        ts_out = jnp.linspace(t0, t1, times)
        ts_flat = ts.flatten()
        ys_flat = ys.reshape((neurons, -1, 3))
        sort_idx = jnp.argsort(ts_flat)
        ts_flat = ts_flat[sort_idx]
        ys_flat = ys_flat[:, sort_idx, :]
        idx = jnp.searchsorted(ts_flat, ts_out)
        ys_out = ys_flat[:, idx, :]
        return ts_out, ys_out

    return _plottable_neuron(sol.ts, sol.ys)


def interleave(arr1: Array, arr2: Array) -> Array:
    out = jnp.empty((arr1.size + arr2.size,), dtype=arr1.dtype)
    out = out.at[0::2].set(arr2)
    out = out.at[1::2].set(arr1)
    return out


def marcus_lift(
    t0: RealScalarLike,
    t1: RealScalarLike,
    spike_times: Float[Array, " max_spikes"],
    spike_mask: Float[Array, "max_spikes num_neurons"],
) -> Float[Array, " 2_max_spikes"]:
    """Lifts a spike train to a discretisation of the Marcus lift
    (with time augmentation).

    **Arguments**:

    - `t0`: The start time of the path.
    - `t1`: The end time of the path.
    - `spike_times`: The times of the spikes.
    - `spike_mask`: A mask indicating the corresponding spiking neuron.

    **Returns**:

    - An array of shape `(2 * max_spikes, num_neurons + 1)` representing the Marcus lift.
    """
    num_neurons = spike_mask.shape[1]
    finite_spikes = jnp.where(jnp.isfinite(spike_times), spike_times, t1).reshape(
        (-1, 1)
    )
    spike_cumsum = jnp.cumsum(spike_mask, axis=0)
    # last_spike_time = jnp.max(jnp.where(spike_times < t1, spike_times, -jnp.inf))
    spike_cumsum_shift = jnp.roll(spike_cumsum, 1, axis=0)
    spike_cumsum_shift = spike_cumsum_shift.at[0, :].set(
        jnp.zeros(num_neurons, dtype=spike_cumsum_shift.dtype)
    )
    arr1 = jnp.hstack([finite_spikes, spike_cumsum])
    arr2 = jnp.hstack([finite_spikes, spike_cumsum_shift])
    out = jax.vmap(interleave, in_axes=1)(arr1, arr2).T
    # Makes sure the path starts at t0
    out = jnp.roll(out, 1, axis=0)
    out = out.at[0, :].set(jnp.insert(jnp.zeros(num_neurons), 0, t0))
    # time_capped = jnp.where(out[:, 0] < t1, out[:, 0], last_spike_time)
    # out = out.at[:, 0].set(time_capped)
    return out


@eqx.filter_jit
def cap_fill_ravel(ts, ys, spike_cap=10):
    # Cap the number of spikes
    ys_capped = ys[:spike_cap]
    ts_capped = ts[:spike_cap]
    spikes, neurons, times, _ = ys_capped.shape

    # Fill up infs
    idx = ts_capped > ts_capped[:, -1, None]
    idx_y = jnp.tile(idx[:, None, :, None], (1, neurons, 1, 3))
    ts_capped = jnp.where(idx, ts_capped[:, -1, None], ts_capped)
    ys_capped = jnp.where(idx_y, ys_capped[:, :, -1, None, :], ys_capped)

    xs = (ts_capped, ys_capped)
    carry_ys = jnp.zeros((neurons, times, 3))
    carr_ts = jnp.array(0.0)
    carry = (carr_ts, carry_ys)

    def _fill(carry, x):
        _ts, _ys = x
        carry_ts, carry_ys = carry
        _ys_fill_val = jnp.tile(_ys[:, None, -1], (1, times, 1))
        _ts_fill_val = _ts[-1]
        _ys_out = jnp.where(jnp.isinf(_ys), _ys_fill_val, _ys)
        _ts_out = jnp.where(jnp.isinf(_ts), _ts_fill_val, _ts)
        assert isinstance(_ys_out, Array)
        _ys_all_inf = jnp.all(jnp.isinf(_ys_out))
        _ts_all_inf = jnp.all(jnp.isinf(_ts_out))
        _ys_out = jnp.where(_ys_all_inf, carry_ys, _ys_out)
        _ts_out = jnp.where(_ts_all_inf, carry_ts, _ts_out)
        assert isinstance(_ys_out, Array)
        new_carry_ys = jnp.tile(_ys_out[:, None, -1], (1, times, 1))
        new_carry_ts = _ts_out[-1]
        new_carry = (new_carry_ts, new_carry_ys)
        out = (_ts_out, _ys_out)
        return new_carry, out

    _, xs_filled_capped = jax.lax.scan(_fill, carry, xs=xs)
    ts_filled_capped, ys_filled_capped = xs_filled_capped

    # Ravel out the "spikes" dimension
    # (spikes, neurons, times, 3) -> (neurons, spikes, times, 3)
    ys_filled_capped = jnp.transpose(ys_filled_capped, (1, 0, 2, 3))
    # (spikes, neurons, times, 3) -> (neurons, spikes*times, 3)
    ys_filled_capped_ravelled = ys_filled_capped.reshape((neurons, -1, 3))
    ts_filled_capped_ravelled = jnp.ravel(ts_filled_capped)
    return ts_filled_capped_ravelled, ys_filled_capped_ravelled


class SpikeTrain(AbstractPath):
    t0: Real
    t1: Real
    num_spikes: int
    spike_times: Array
    spike_cumsum: Array
    num_neurons: int

    def __init__(self, t0, t1, spike_times, spike_mask):
        max_spikes, num_neurons = spike_mask.shape
        self.num_neurons = num_neurons
        self.t0 = t0
        self.t1 = t1
        self.num_spikes = spike_times.shape[0]
        self.spike_times = jnp.insert(spike_times, 0, t0)
        self.spike_cumsum = jnp.cumsum(
            jnp.insert(spike_mask, 0, jnp.full_like(spike_mask[0], False), axis=0),
            axis=0,
        )

    def evaluate(
        self, t0: Real, t1: Optional[Real] = None, left: Optional[bool] = True
    ) -> Array:
        del left
        if t1 is not None:
            return self.evaluate(t1 - t0)
        idx = jnp.searchsorted(self.spike_times, t0)
        idx = jnp.where(idx > 0, idx - 1, idx)
        out = jax.lax.dynamic_slice(self.spike_cumsum, (idx, 0), (self.num_neurons, 1))[
            :, 0
        ]
        return out


class SingleSpikeTrain(AbstractPath):
    t0: Real
    t1: Real
    spike_times: Array

    def evaluate(
        self, t0: Real, t1: Optional[Real] = None, left: Optional[bool] = True
    ) -> Array:
        del left
        if t1 is not None:
            return self.evaluate(t1 - t0)
        return jnp.where(self.spike_times >= t0, 1.0, 0.0)


# A version of VirtualBrownianTree that will not throw an error when differentiated
class BrownianPath(VirtualBrownianTree):
    @eqxi.doc_remove_args("_spline")
    def __init__(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        tol: RealScalarLike,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea]
        ] = BrownianIncrement,
        _spline: _Spline = "sqrt",
    ):
        super().__init__(t0, t1, tol, shape, key, levy_area, _spline)

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> Union[PyTree[Array], BrownianIncrement, SpaceTimeLevyArea]:
        t0 = jax.lax.stop_gradient(t0)
        # map the interval [self.t0, self.t1] onto [0,1]
        t0 = linear_rescale(self.t0, t0, self.t1)
        levy_0 = self._evaluate(t0)
        if t1 is None:
            levy_out = levy_0
            levy_out = jtu.tree_map(_make_levy_val, self.shape, levy_out)
        else:
            t1 = jax.lax.stop_gradient(t1)
            # map the interval [self.t0, self.t1] onto [0,1]
            t1 = linear_rescale(self.t0, t1, self.t1)
            levy_1 = self._evaluate(t1)
            levy_out = jtu.tree_map(_levy_diff, self.shape, levy_0, levy_1)

        levy_out = levy_tree_transpose(self.shape, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, (BrownianIncrement, SpaceTimeLevyArea))
        return levy_out if use_levy else levy_out.W


class NetworkState(eqx.Module):
    """State of the neural network simulation at a given time."""

    ts: Real[Array, "samples spikes times"]
    ys: Float[Array, "samples spikes neurons times 3"]
    tevents: eqxi.MaybeBuffer[Real[Array, "samples spikes"]]
    t0: Real[Array, " samples"]
    y0: Float[Array, "samples neurons 3"]
    num_spikes: int
    event_mask: Bool[Array, "samples neurons"]
    event_types: eqxi.MaybeBuffer[Bool[Array, "samples spikes neurons"]]
    key: Any


def buffers(state: NetworkState):
    """Get the buffers from a NetworkState.

    This function is deprecated and kept for backward compatibility.
    """
    assert type(state) is NetworkState
    return state.tevents, state.ts, state.ys, state.event_types


class SpikingNeuralNet(eqx.Module):
    """A class representing a generic stochastic spiking neural network."""

    num_neurons: Int
    network: Bool[ArrayLike, "neurons neurons"] = eqx.field(static=True)
    read_out_neurons: Sequence[Int]
    v_reset: Float
    v_rest: Float
    alpha: Float
    mu: Float[ArrayLike, " 2"]
    drift_vf: Callable[..., Float[ArrayLike, "neurons 3"]]
    cond_fn: List[Callable[..., Float]]
    intensity_fn: Callable[..., Float]
    sigma: Optional[Float[ArrayLike, "2 2"]]
    diffusion_vf: Optional[Callable[..., Float[ArrayLike, "neurons 3 2 neurons"]]]

    def __init__(
        self,
        num_neurons: Int,
        intensity_fn: Callable[..., Float],
        v_reset: Float = 1.0,
        v_rest: Float = 0.0,
        alpha: Float = 3e-2,
        network: Optional[Bool[ArrayLike, "neurons neurons"]] = None,
        read_out_neurons: Optional[Sequence[Int]] = None,
        mu: Optional[Float[ArrayLike, " 2"]] = None,
        diffusion: bool = False,
        sigma: Optional[Float[ArrayLike, "2 2"]] = None,
        key: Optional[Any] = None,
    ):
        """**Arguments**:

        - `num_neurons`: The number of neurons in the network.
        - `intensity_fn`: The intensity function for spike generation.
            Should take as input a scalar (voltage) and return a scalar (intensity).
        - `v_reset`: The reset voltage value for neurons. Defaults to 1.0.
        - `v_rest`: The resting membrane potential. Defaults to 0.0.
        - `alpha`: Constant controlling the refractory period. Defaults to 3e-2.
        - `network`: The connectivity matrix of the network. Should be a square matrix of size
            `num_neurons` with the $ij$'th element being `False` if there is no connection from
            neuron $i$ to neuron $j$. If none is provided, the network is fully connected.
        - `read_out_neurons`: Sequence of indices for neurons that should be treated as read-out neurons.
        - `mu`: A 2-dimensional vector describing the drift term of each neuron.
            If none is provided, the values are randomly initialized.
        - `diffusion`: Whether to include diffusion term in the SDE. Defaults to False.
        - `sigma`: A 2 by 2 diffusion matrix. If none is provided, the values are randomly
            initialized.
        - `key`: The random key for initialization. If None,
            the key is set to `jax.random.PRNGKey(0)`.
        """

        self.num_neurons = num_neurons
        self.intensity_fn = intensity_fn
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.alpha = alpha

        if key is None:
            key = jax.random.PRNGKey(0)

        mu_key, sigma_key = jr.split(key, 2)

        if network is None:
            network = np.full((num_neurons, num_neurons), False)

        self.network = network

        if read_out_neurons is None:
            read_out_neurons = []

        self.read_out_neurons = read_out_neurons

        if mu is None:
            mu = jr.uniform(mu_key, (2,), minval=0.5)

        self.mu = mu

        def drift_vf(t, y, input_current):
            ic = input_current(t)

            @jax.vmap
            def _vf(y, ic):
                mu1, mu2 = self.mu  # type: ignore
                v, i, _ = y
                v_out = mu1 * (i + ic - v)
                i_out = -mu2 * i
                s_out = self.intensity_fn(v)
                out = jnp.array([v_out, i_out, s_out])
                out = eqx.error_if(out, jnp.any(jnp.isnan(out)), "out is nan")
                out = eqx.error_if(out, jnp.any(jnp.isinf(out)), "out is inf")
                return out

            return _vf(y, ic)

        self.drift_vf = drift_vf

        if diffusion:
            if sigma is None:
                sigma = jr.normal(sigma_key, (2, 2))
                sigma = jnp.dot(sigma, sigma.T)
                self.sigma = sigma

            sigma_large = jnp.zeros((num_neurons, 3, 2, num_neurons))
            for k in range(num_neurons):
                sigma_large = sigma_large.at[k, :2, :, k].set(sigma)

            def diffusion_vf(t, y, args):
                return sigma_large

            self.diffusion_vf = diffusion_vf
            self.sigma = sigma
        else:
            self.sigma = None
            self.diffusion_vf = None

        def cond_fn(t, y, args, n, **kwargs):
            return y[n, 2]

        self.cond_fn = [
            ft.partial(cond_fn, n=n)
            for n in range(self.num_neurons)
            if n not in read_out_neurons
        ]

    @eqx.filter_jit
    def __call__(
        self,
        input_current: Callable[..., Float[Array, " neurons"]],
        t0: Real,
        t1: Real,
        num_samples: Int,
        *,
        key,
        w: Float[Array, "neurons neurons"],
        input_spikes: Optional[Float[Array, "samples input_neurons"]] = None,
        input_weights: Optional[Float[Array, "neurons input_neruons"]] = None,
        v0: Optional[Float[Array, "samples neurons"]] = None,
        i0: Optional[Float[Array, "samples neurons"]] = None,
        dt: Real = 0.01,
        dt0: Real = 0.01,
        max_steps: Int = 1000,
    ):
        """**Arguments:**

            `input_current`: The input current to the SNN model. Should be a function
                taking as input a scalar time value and returning an array of shape
                `(self.num_neurons,)` or `(num_samples, self.num_neurons)`.
            `t0`: The starting time of the simulation.
            `t1`: The ending time of the simulation.
            `num_samples`: The number of samples to simulate.
            `key`: The random key used for generating random numbers.
            `w`: The weight matrix for the network. Should be a square matrix of size `num_neurons`.
            `input_spikes`: Optional input spike times.
            `input_weights`: Optional input spike weights.
            `v0`: The initial membrane potential of the neurons. If None,
                it will be randomly generated. Otherwise, it should be a vector of shape
                `(num_samples, self.num_neurons)`.
            `i0`: The initial membrane potential of the neurons. If None,
                it will be randomly generated. Otherwise, it should be a vector of shape
                `(num_samples, self.num_neurons)`.
            `dt`: The time resolution for capturing voltage traces and other data.
            `dt0`: The time step size used in the differential equation solve.
            `max_steps`: The maximum number of steps allowed in the differential equation solve.

        **Returns:**

            `Solution`: An object containing the simulation results,
                including the time points, membrane potentials,
                 spike times, spike marks, and the number of spikes.
        """
        # Check that input current is of correct shape
        ic_shape = jax.eval_shape(input_current, 0)
        assert ic_shape.shape == (self.num_neurons,)

        # Check that v0 and i0 are of the correct shape
        if v0 is not None:
            assert v0.shape == (num_samples, self.num_neurons)
        if i0 is not None:
            assert i0.shape == (num_samples, self.num_neurons)

        t0, t1 = float(t0), float(t1)
        _t0 = jnp.broadcast_to(t0, (num_samples,))
        key, bm_key, init_key = jr.split(key, 3)
        s0_key, i0_key, v0_key = jr.split(init_key, 3)
        # to ensure that s0 != -inf, we set minval=1e-10
        s0 = (
            jnp.log(jr.uniform(s0_key, (num_samples, self.num_neurons), minval=1e-10))
            - self.alpha
        )
        s0 = eqx.error_if(s0, jnp.any(jnp.isinf(s0)), "s0 is inf")
        if v0 is None:
            v0 = jr.uniform(
                v0_key, (num_samples, self.num_neurons), minval=0.0, maxval=self.v_reset
            )
        if i0 is None:
            i0 = jr.uniform(
                i0_key, (num_samples, self.num_neurons), minval=0.0, maxval=self.v_reset
            )
        y0 = jnp.dstack([v0, i0, s0])

        # Calculate number of time points per segment based on dt
        num_save = max(2, int((t1 - t0) / dt))
        max_spikes = max(20, int((t1 - t0) / dt))

        # Preallocate arrays but with adaptive sizes based on dt
        ys = jnp.full((num_samples, max_spikes, self.num_neurons, num_save, 3), jnp.inf)
        ts = jnp.full((num_samples, max_spikes, num_save), jnp.inf)
        tevents = jnp.full((num_samples, max_spikes), jnp.inf)
        num_spikes = jnp.zeros((), dtype=jnp.int32)
        event_mask = jnp.full((num_samples, self.num_neurons), False)
        event_types = jnp.full((num_samples, max_spikes, self.num_neurons), False)

        # Initialize state
        init_state = NetworkState(
            ts, ys, tevents, _t0, y0, num_spikes, event_mask, event_types, key
        )

        stepsize_controller = diffrax.ConstantStepSize()
        vf = diffrax.ODETerm(self.drift_vf)
        root_finder = optx.Newton(1e-2, 1e-2, optx.rms_norm)
        event = diffrax.Event(self.cond_fn, root_finder)
        solver = diffrax.Euler()
        w_update = w.at[self.network].set(0.0)
        # bm_key is not updated in body_fun since we want to make sure that the same Brownian path
        # is used for before and after each spike.
        bm_key = jr.split(bm_key, num_samples)

        if input_weights is not None:
            assert input_spikes is not None
            input_dim = input_weights.shape[1]
            input_w_large = jnp.zeros((self.num_neurons, 3, input_dim))
            input_w_large = input_w_large.at[:, 1, :].set(input_weights)

            def input_vf(t, y, args):
                return input_w_large

        @jax.vmap
        def trans_fn(y, w, key):
            v, i, s = y
            v_out = v - jnp.where(s > -1e-3, self.v_reset, 0.0)
            i_out = i + jnp.sum(w)
            s_out = jnp.where(
                s > -1e-3, jnp.log(jr.uniform(key, minval=1e-10)) - self.alpha, s
            )
            # ensures that s_out does not exceed 0 in cases where two events are triggered
            # s_out = jnp.minimum(s_out, -1e-3)

            i_out = jnp.maximum(i_out, self.v_rest)

            return jnp.array([v_out, i_out, s_out])

        def body_fun(state: NetworkState) -> NetworkState:
            new_key, trans_key = jr.split(state.key, 2)
            trans_key = jr.split(trans_key, num_samples)

            @jax.vmap
            def update(_t0, y0, trans_key, bm_key, input_spike):
                # Calculate time points for this segment based on dt
                ts = jnp.where(
                    _t0 < t1 - dt,
                    jnp.linspace(_t0, jnp.minimum(_t0 + dt * num_save, t1), num_save),
                    jnp.full((num_save,), _t0),
                )
                ts = eqxi.error_if(ts, ts[1:] < ts[:-1], "ts must be increasing")
                trans_key = jr.split(trans_key, self.num_neurons)
                saveat_ts = diffrax.SubSaveAt(ts=ts)
                saveat_t1 = diffrax.SubSaveAt(t1=True)
                saveat = diffrax.SaveAt(subs=(saveat_ts, saveat_t1))
                terms = vf
                multi_terms = []
                if self.diffusion_vf is not None:
                    bm = BrownianPath(
                        t0 - 1,
                        t1 + 1,
                        tol=dt0 / 2,
                        shape=(2, self.num_neurons),
                        key=bm_key,
                    )
                    cvf = diffrax.ControlTerm(self.diffusion_vf, bm)
                    multi_terms.append(cvf)
                if input_spike is not None:
                    assert input_weights is not None
                    input_st = SingleSpikeTrain(t0, t1, input_spike)
                    input_cvf = diffrax.ControlTerm(input_vf, input_st)
                    multi_terms.append(input_cvf)
                if multi_terms:
                    terms = diffrax.MultiTerm(terms, *multi_terms)

                sol = diffrax.diffeqsolve(
                    terms,
                    solver,
                    _t0,
                    t1,
                    dt0,
                    y0,
                    input_current,
                    throw=False,
                    stepsize_controller=stepsize_controller,
                    saveat=saveat,
                    event=event,
                    max_steps=max_steps,
                )

                assert sol.event_mask is not None
                event_mask = jnp.array(sol.event_mask)
                event_happened = jnp.any(event_mask)

                assert sol.ts is not None
                ts = sol.ts[0]
                _t1 = sol.ts[1]
                tevent = _t1[0]
                # If tevent > t1 we normalize to keep within range
                tevent = jnp.where(tevent > t1, tevent * (t1 / tevent), tevent)
                tevent = eqxi.error_if(tevent, jnp.isnan(tevent), "tevent is nan")

                assert sol.ys is not None
                ys = sol.ys[0]
                _y1 = sol.ys[1]
                yevent = _y1[0].reshape((self.num_neurons, 3))
                yevent = jnp.where(_t0 < t1, yevent, y0)
                yevent = eqxi.error_if(
                    yevent, jnp.any(jnp.isnan(yevent)), "yevent is nan"
                )
                yevent = eqxi.error_if(
                    yevent, jnp.any(jnp.isinf(yevent)), "yevent is inf"
                )

                event_array = jnp.array(yevent[:, 2] > -1e-3)
                w_update_t = jnp.where(
                    jnp.tile(event_array, (self.num_neurons, 1)).T, w_update, 0.0
                ).T
                w_update_t = jnp.where(event_happened, w_update_t, 0.0)
                ytrans = trans_fn(yevent, w_update_t, trans_key)
                ytrans = eqx.error_if(
                    ytrans, ~jnp.all(ytrans[:, 2] < 0), "s is not negative"
                )
                ys = jnp.transpose(ys, (1, 0, 2))

                return ts, ys, tevent, ytrans, event_mask

            _ts, _ys, tevent, _ytrans, event_mask = update(
                state.t0, state.y0, trans_key, bm_key, input_spikes
            )
            num_spikes = state.num_spikes + 1

            ts = state.ts
            ts = ts.at[:, state.num_spikes].set(_ts)

            ys = state.ys
            ys = ys.at[:, state.num_spikes].set(_ys)

            event_types = state.event_types
            event_types = event_types.at[:, state.num_spikes].set(event_mask)

            tevents = state.tevents
            tevents = tevents.at[:, state.num_spikes].set(tevent)

            new_state = NetworkState(
                ts=ts,
                ys=ys,
                tevents=tevents,
                t0=tevent,
                y0=_ytrans,
                num_spikes=num_spikes,
                event_mask=event_mask,
                event_types=event_types,
                key=new_key,
            )

            return new_state

        def stop_fn(state: NetworkState) -> bool:
            return (state.num_spikes < max_spikes) & (jnp.min(state.t0) < t1)

        final_state = eqxi.while_loop(
            stop_fn,
            body_fun,
            init_state,
            buffers=buffers,
            max_steps=max_spikes,
            kind="checkpointed",
        )

        ys = final_state.ys
        ts = final_state.ts
        spike_times = final_state.tevents
        spike_marks = final_state.event_types
        num_spikes = final_state.num_spikes

        sol = Solution(
            t1=t1,
            ys=ys,
            ts=ts,
            spike_times=spike_times,
            spike_marks=spike_marks,
            num_spikes=num_spikes,
            dt=dt,
        )
        return sol

    def run(self, *args, **kwargs):
        sol = self.__call__(*args, **kwargs)
        return sol, plottable_paths(sol)


def _build_forward_network(in_size, out_size, width_size, depth):
    if depth <= 1:
        width_size = out_size
    num_neurons = in_size + width_size * (depth - 1) + out_size
    network_out = np.full((num_neurons, num_neurons), True)
    layer_idx = [0] + [in_size] + [width_size] * (depth - 1) + [out_size]
    layer_idx = np.cumsum(np.array(layer_idx))
    for i in range(depth):
        lrows = layer_idx[i]
        urows = layer_idx[i + 1]
        lcols = layer_idx[i + 1]
        ucols = layer_idx[i + 2]
        network_fill = np.full((urows - lrows, ucols - lcols), False)
        network_out[lrows:urows, lcols:ucols] = network_fill
    return network_out


class FeedForwardSNN(SpikingNeuralNet):
    """A convenience wrapper around `SpikingNeuralNet` for a feedforward network."""

    in_size: Int
    out_size: Int
    width_size: Int
    depth: Int

    def __init__(
        self,
        in_size,
        out_size,
        width_size,
        depth,
        intensity_fn,
        key,
        read_out_layer=None,
        **kwargs,
    ):
        """**Arguments**:

        - `in_size`: The number of input neurons.
        - `out_size`: The number of output neurons.
        - `width_size`: The number of neurons in each hidden layer.
        - `depth`: The number of hidden layers.
        - `intensity_fn`: The intensity function for spike generation.
            Should take as input a scalar (voltage) and return a scalar (intensity).
        - `key`: The random key for initialization.
        - `**kwargs`: Additional keyword arguments passed to `SpikingNeuralNet`.
        """
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        num_neurons = self.in_size + self.width_size * (self.depth - 1) + self.out_size
        network = _build_forward_network(
            self.in_size, self.out_size, self.width_size, self.depth
        )

        if read_out_layer:
            read_out_neurons = list(range(num_neurons - out_size, num_neurons))
        else:
            read_out_neurons = None

        super().__init__(
            num_neurons=num_neurons,
            intensity_fn=intensity_fn,
            network=network,
            key=key,
            read_out_neurons=read_out_neurons,
            **kwargs,
        )

    def __call__(
        self,
        input_current: Callable[..., Float[Array, " input_size"]],
        t0: Real,
        t1: Real,
        num_samples: int,
        *,
        w: Float[Array, "neurons neurons"],
        input_spikes: Optional[Float[Array, "samples input_neurons"]] = None,
        input_weights: Optional[Float[Array, "input_size input_neruons"]] = None,
        v0: Real[Array, " neurons"],
        i0: Real[Array, " neurons"],
        key: Any,
        dt: Real = 0.01,
        dt0: Real = 0.01,
        max_steps: Int = 1000,
    ):
        """**Arguments**:

        - `input_current`: The input current to the SNN model. Should be a function
            taking as input a scalar time value and returning an arrray of shape
            `(self.in_size,)` or `(num_samples, self.in_size)`.
        - `t0`: The starting time of the simulation.
        - `t1`: The ending time of the simulation.
        - `num_samples`: The number of samples to simulate.
        - `w`: The weight matrix for the network. Should be a square matrix of size `num_neurons`.
        - `key`: The random key used for generating random numbers.
        - `v0`: The initial membrane potential of the neurons. Should be a vector of shape
            `(num_samples, self.num_neurons)`.
        - `i0`: The initial membrane potential of the neurons. Should be a vector of shape
            `(num_samples, self.num_neurons)`.
        - `dt`: The time resolution for capturing voltage traces and other data.
        - `dt0`: The time step size used in the differential equation solve.
        - `max_steps`: The maximum number of steps allowed in the differential equation solve.

        **Returns:**

            `Solution`: An object containing the simulation results,
                including the time points, membrane potentials,
                 spike times, spike marks, and the number of spikes.
        """

        def _input_current(t: Float) -> Array:
            return jnp.hstack(
                [input_current(t), jnp.zeros((self.num_neurons - self.in_size,))]
            )

        if input_weights is not None:
            assert input_spikes is not None
            input_dim = input_weights.shape[1]
            _input_weights = jnp.vstack(
                [input_weights, jnp.zeros((self.num_neurons - self.in_size, input_dim))]
            )
        else:
            _input_weights = None

        return super().__call__(
            _input_current,
            t0,
            t1,
            num_samples,
            key=key,
            w=w,
            input_spikes=input_spikes,
            input_weights=_input_weights,
            v0=v0,
            i0=i0,
            dt=dt,
            dt0=dt0,
            max_steps=max_steps,
        )
