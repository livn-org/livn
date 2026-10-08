from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def feasibility_transformer(patience: int = 150):
    from dmosopt.model_transformer import JointFTTransformer

    class FeasibilityTransformer(JointFTTransformer):
        validation_split = 0.2

        def autofit(self, x, y, yC, **kwargs):
            import keras

            kwargs.setdefault("validation_split", self.validation_split)
            extra = list(kwargs.pop("callbacks", ()) or ())
            extra.append(
                keras.callbacks.EarlyStopping(
                    monitor={
                        "c+o": "val_objectives_loss",
                        "c": "val_loss",
                        "o": "val_mae",
                    }[self.mode],
                    patience=patience,
                    mode="min",
                    restore_best_weights=True,
                )
            )
            return super().autofit(x, y, yC, callbacks=extra, **kwargs)

        def make_feasible(
            self,
            X,
            targets: str = "constraint",
            max_iterations: int = 1000,
            learning_rate: float = 1e-3,
            plateau_window: int = 50,
            bound_margin: float = 0.01,
            min_iterations: int = 0,
            constraint_columns=None,
        ) -> tuple[np.ndarray, dict]:
            import jax
            import jax.numpy as jnp
            import keras

            model = self
            mode = model.mode
            X = np.atleast_2d(np.asarray(X, dtype=np.float32))
            lo = np.asarray(model.xlb, dtype=np.float32).ravel()
            hi = np.asarray(model.xub, dtype=np.float32).ravel()
            y_norm = None if model.y_norm_ is None else jnp.asarray(model.y_norm_)
            focal = keras.losses.BinaryFocalCrossentropy()

            def heads(x):
                out = model(x, training=False)
                if mode == "c+o":
                    return out["objectives"], out["constraints"]
                return (out, None) if mode == "o" else (None, out)

            terms = []
            if "objective" in targets and mode != "c" and y_norm is not None:

                def hypervolume(x):
                    o, _ = heads(x)
                    nadir = jnp.max(jnp.concatenate([o, y_norm], axis=0), axis=0)
                    front = o / (nadir + 1e-8)
                    return -jnp.sum(jnp.prod(jnp.maximum(1.1 - front, 0.0), axis=-1))

                terms.append(hypervolume)
            if "constraint" in targets and mode != "o" and model.num_constraints:
                columns = (
                    None
                    if constraint_columns is None
                    else jnp.asarray(
                        [
                            int(i)
                            for i in constraint_columns
                            if int(i) < model.num_constraints
                        ]
                    )
                )

                def infeasibility(x):
                    _, c = heads(x)
                    if columns is not None:
                        c = c[:, columns]
                    return focal(jnp.ones_like(c), c)

                terms.append(infeasibility)
            if not terms:
                return X, {"steps": 0}

            steps = [jax.jit(jax.value_and_grad(term)) for term in terms]
            rate = jnp.asarray(float(learning_rate) * (hi - lo))
            inset = float(bound_margin) * (hi - lo)
            lo_in, hi_in = jnp.asarray(lo + inset), jnp.asarray(hi - inset)
            x = jnp.asarray(X)
            m = jnp.zeros_like(x)
            v = jnp.zeros_like(x)
            b1, b2, eps = 0.9, 0.999, 1e-7
            history: list[float] = []
            iteration = 0
            for iteration in range(1, int(max_iterations) + 1):
                evaluated = [step(x) for step in steps]
                losses = [float(value) for value, _ in evaluated]
                grads = [g for _, g in evaluated]
                if len(grads) > 1:
                    ref = jnp.linalg.norm(grads[0])
                    factors = [ref / (jnp.linalg.norm(g) + 1e-8) for g in grads]
                    grad = sum(f * g for f, g in zip(factors, grads, strict=True))
                    loss = float(
                        sum(
                            float(f) * term
                            for f, term in zip(factors, losses, strict=True)
                        )
                    )
                else:
                    grad, loss = grads[0], losses[0]
                if not np.isfinite(loss):
                    return X, {"steps": iteration, "failed": True}
                history.append(loss)
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad**2
                mhat = m / (1 - b1**iteration)
                vhat = v / (1 - b2**iteration)
                x = jnp.clip(x - rate * mhat / (jnp.sqrt(vhat) + eps), lo_in, hi_in)
                if iteration >= min_iterations and len(history) > plateau_window:
                    recent = history[-plateau_window:]
                    q1, q3 = np.percentile(recent, [25, 75])
                    median = np.median(recent)
                    if (q3 - q1) / (abs(median) if median != 0 else 1.0) < 0.01:
                        break
            info = {
                "steps": iteration,
                "loss_start": history[0] if history else float("nan"),
                "loss_end": history[-1] if history else float("nan"),
            }
            return np.asarray(x, dtype=np.float32), info

    return FeasibilityTransformer


class Projecting:
    MIN_STEPS = 200

    def __init__(
        self, optimizer, surrogate, seen, targets, iterations, after, liveness=()
    ):
        self._wrapped = optimizer
        self._wrapped.x_distance_metrics = None
        self._surrogate = surrogate
        self._seen = np.asarray(seen, dtype=np.float64)
        self._targets = targets
        self._iterations = int(iterations)
        self._after = int(after)
        self._liveness = tuple(int(c) for c in liveness)
        self._generations = 0
        self._cache: tuple[bytes, tuple] | None = None

    @property
    def population_objectives(self):
        parameters = np.array(self._wrapped.state.population_parm, copy=True)
        objectives = np.array(self._wrapped.state.population_obj, copy=True)
        if self._generations < self._after:
            return parameters, objectives
        key = parameters.tobytes()
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        move = np.zeros(len(parameters), dtype=bool)
        move[len(parameters) // 2 :] = True
        if len(self._seen):
            from scipy.spatial.distance import cdist

            move |= cdist(parameters, self._seen).min(axis=1) <= 1e-12
        dead = self.predicted_dead(parameters)
        rest = move & ~dead
        info = {}
        if rest.any():
            moved, info = self._surrogate.make_feasible(
                parameters[rest],
                targets=self._targets,
                max_iterations=self._iterations,
                min_iterations=self.MIN_STEPS,
            )
            parameters[rest] = moved
        dead_info = {}
        if dead.any():
            moved, dead_info = self._surrogate.make_feasible(
                parameters[dead],
                targets="constraint",
                max_iterations=self._iterations,
                min_iterations=self.MIN_STEPS,
                constraint_columns=self._liveness,
            )
            parameters[dead] = moved
        move = rest | dead
        still = self.predicted_dead(parameters)
        logger.info(
            "feasibility projection: %d of %d moved (%d predicted dead, %d elites); "
            "rest %s steps, loss %.4g -> %.4g; dead %s steps, loss %.4g -> %.4g; "
            "predicted dead after %d",
            int(move.sum()),
            len(parameters),
            int(dead.sum()),
            int(dead[: len(parameters) // 2].sum()),
            info.get("steps"),
            info.get("loss_start", float("nan")),
            info.get("loss_end", float("nan")),
            dead_info.get("steps"),
            dead_info.get("loss_start", float("nan")),
            dead_info.get("loss_end", float("nan")),
            int(still.sum()),
        )
        predicted = np.asarray(self._surrogate.evaluate(parameters))
        result = (
            parameters,
            predicted if predicted.shape == objectives.shape else objectives,
        )
        self._cache = (key, result)
        return result

    def predicted_dead(self, parameters) -> np.ndarray:
        model = self._surrogate._wrapped
        columns = [c for c in self._liveness if c < int(model.num_constraints)]
        if not columns or model.mode == "o":
            return np.zeros(len(parameters), dtype=bool)
        out = model.predict(np.asarray(parameters, dtype=np.float32), verbose=0)
        probabilities = np.asarray(out["constraints"] if isinstance(out, dict) else out)
        return np.prod(probabilities[:, columns], axis=1) < 0.5

    def get_population_strategy(self):
        return self.population_objectives

    def generate(self, *args, **kwargs):
        self._generations += 1
        return self._wrapped.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def resolve_liveness(spec, constraint_names, num_constraints: int) -> tuple[int, ...]:
    spec = tuple(spec or ())
    if not spec:
        return ()

    if all(isinstance(c, (int, np.integer)) for c in spec):
        keep = tuple(int(c) for c in spec if 0 <= int(c) < num_constraints)
        dropped = [int(c) for c in spec if int(c) not in keep]
        if dropped:
            logger.warning(
                "feasibility_liveness %s is outside the %d constraint columns "
                "and was dropped; prefer constraint names, which are checked",
                dropped,
                num_constraints,
            )
        return keep

    names = list(constraint_names or ())
    if not names:
        logger.warning(
            "feasibility_liveness names %s cannot be resolved as the problem "
            "declares no constraint columns",
            list(spec),
        )
        return ()

    index = {n: i for i, n in enumerate(names)}
    missing = [c for c in spec if c not in index]
    if missing:
        raise KeyError(
            f"feasibility_liveness names {missing} are not constraints of this "
            f"problem; it has {sorted(index)}"
        )
    return tuple(index[c] for c in spec)
