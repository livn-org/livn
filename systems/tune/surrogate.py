from __future__ import annotations


def constraint_columns(weights: dict, names) -> dict[int, float]:
    index = {n: i for i, n in enumerate(names)}
    if missing := sorted(set(weights) - set(index)):
        raise KeyError(
            f"constraint weights name {missing}, which are not constraints of "
            f"this problem; it has {list(names)}"
        )
    return {index[n]: float(w) for n, w in weights.items()}


def early_stopping_transformer(patience: int = 150):
    from dmosopt.model_transformer import JointFTTransformer

    class EarlyStoppingTransformer(JointFTTransformer):
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

    return EarlyStoppingTransformer
