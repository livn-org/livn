def supports(env_or_class, *capabilities) -> bool:
    from livn.types import supports as _supports

    return _supports(env_or_class, *capabilities)


def backend_supports(*capabilities) -> bool:
    from livn.env import Env

    return supports(Env, *capabilities)


def missing(*capabilities) -> list[str]:
    return [c for c in capabilities if not backend_supports(c)]
