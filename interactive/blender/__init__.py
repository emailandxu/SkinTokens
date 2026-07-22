"""H3D Skintokens Blender extension."""


def register() -> None:
    from . import addon

    addon.register()


def unregister() -> None:
    from . import addon

    addon.unregister()


__all__ = ["register", "unregister"]
