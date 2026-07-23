"""Run after installing the built extension into an isolated Blender profile."""

from __future__ import annotations

import importlib
import tempfile

import bpy  # type: ignore
import numpy as np


matches = [
    name
    for name in bpy.context.preferences.addons.keys()
    if name.endswith(".h3d_skintokens")
]
assert len(matches) == 1, matches
package_name = matches[0]
extension = importlib.import_module(package_name)
addon = importlib.import_module(f"{package_name}.addon")
apply_skin = importlib.import_module(f"{package_name}.apply_skin")
transport = importlib.import_module(f"{package_name}.transport")

assert addon.__package__ == package_name
assert transport.EXTENSION_PACKAGE_ID == "h3d_skintokens"
assert transport.EXTENSION_VERSION == "1.0.7"
assert addon.SKINTOKENS_PT_interactive.bl_label == "H3D Skintokens"
assert addon.SKINTOKENS_PT_interactive.bl_category == "H3D Skintokens"
assert hasattr(bpy.types.Scene, "skintokens_status")
assert str(
    bpy.context.preferences.addons[package_name].preferences.server_url
) == "http://172.17.60.251:8765"
assert addon._version_is_newer("1.8.4", "1.8.3")
assert addon._version_is_newer("2.0.0", "1.99.99")
assert addon._version_is_newer("1.8.4", "1.8.4-beta.1")
assert not addon._version_is_newer("1.8.4", "1.8.4")
assert not addon._version_is_newer("1.8.4-beta.1", "1.8.4")
repository = addon._extension_repository(bpy.context.preferences)
if repository is not None:
    _repo_index, repo = repository
    if not repo.use_remote_url:
        assert addon._extension_update_info(bpy.context.preferences) is None
assert bpy.app.handlers.undo_post.count(addon.sync_model_after_history_change) == 1
assert bpy.app.handlers.redo_post.count(addon.sync_model_after_history_change) == 1
assert (
    bpy.app.handlers.depsgraph_update_post.count(
        addon.sync_model_after_armature_change
    )
    == 1
)

source = np.arange(12, dtype=np.float32).reshape(3, 4)
encoded = transport.encode_float32_array(source)
decoded = transport.decode_float32_array(encoded, source.shape)
np.testing.assert_array_equal(decoded, source)

armature_data = bpy.data.armatures.new("SkinTokensExtensionSmokeArmature")
armature = bpy.data.objects.new(armature_data.name, armature_data)
bpy.context.scene.collection.objects.link(armature)
apply_skin.apply_armature_context(
    armature.name,
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [-1, 0],
    ["root", "child"],
    select_name="child",
    mode_after="POSE",
)
assert apply_skin.active_armature_bone_name(armature.name) == "child"

blend_path = f"{tempfile.gettempdir()}/skintokens_extension_handler_smoke.blend"
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
bpy.ops.wm.open_mainfile(filepath=blend_path)
assert bpy.app.handlers.undo_post.count(addon.sync_model_after_history_change) == 1
assert bpy.app.handlers.redo_post.count(addon.sync_model_after_history_change) == 1
assert (
    bpy.app.handlers.depsgraph_update_post.count(
        addon.sync_model_after_armature_change
    )
    == 1
)

extension.unregister()
assert not hasattr(bpy.types.Scene, "skintokens_status")
assert addon.sync_model_after_history_change not in bpy.app.handlers.undo_post
assert addon.sync_model_after_history_change not in bpy.app.handlers.redo_post
assert addon.sync_model_after_armature_change not in bpy.app.handlers.depsgraph_update_post

extension.register()
assert hasattr(bpy.types.Scene, "skintokens_status")
assert bpy.app.handlers.undo_post.count(addon.sync_model_after_history_change) == 1
assert bpy.app.handlers.redo_post.count(addon.sync_model_after_history_change) == 1
assert (
    bpy.app.handlers.depsgraph_update_post.count(
        addon.sync_model_after_armature_change
    )
    == 1
)

print("SKINTOKENS_EXTENSION_SMOKE_OK", package_name)
