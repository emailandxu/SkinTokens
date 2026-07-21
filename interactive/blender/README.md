# SkinTokens Interactive

SkinTokens Interactive connects Blender 4.2 or newer to a separately deployed
SkinTokens model service. The extension uploads the selected mesh, lets users
generate or edit an armature, and applies generated skin weights in Blender.

After installing the extension from disk, enable Blender Online Access, open
the extension preferences, configure the model service URL, and use **Test
Connection** before starting a session. Mesh geometry is sent to that service.
When Online Access is disabled, the SkinTokens sidebar provides an **Allow
Online Access** button. If Blender was launched with `--offline-mode`, restart
it without that option first.

The model service and checkpoint are not included in this extension package.
