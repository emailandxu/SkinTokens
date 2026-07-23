# H3D Skintokens

H3D Skintokens connects Blender 4.2 or newer to a separately deployed
SkinTokens model service. The extension uploads the selected mesh, lets users
generate or edit an armature, and applies generated skin weights in Blender.

After installing the extension from disk, enable Blender Online Access and use
**Test Connection** before starting a session. The server URL can be edited in
the extension UI. Mesh geometry is sent to that service.
When Online Access is disabled, the H3D Skintokens sidebar provides an **Allow
Online Access** button. If Blender was launched with `--offline-mode`, restart
it without that option first.

The model service and checkpoint are not included in this extension package.

For repository-based installation, add the H3D model server as a Blender
Extension Repository, then install the package by ID:

```bash
blender --online-mode --command extension repo-add skintokens \
  --name "H3D Skintokens Extensions" \
  --url http://SERVER:8765/blender/extensions/
blender --online-mode --command extension install \
  --sync --enable h3d_skintokens
```

Repository installations trigger one non-blocking sync of their own repository
when the extension loads. When the synchronized repository index contains a newer
H3D Skintokens version, the idle **Start** button becomes **Update** and delegates
the replacement to Blender's native extension installer. Packages installed
into a local repository from disk keep the normal **Start** button.
