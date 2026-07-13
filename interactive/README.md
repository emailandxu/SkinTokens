# SkinTokens Interactive MVP

This directory is additive. It imports the existing SkinTokens code but does
not require edits outside `interactive/`.

## Protocol

All local services use `multiprocessing.connection` over Unix sockets. There is
no HTTP `POST`; requests are Python dicts with a `command` field.

Implemented model commands:

- `start`: load an OBJ into a model session and cache mesh conditioning.
- `next`: continue autoregressive skeleton generation freely until one complete
  joint or skeleton `EOS` is produced.
- `skin`: treat the provided/current skeleton context as final and generate
  skin tokens.
- `reset`: delete a model session.
- `branch`: reserved for future forced-branch interaction; not implemented in
  the MVP.

The client owns editable skeleton context. Send edited `joints`, `parents`, and
`joint_names` with `next` or `skin`; the server re-tokenizes that context before
continuing generation.

## Start The Model Server

```bash
uv run python -m interactive.server --device cuda
```

CPU mode is available for debugging:

```bash
uv run python -m interactive.server --device cpu
```

## Direct Model Smoke

In another terminal:

```bash
uv run python -m interactive.client model \
  --obj examples/xiaobaozi.obj \
  --steps 2
```

To generate skin from the current partial skeleton:

```bash
uv run python -m interactive.client model \
  --obj examples/xiaobaozi.obj \
  --steps 4 \
  --skin \
  --output results/xiaobaozi_interactive_skin.txt
```

## Blender Validation Server

Run this inside Blender's Python environment, after the model server is up:

```bash
blender --background --python-expr \
  "import sys; sys.path.insert(0, '/path/to/SkinTokens'); import interactive.blender.server as s; s.main()"
```

Then drive it from a normal terminal:

```bash
uv run python -m interactive.client blender \
  --obj examples/xiaobaozi.obj \
  --steps 2
```

The validation server imports the OBJ into Blender on `start`, draws preview
joints/bones on `next`, and applies generated `skin` rows as mesh vertex groups
after `skin`. Creating a final armature object is left for the addon UI phase.

## Blender Addon UI

Start the model server first:

```bash
uv run python -m interactive.server --device cuda
```

Then launch Blender with the addon registered and `xiaobaozi.obj` loaded:

```bash
/snap/bin/blender --python interactive/blender/launch.py
```

To load another OBJ or prefill a heter-skinning txt:

```bash
/snap/bin/blender --python interactive/blender/launch.py -- \
  --obj examples/xiaobaozi.obj \
  --txt examples/samples/xiaobaozi_skin.txt
```

FBX files with an armature can also be loaded:

```bash
/snap/bin/blender --python interactive/blender/launch.py -- \
  --fbx examples/m2/cloth/SK_97000000501.FBX
```

Open the 3D View sidebar with `N`, then use the `SkinTokens` tab:

- `Start`: create a model session for the selected mesh.
- `Import TXT`: load a heter-skinning txt skeleton/skin into the current session.
- `Import FBX Rig`: load the selected mesh armature/vertex groups into the current session.
- `Next`: freely generate one more joint or skeleton EOS.
- `Skin`: generate skin for the current skeleton and apply vertex groups.
