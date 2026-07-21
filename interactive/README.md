# SkinTokens Interactive MVP

This directory is additive. It imports the existing SkinTokens code but does
not require edits outside `interactive/`.

## Protocol

The Blender addon talks to the model service over HTTP. `Start` uploads the OBJ
once, `Next` and `Sync` exchange skeleton JSON, and `Skin` downloads one
compressed full-resolution weight matrix that is applied directly to Blender
vertex groups. The original Unix-socket entry point remains available for
local debugging.

The client owns the editable skeleton context. Runtime model conditioning is an
LRU cache on the server and can be rebuilt from the uploaded OBJ when evicted.
Lightweight session records use a separate 512-entry LRU by default. Session
request locks exist only while requests are active or waiting, so idle and
invalid session IDs do not leave lock entries behind.
Uploaded meshes are content-addressed under `.runtime/interactive_assets/` and
are intentionally not deleted by this version.

Implemented model commands:

- `start`: load an OBJ into a model session and cache mesh conditioning.
- `next`: continue autoregressive skeleton generation freely until one complete
  joint or skeleton `EOS` is produced.
- `skin`: treat the provided/current skeleton context as final and generate
  skin tokens.
- `reset`: delete a model session.
- `branch`: continue from a selected branch parent.

The client owns editable skeleton context. Send edited `joints`, `parents`, and
`joint_names` with `next` or `skin`; the server re-tokenizes that context before
continuing generation.

Each server session also keeps a lightweight reserved-name registry outside the
GPU runtime cache. Current client context names are added on every request and
are not reused after Delete. If a decoded Next/Rig name is already reserved,
the server assigns the next unused `bone_N`. Runtime LRU eviction preserves
this registry; Reset/Finish removes it with the session.

## Start The HTTP Model Server

```bash
uv run python -m interactive.http_server \
  --host 0.0.0.0 \
  --port 8765 \
  --device cuda \
  --max-runtime-sessions 8 \
  --max-sessions 512
```

On first startup the server creates `.runtime/server.json`. This is the shared
configuration file for both HTTP and local socket server entry points. It keeps
the listener, model/device, storage directories, usage directory, both LRU
limits, session timeout, cleanup interval, context limit, upload limit, GPU
queue capacity, and default skin postprocess in one place. Relative runtime
paths in this file are resolved from the directory containing the config file.

The default session settings are:

```json
{
  "max_runtime_sessions": 8,
  "max_sessions": 512,
  "session_idle_timeout_seconds": 3600,
  "session_cleanup_interval_seconds": 60
}
```

A session expires after one hour without an authorized session request. The
cleanup thread checks every minute, removes its SessionRecord and GPU runtime,
and logs
`session_end` with `end_reason=idle_timeout`. Saved mesh files remain untouched.
The server reads the configuration at startup; restart it after editing the
file. Existing command-line options take precedence over values in the file.
Use `--config PATH` to select a different JSON file.

CPU mode is available for debugging:

```bash
uv run python -m interactive.http_server --host 127.0.0.1 --device cpu
```

Use `127.0.0.1` when Blender and the model are on the same machine. For another
machine on a trusted LAN, set the Blender `Server URL` to
`http://MODEL_MACHINE_IP:8765`. This MVP does not provide TLS or real user
authentication, so do not expose it directly to the public internet.

## Direct Model Smoke

In another terminal:

```bash
uv run python -m interactive.client model \
  --endpoint http://127.0.0.1:8765 \
  --obj examples/xiaobaozi.obj \
  --steps 2
```

To generate skin from the current partial skeleton:

```bash
uv run python -m interactive.client model \
  --endpoint http://127.0.0.1:8765 \
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

The validation server imports the OBJ and creates an empty working Armature on
`start`, adds generated bones directly to that Armature on `next`/`rig`, and
applies generated `skin` rows as mesh vertex groups after `skin`.

## Blender Addon UI

### Build The Blender 4.2+ Extension

The distributable extension is rooted at `interactive/blender/`. It contains a
self-contained HTTP transport and does not import the repository-level
`interactive.protocol`.

Validate the source manifest from that directory:

```bash
cd interactive/blender
blender --command extension validate
```

Build and validate the release archive from the repository root:

```bash
mkdir -p dist
blender --command extension build \
  --source-dir interactive/blender \
  --output-dir dist
blender --command extension validate \
  dist/skintokens_interactive-1.8.0.zip
```

Install the ZIP with `Preferences > Extensions > Install from Disk`. Enable
Online Access, then configure and test the model service URL in the extension
preferences. The release archive intentionally excludes the model, checkpoint,
development launcher, local socket server, tests, and caches.

Start the HTTP model server first:

```bash
uv run python -m interactive.http_server --host 127.0.0.1 --device cuda
```

Then launch Blender with the addon registered and `xiaobaozi.obj` loaded:

```bash
/snap/bin/blender --python interactive/blender/launch.py
```

For a model server on another machine:

```bash
/snap/bin/blender --python interactive/blender/launch.py -- \
  --server-url http://MODEL_MACHINE_IP:8765
```

To load another OBJ or prefill a heter-skinning txt:

```bash
/snap/bin/blender --python interactive/blender/launch.py -- \
  --obj examples/xiaobaozi.obj \
  --txt examples/samples/xiaobaozi_skin.txt
```

The output path and generation settings are launch-only options and are not
shown in the addon panel. For example:

```bash
/snap/bin/blender --python interactive/blender/launch.py -- \
  --obj examples/xiaobaozi.obj \
  --output results/xiaobaozi_interactive_skin.txt \
  --next-tokens 16 --skin-tokens 2048 \
  --top-k 5 --top-p 0.95 --temperature 1.5 \
  --repetition-penalty 1.2 --num-beams 1
```

FBX files with an armature can also be loaded:

```bash
/snap/bin/blender --python interactive/blender/launch.py -- \
  --fbx examples/m2/cloth/SK_97000000501.FBX
```

When `Start` is pressed for a mesh with an Armature modifier or an Armature
parent, the addon automatically reads the scene skeleton, syncs it as the model
context, and uses that Armature as the only editable skeleton. Every original
bone is preserved; vertex groups are not used to trim the skeleton. Meshes
without an armature receive an empty SkinTokens Armature at Start.
For an unbound Armature with no skin or modifier, select the target mesh and the
Armature together, then press `Start`; either object may be active. The mesh is
uploaded while the selected Armature supplies the complete skeleton context.

The model service rejects skeletons above `--max-context-bones` instead of
silently deleting bones. The default limit is 96. For example:

```bash
uv run python -m interactive.http_server \
  --host 0.0.0.0 \
  --device cuda \
  --max-context-bones 96 \
  --skin-postprocess vae-reconstruction
```

Open the 3D View sidebar with `N`, then use the `SkinTokens` tab:

- `Server URL`: HTTP address of the model server.
- `Start` / `Finish`: one state-dependent button creates or releases the model
  session. Finish keeps the Mesh, Armature, skin weights, and source asset
  metadata. When `--output` was supplied at launch, it serializes the current
  Armature and current Blender vertex groups to that path, including applied
  VAE reconstruction and manual weight edits.
- `Next`: generate the next natural DFS unit. The model may continue the chain,
  pop to another branch, or emit skeleton EOS. Any new bone becomes active.
- `Force Next`: generate one child using the active Edit/Pose bone as its
  explicit parent. A parent is required once the Armature is nonempty.
- `Rig`: continue from the current skeleton until skeleton EOS, preserving all
  bones that are already present.
- `Split`: divide the active bone's displayed downstream segment.
- `Delete`: directly remove an active leaf; for an internal bone, dissolve its
  displayed downstream joint and preserve that joint's children.
- `Skin`: generate skin for the current skeleton and apply vertex groups.
- `VAE Level`: choose reconstruction level 0-3 for exactly one active skin
  vertex group or selected Pose/Edit bone. Level 0 is the original field.
- Checkmark: commit the visible VAE preview as the new level 0.

Before Start creates a live Blender session, all generation and editing
controls are disabled. `Rig` and `Skin` occupy the first operation row, followed
by `Next` / `Force Next`, `Split` / `Delete`, and the VAE row. The state-dependent
`Start/Finish` button is the final control at the bottom of the panel. The VAE
slider uses most of its row and Apply is a compact checkmark. Output, generation
parameters, and the raw Session ID are intentionally hidden. Finish cancels
pending automatic Armature synchronization and removes transient session IDs
from the Mesh. Skin and
Finish do not create an intermediate TXT; a failed server reset does not
prevent local cleanup. Import TXT remains available to scripts and launch
parameters but its path and button are intentionally hidden from the panel.

Start, Next, Force Next, Rig, Skin, Split, Delete, VAE reconstruction, and
automatic Armature synchronization perform HTTP work on a daemon worker
thread. Blender scene reads and result application remain on the main thread.
While one request is active, plugin controls are disabled but the Blender UI
stays responsive. If the Armature or relevant skin weights change before a
result returns, that stale result is discarded and the current Armature is
resynchronized. Finish writes local output on the main thread and sends the
server reset in the background.

The server-side `midprocess` defaults to `dfs-ensemble`; it reuses the mesh
encoder and VAE conditioning already held by the session. The default profile
is eight candidates in one batch with ten beams. The selector minimizes
cross-subtree leakage while allowing at most 1% relative regression in far-bone
mass and distance regret. Full-resolution interpolation and optional server
postprocess run once on the winner, then the compressed weights are returned in
memory. These server parameters are intentionally not shown in the Blender
panel.

`skin_postprocess=vae-reconstruction` projects every normalized bone weight
field through the pretrained 32-token Hard-FSQ VAE, then applies Top-K and
normalizes the complete skin again. It reuses the VAE already loaded by the
model service. The local validation client can request it with:

```bash
.venv/bin/python -m interactive.client model \
  --skin \
  --skin-postprocess vae-reconstruction
```

The Blender slider is independent of the server default above. After skinning,
either activate one bone vertex group while the mesh is active, or select
exactly one bone in Pose/Edit mode. The first nonzero level generates and
returns that bone's complete 0-3 reconstruction trajectory in one request.
The plugin caches the four one-dimensional fields for the current session,
so moving that bone's slider afterward is local. Each bone keeps its own level;
the selected fields are composed with the original skin and normalized before
they are applied. Vertex groups remain editable at every level. `Apply` keeps
the complete visible skin exactly as it is, clears the current trajectory, and
resets every bone to level 0. Selecting a nonzero level afterward starts a new
three-step reconstruction cycle from that committed result. A new Skin result
or Import TXT also clears the old trajectory cache.

The parent selector has been removed. Next leaves parent selection to the model,
so repeated clicks can naturally pop the DFS stack and create other branches.
Force Next captures the active Edit/Pose bone as the explicit parent; selecting
an older off-stack bone first focuses its DFS path without rebuilding existing
bones. Any generated bone becomes active. Bone custom properties hold a stable
joint ID, parent ID, and DFS order. Selecting an EditBone head or tail promotes
the active item to a full-bone selection. Edit-mode head and hierarchy changes
are automatically synced about 250 ms after placement. The hidden Refresh operator
remains available to scripts. Refresh trusts Blender's current parent links, so
native Delete keeps children that Blender reparents. Deleting a root from a
branched rig leaves multiple roots; Refresh rejects that invalid context without
erasing the remaining bones. The plugin Delete command provides predictable
leaf-delete and internal-dissolve behavior without requiring connected Blender
bones. Blender Undo and Redo read the restored Armature and synchronize that
exact hierarchy to the model service without changing modes inside the undo
callback.

## Runtime Lifecycle

- Session metadata maps a client session to the saved mesh asset and keeps only
  small counters/context metadata. It uses a separate LRU with a default limit
  of 512 records.
- A session idle for 3600 seconds is removed by the periodic cleanup task and
  reported with the `idle_timeout` end reason.
- GPU runtime tensors are kept in a fixed-size LRU cache with a default limit
  of 8. Runtime eviction is not a session end.
- An LRU miss rebuilds runtime tensors from the saved mesh before continuing
  with the complete skeleton sent by Blender.
- `Skin` does not special-case cache eviction; it follows the same LRU policy.
- `Finish`/`Reset` removes session metadata and runtime tensors. Session-record
  LRU eviction does the same for the least recently used session. Both keep the
  saved mesh.
- Mesh cleanup is outside the scope of this version.

## Usage Reporting

The model service appends two privacy-limited lifecycle events for each
session: `session_start` after Start has successfully loaded the mesh, and
`session_end` when Finish/Reset, session-record LRU eviction, addon unload, or
service shutdown releases it. Events are stored as daily UTC JSONL files under
`.runtime/usage/YYYY-MM-DD.jsonl` by default. Use `--usage-dir PATH` on either
model-server entry point to place them elsewhere.

Start records the anonymous client/session IDs, source IP address,
software/model versions, initial bone count, and vertex count. End records the
same source IP together with the final and maximum bone counts, final
added/removed bones, net and peak added bones, successful Skin count, whether
Skin ever succeeded, duration, and the end reason. Logs do not include OBJ/TXT
paths, filenames, bone names, joint coordinates, skin fields, or mesh geometry.
The HTTP service uses the socket peer address (`REMOTE_ADDR`) and does not trust
`X-Forwarded-For`; when deployed behind a reverse proxy, it therefore records
the proxy address unless trusted-proxy handling is added at deployment time.

Generate an aggregate report with:

```bash
uv run python -m interactive.usage_report
uv run python -m interactive.usage_report \
  --from 2026-07-01 --to 2026-07-31 \
  --output results/usage-2026-07.json
```

The report includes started/ended/incomplete sessions, unique source IP count,
end-reason counts, successful Skin uses, bone changes, and duration statistics.
A Start without a matching End, for example after a process crash, remains an
incomplete session instead of being counted as completed work.

## One-Shot Script Entry Point

`python -m infer` is a thin compatibility command for callers that need one
blocking OBJ-to-TXT subprocess. It is an HTTP client and never loads a model or
creates an `InteractiveModelServer`; the model service must already be running.
The command uploads the OBJ, runs Start, optional Sync or Rig, Skin, and Finish,
then writes the returned TXT on the client machine.

```bash
.venv/bin/python -m infer \
  --input examples/xiaobaozi.obj \
  --output results/xiaobaozi_skin.txt
```

By default infer derives the URL from `host` and `port` in
`.runtime/server.json`. Use `--server-url http://MODEL_MACHINE_IP:8765` when the
service is on another machine. The old server-side options such as `--device`
remain accepted temporarily for subprocess compatibility, but are ignored;
configure them on the running service instead.

Pass `--txt path/to/rig.txt` to use its complete skeleton instead of generating
one. The aliases `--skin-mode` and `--postprocess` remain available for simple
callers.

## Local Unix-Socket Compatibility

The previous local-only server remains available:

```bash
uv run python -m interactive.server --device cuda --max-runtime-sessions 8
```
