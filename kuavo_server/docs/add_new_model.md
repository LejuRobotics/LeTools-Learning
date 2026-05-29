# Add A New Model Repo

This document defines the standard workflow for adding a new external model repo
into the `kuavo_deploy <-> server <-> model` system.

## Architecture Boundary

Keep these boundaries fixed:

- `kuavo_deploy` stays robot-side only.
- `kuavo_deploy/kuavo_service/client.py` sends stable Kuavo payloads and does not
  contain model-specific logic.
- `kuavo_server/` is the only place where server runtime and model-specific
  conversion should live.
- External repos such as `VLA_pretraining/` or `LingBot-VLA/` keep their native
  model loading, preprocessing, and inference code.

## Standard Runtime

The validated runtime is:

1. Start the model environment.
2. Start `kuavo_server/serve.py` with an adapter.
3. Start the robot environment.
4. Start `kuavo_deploy` in `policy_type=client` mode.

The model environment and the robot environment can be different conda envs.

## Input Contract

All new adapters should accept the standard Kuavo payload produced by
`PolicyClient`:

- `observation.state`
- `observation.images.head_cam_h`
- `observation.images.wrist_cam_l`
- `observation.images.wrist_cam_r`
- `prompt`

If the target model needs different names, shapes, dtypes, temporal windows, or
camera layouts, do the conversion inside the adapter.

## Output Contract

Each adapter returns one Kuavo action for the current control step:

- `both` arms: expected 16-dim action when the model supports both arms
- `left` or `right`: expected 8-dim action when the runtime uses a single arm

If the model returns action chunks, the adapter owns chunk caching and must
expose `reset()` so the cache is cleared across episodes.

## New Model Checklist

### Step 1: Inspect the new model repo

Find:

- the native inference entrypoint
- the exact model input keys and shapes
- the exact output action shape
- whether the model has internal chunk execution or action buffering
- which paths are resolved relative to cwd and must be made absolute
- whether the repo relies on `uv`, `src/` layout, or workspace packages that must be added to `sys.path`

### Step 2: Create an adapter

Add a new file under `kuavo_server/adapters/`.

Recommended filename pattern:

- `lingbot_vla.py`
- `openvla_xxx.py`
- `my_model_repo.py`

Use `template_adapter.py` as the starting point.

### Step 3: Keep imports repo-local

The adapter should:

- resolve the external repo root
- prepend that repo root to `sys.path`
- prepend repo-local `src/` or workspace package paths when the model repo is not a flat package
- import the model repo's native inference class
- avoid copying model internals into `kuavo_data_challenge`

### Step 4: Convert obs and action in one place

Do all compatibility work inside the adapter:

- image layout conversion
- dtype conversion
- prompt mapping
- robot state flattening
- left/right/both arm slicing
- chunk-to-step conversion

Do not push these conversions into `kuavo_deploy`.

### Step 5: Register the adapter

Import the new adapter from `kuavo_server/builtin_adapters.py`.

This makes it available to:

- `python kuavo_server/serve.py --adapter ...`
- `python -m kuavo_server.serve --adapter ...`

If the external repo hardcodes a checkpoint-relative config layout, build a
runtime staging directory inside the workspace and place the expected config
there. `lingbot_vla.py` shows this pattern.

If the external repo already exposes a stable policy wrapper but still mixes
checkpoint-local and YAML-relative paths, resolve those paths inside the
adapter instead of relying on cwd. `wall_x.py` shows this pattern.

### Step 6: Add launch docs

Update:

- `kuavo_server/README.md`
- `AGENTS.md`

Document:

- which conda env starts the server
- the example serve command
- repo-specific assumptions
- known shape/path quirks

### Step 7: Validate

Validation order:

1. adapter import succeeds
2. `serve.py --help` shows the adapter
3. adapter dry run works if implemented
4. server boots and prints the listening endpoint
5. `kuavo_deploy` client can connect
6. one full sim workflow completes

## Common Failure Modes

### Wrong repo imported

Symptom:

- traceback points to another checkout or another installed package

Fix:

- ensure entry scripts prepend the current repo root to `sys.path`
- inspect `python -c "import kuavo_deploy.config as c; print(c.__file__)"`

### Relative path resolved against cwd

Symptom:

- files under `configs/`, `assets/`, or `norm.json` are reported missing

Fix:

- resolve those paths relative to the external repo root or the module file

### `uv` repo imports fail even though the repo is correct

Symptom:

- the repo uses `uv`, but imports such as `openpi` or `openpi_client` fail from `kuavo_server/serve.py`

Fix:

- start `serve.py` from the model repo's `uv` environment
- prepend repo-local `src/`, workspace package `src/`, and vendored dependency `src/` directories inside the adapter

### Model code assumes a different transformers version

Symptom:

- import branches choose unavailable classes

Fix:

- make the external repo code fallback safely
- keep this compatibility patch in the model repo or wrap it defensively in the adapter

### Action chunk leaks across episodes

Symptom:

- first actions of a new episode are stale

Fix:

- implement adapter `reset()`
- ensure `PolicyClient.reset()` hits the server reset endpoint

## LingBot-VLA Next

When onboarding `LingBot-VLA`, follow the exact same flow:

1. find the stable native inference class
2. create `kuavo_server/adapters/lingbot_vla.py`
3. keep `kuavo_deploy` unchanged
4. validate with the same two-env runtime
