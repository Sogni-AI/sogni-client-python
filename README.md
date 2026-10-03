# Sogni Client for Python

An async Python SDK for image, video, audio, and LLM inference on the Sogni
Supernet. It follows the public surface and wire protocol of the TypeScript
`sogni-client`, while using Python naming conventions and async iterators.

> The Python port is currently beta. Keep credentials in environment variables
> or your system keychain; never commit them to source control.

[Official quickstart](https://docs.sogni.ai/sogni-sdk/python/) ·
[Examples](https://github.com/Sogni-AI/sogni-client-python/tree/main/examples) ·
[Sogni API reference](https://docs.sogni.ai/api-reference/)

## Install

Install the latest release from PyPI:

```bash
python -m pip install --upgrade sogni-client
```

For an editable source checkout:

```bash
git clone https://github.com/Sogni-AI/sogni-client-python.git
cd sogni-client-python
python -m pip install -e .
```

Python 3.10 or newer is required.

## Create an image

```python
import asyncio
import os

from sogni_client import SogniClient


async def main() -> None:
    async with await SogniClient.create(
        api_key=os.environ["SOGNI_API_KEY"],
        app_id="my-image-app",
        app_source="my-app",
    ) as sogni:
        project = await sogni.projects.create(
            type="image",
            model_id="krea2_turbo_fp8_scaled",
            positive_prompt="A tiny observatory above a sea of clouds",
            negative_prompt="text, watermark",
            number_of_media=1,
            width=1024,
            height=1024,
            steps=8,
        )
        print(await project.wait_for_completion())


asyncio.run(main())
```

Socket clients require a stable `app_id`. Generate it once per application
installation and persist it across process restarts; do not generate a fresh
UUID each time the application starts. REST-only clients can omit it by passing
`disable_socket=True`.

The example uses **Krea 2 Turbo** (`krea2_turbo_fp8_scaled`) because it is the
only model an account's free monthly render credits can be spent on over the
API — every other model needs paid credits, so a brand-new key would otherwise
fail on its first call. It is an 8-step model, hence `steps=8`.

## Edit an image with Krea 2 Identity Edit

Worker image projects accept `output_format="png"`, `"jpg"`, or `"webp"`.
Set `embed_prompt_metadata=False` to omit embedded prompt and generation settings.
Image strength and seed values of `0` are preserved.

Pass one or two local reference images through `context_images`. For two-image
edits, place the base scene first and the identity or detail reference second.

```python
project = await sogni.projects.create(
    type="image",
    model_id="krea2_identity_edit_v1_2",
    positive_prompt=(
        "Change only the jacket to vivid sapphire blue. Preserve the exact "
        "facial identity, expression, framing, background, and lighting."
    ),
    number_of_media=1,
    width=1024,
    height=1024,
    steps=10,
    guidance=1,
    token_type="spark",
    context_images=["reference.png"],
)
print(await project.wait_for_completion(timeout=900))
```

The runnable example accepts one or two image paths and can also create a batch:

```bash
python examples/krea_identity_edit.py reference.png \
  --prompt "Change only the jacket to vivid sapphire blue; preserve identity."

python examples/krea_identity_edit.py scene.png identity.png \
  --prompt "Use the first image as the base scene and the second for identity." \
  --count 4
```

## Generate speech with Qwen3-TTS

Qwen3-TTS exposes three audio model IDs: studio voices, voice cloning, and
voice design. The prompt is the script to read aloud.

```python
project = await sogni.projects.create(
    type="audio",
    model_id="qwen3_tts_1.7b_custom_voice_bf16",
    positive_prompt="Every render on the Supernet runs on somebody else's GPU.",
    number_of_media=1,
    speaker="serena",
    instruct="warm and unhurried, close to the mic",
    output_format="mp3",
)
print(await project.wait_for_completion())
```

Voice Clone uses `qwen3_tts_1.7b_voice_clone_bf16` and requires a 3–30 second
`reference_audio` clip. Supply `reference_text` with the exact words spoken in
that clip whenever possible; the transcript is the strongest control on how
closely the clone preserves the source voice and accent. Voice Design uses
`qwen3_tts_1.7b_voice_design_bf16` and requires `instruct` to describe the
speaker to invent.

## Upscale a video with FlashVSR

`FLASHVSR_VIDEO_UPSCALE_MODEL_ID` (`flashvsr_v1.1_tiny_long_bf16`) upscales one
finished video to 1080p or 1440p on its short edge. It is promptless and
separate from video generation: it keeps every source frame, the exact frame
rate (including fractional rates such as 24000/1001), the full aspect ratio,
and the original audio, and it never trims, crops, restyles, or interpolates.

Sources must be at most 768px on the short edge and about 1344×768 pixels
overall (768×1344 in portrait), 1-60 fps at a constant frame rate, SDR, square
pixels with rotation applied, and 100 MB or less. The client sets no
frame-count or duration limit: the server enforces the maximum clip length and
refuses a source that is too long with a clear error. The output is at most
twice the source size, so 1080p needs a source short edge of at least 540px and
1440p at least 720px.
You do not send the source's frame count, frame rate, or size: the server
probes the upload and uses its verified values. `frames`, `fps`, `width`, and
`height` are optional, and any you do send must match the source.

```python
from sogni_client import FLASHVSR_VIDEO_UPSCALE_MODEL_ID

project = await sogni.projects.create(
    type="video",
    network="fast",
    model_id=FLASHVSR_VIDEO_UPSCALE_MODEL_ID,
    positive_prompt="",
    number_of_media=1,
    reference_video="clip.mp4",
    upscale_resolution=1440,  # or 1080: the output's short edge
)
print(await project.wait_for_completion())  # MP4 with the original audio
```

Three optional choices tune the render. `detail_preference` is `"stable"`
(default, More Stable) or `"sharper"`; `processing_speed` is `"stable"`
(default, More Stable) or `"faster"`; `seed` defaults to `0` for a repeatable
result, and `-1` asks for a random seed. Sharper, Faster and any seed other
than `0` or `-1` need a worker release that supports them; until one is
connected, the server refuses those requests.

To show a price first, call `estimate_video_cost()` with the output `width` and
`height` (the source scaled so its short edge equals the target, both edges
rounded to even pixels), the source's `frames` and `fps`, `steps=1`, and
`source_width`/`source_height`; the job itself is charged from the verified
source.

## MiniMax H3 two-stage output (720p, 1080p and 2K)

720p, 1080p and 2K MiniMax H3 two-stage output are the FastH3 Two-Stage model
ids, not a request option: `minimax-h3-fastvideo-int8_t2v_turbo_2stage`,
`minimax-h3-fastvideo-int8_i2v_turbo_2stage`,
`minimax-h3-fastvideo-int8_flf2v_turbo_2stage` and the audio-guide
`minimax-h3-fastvideo-int8_ia2v_turbo_2stage`,
`minimax-h3-fastvideo-int8_flfa2v_turbo_2stage` and
`minimax-h3-fastvideo-int8_a2v_turbo_2stage`, plus the Ref2VA Two-Stage ids
`minimax-h3-ref2va-fp8_r2v_2stage` (Standard, 20 steps) and
`minimax-h3-ref2va-fp8_r2v_balanced_2stage` (Balanced, 8 steps). Each FastH3 id
takes exactly the request of its FastH3 Turbo id (canvas, frames, 4 steps,
Euler/simple, inputs, LoRAs), and each Ref2VA id the request of its one-stage
R2V id (canvas, frames, steps, sampling, references, LoRAs).
The tier renders the canvas, then the worker enlarges it 2× and refines it, so the
clip is delivered at exactly twice the canvas width and height with the same
frame count, 24 fps timing and audio. Keep `width`/`height` on the normal H3 grid
and pick the canvas for the delivery you want:

| Choice | Canvas to send | Delivered |
| --- | --- | --- |
| 720p | chosen aspect at a 384 px short edge (1344×768 → 672×384) | 1344×768 |
| 1080p | chosen aspect at a 544 px short edge (1344×768 → 960×544) | 1920×1088 |
| 2K | the 768p canvas (1344×768) | 2688×1536 |

Portrait keeps the aspect: 384×672 delivers 768×1344, 544×960 delivers
1088×1920. Price it with `estimate_video_cost()` using the `_2stage` model id and
that canvas. `projects.create()` and `estimate_video_cost()` raise `ApiError` before
sending anything if the retired `output_scale`/`outputScale` is passed (the
server refuses it too), naming the two-stage ids to use. One `_2stage` id per
workflow serves every canvas class, so job history and cost reports show the
same id at 720p, 1080p and 2K (720p is priced like one-stage FastH3); the
short-lived `_2stage_720p` spellings were retired on 2026-09-14 and the socket
answers them with "Model not found". Only a `_2stage` id renders two-stage: a base
FastH3 id always runs one-stage at the canvas it sends. Hosted chat tools select these ids with
`minimax-h3-fasth3-turbo-2stage` (text or first frame),
`minimax-h3-fasth3-t2v-turbo-2stage`, `minimax-h3-fasth3-i2v-turbo-2stage` and
`minimax-h3-fasth3-flf2v-turbo-2stage`, and Ref2VA Two-Stage with
`minimax-h3-r2v-2stage` and `minimax-h3-r2v-balanced-2stage`; on those selectors `targetResolution`
names the delivered class (`720` renders the 384 px canvas, `1080` the 544 px
canvas, `1440` or omitted the 768p canvas for 2K).

```python
project = await sogni.projects.create(
    type="video",
    network="fast",
    model_id="minimax-h3-fastvideo-int8_t2v_turbo_2stage",
    number_of_media=1,
    steps=4,
    positive_prompt="integrated_multimodal_description: [Shot 1] ...",
    duration=8,
    width=1344,
    height=768,  # delivered at 2688x1536; send 960x544 for 1920x1088, 672x384 for 1344x768
)
```

## MiniMax H3 audio guide (image, first/last frame, or audio only)

The FastH3 audio guide drives the video with an uploaded `reference_audio` from
frame 0 and keeps that audio in the output, trimmed to the video length:

| Model id | Uploads |
| --- | --- |
| `minimax-h3-fastvideo-int8_ia2v_turbo` | `reference_image` + `reference_audio` |
| `minimax-h3-fastvideo-int8_flfa2v_turbo` | `reference_image` + `reference_image_end` + `reference_audio` |
| `minimax-h3-fastvideo-int8_a2v_turbo` | `reference_audio` only |

Each also has a `_2stage` id that takes the same request and delivers twice the
canvas. A mode refuses any upload it does not take. The optional `audio_start`
(seconds, 0 or greater) offsets the audio window; `loras` and `lora_strengths`
are accepted as on the frame modes (version 5.56.0 and later);
`generate_audio=False` and `audio_duration` are refused before anything is sent, and every other H3 id refuses
`audio_start`. `get_minimax_h3_frames_for_audio_duration(seconds)`
returns the smallest valid frame count covering the audio (124-362), and
`is_minimax_h3_audio_guide_model()` recognizes all six ids. The hosted
`sound_to_video` selectors are `minimax-h3-fasth3-ia2v-turbo`,
`minimax-h3-fasth3-flfa2v-turbo`, `minimax-h3-fasth3-a2v-turbo` and their
`-2stage` forms.

```python
from sogni_client import get_minimax_h3_frames_for_audio_duration

project = await sogni.projects.create(
    type="video",
    network="fast",
    model_id="minimax-h3-fastvideo-int8_flfa2v_turbo",
    number_of_media=1,
    steps=4,
    positive_prompt="The dancer crosses the studio in time with the music.",
    reference_image="first.png",
    reference_image_end="last.png",
    reference_audio="song.m4a",  # the output keeps this audio
    audio_start=12,
    frames=get_minimax_h3_frames_for_audio_duration(audio_seconds - 12),
    width=1344,
    height=768,
)
```

## MiniMax H3 intermediate keyframes

Available in version 5.58.0 and later.

Every H3 workflow except text-to-video accepts `keyframes`, 21 ids in all
(`is_minimax_h3_keyframe_model()`): image-to-video and first/last-frame on every
tier (Standard, Balanced, LightX2V Turbo, FastH3 Turbo and FastH3 Two-Stage), the
six FastH3 Sound to Video ids (ia2v, flfa2v, a2v, one- and two-stage) and the five
Ref2VA r2v ids. A request takes up to `MINIMAX_H3_MAX_KEYFRAMES` (8) still images
pinned at chosen frames between the first and last frame, alongside the
workflow's own uploads. Each entry is
`{"image": ..., "frame_index": ...}` (`frameIndex` works too). `frame_index` is the
0-based frame at 24 fps, an `int` from 1 to `frames - 2` of the job's frame count,
each frame used once. To convert seconds, use `int(seconds * 24 + 0.5)`, which
rounds halves up like the JavaScript SDK's `Math.round`; Python's `round()` rounds
halves to even.

Pass `frames` from the H3 grid (124, 141, 158, ... 362; `124 + n*17`) so the
frame count is exact. `duration` also works but snaps to the grid (`duration=6`
renders 141 frames, not 144), and `calculate_video_frames(model_id, seconds, 24)`
returns the count a duration resolves to.

Frame 0 and the last frame are never keyframes: i2v, flf2v and flfa2v set them
with `reference_image` / `reference_image_end` under their usual rules, ia2v sets
frame 0 with `reference_image`, and a2v and r2v cannot pin them. `context_images`
stays r2v-only. t2v and every non-H3 model refuse a non-empty list; an empty list
is ignored. Each image uploads to its own `keyframeImage1..N` slot in list order,
so an r2v request sends its references (`reference_image` and `context_images`,
which keep their slots) and its keyframes together. If no worker serving the
model can pin keyframes yet, the job is refused with error code `4100`. The
validation errors match the JavaScript SDK's word for word.

Pricing: the first two keyframes are included; each extra keyframe adds output
time at the job's per-second rate, 0.75 s on FastH3 and 0.3 s on every other
tier (an 8 s FastH3 clip with 8 keyframes: 32 + 18 = 50 Spark). Pass
`keyframe_count` (or the job's `keyframes`) to `estimate_video_cost` to quote it.

Writing the prompt for keyframes:

- Name each keyframe `<Picture N>` (MiniMax's keyframe format), numbered in time
  order after the workflow's own pictures: after the first frame `<Picture 1>` on
  i2v and ia2v (after the last frame `<Picture 1>` on a last-frame-only i2v job),
  after `<Picture 1>` (first) and `<Picture 2>` (last) on flf2v and flfa2v, from
  `<Picture 1>` on a2v, and after the last reference image `<Picture N>` on r2v.
  Keyframes are still not references and never count toward the reference
  limits.
- i2v, flf2v and Sound to Video prompts open with one alignment line listing
  every picture at its mark (`frame_index / 24` seconds, two decimals; a last
  frame at the clip's end), in time order, each credited to the shot on screen
  there: `How the reference pictures align with the target video — Picture 1
  (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2
  (from Shot 2) aligns with the 2.88-second mark of the target video.` The shot
  where a keyframe lands says "the shot's keyframe corresponds to
  `<Picture N>`".
- r2v adds `<Picture N> is the keyframe of [Shot M], showing ...` to
  `subject_definitions`, `keyframe completion` to the summary tasks (as in
  `[reference generation + keyframe completion]`), and
  `<Picture N> ([Shot M] keyframe): fully_preserved - ...` to
  `retention_analysis`.
- H3's text encoder never sees the keyframe images (they only pin frames), so
  the prompt must still describe what each keyframe shows at its time: shot
  size, camera angle, where each subject stands, pose, setting and light.
- On Sound to Video the uploaded audio drives the performance; keyframes pin how
  it looks at their times, so describe the look at each keyframe's time and let
  the audio carry timing and delivery.
- When a keyframe changes the framing, camera angle, location or light, start a
  new shot (a hard cut) at its time: `[Shot N] At MM:SS.mmm, the camera cuts to
  ..., whose keyframe corresponds to <Picture N>.`, where the time is
  `frame_index / 24` seconds (frame 144 is `00:06.000`). Two differently
  framed or lit stills inside one continuous shot cross-fade into each other, and
  a shot described differently from its still can flash the still for a single
  frame.

```python
# 192 frames (8 s): frame 0 is reference_image, frame 191 is reference_image_end.
# The keyframes are <Picture 3> and <Picture 4>, after the first and last
# frames. flf2v_prompt opens with the alignment line (Picture 1 at 0.00, Picture
# 3 at 2.50, Picture 4 at 6.00, Picture 2 at 8.00), says "the shot's keyframe
# corresponds to <Picture 3>" inside [Shot 1], and starts [Shot 2] At
# 00:06.000 with the new camera angle <Picture 4> shows.
project = await sogni.projects.create(
    type="video",
    network="fast",
    model_id="minimax-h3-fastvideo-int8_flf2v_turbo",
    number_of_media=1,
    steps=4,
    positive_prompt=flf2v_prompt,
    reference_image="first.png",
    reference_image_end="last.png",
    keyframes=[
        {"image": "same-shot.png", "frame_index": 60},  # 2.5 s, same framing and light
        {"image": "new-angle.png", "frame_index": 144},  # 6 s, new angle: the prompt cuts
    ],
    frames=192,  # on the 124 + n*17 grid, so frame_index may be 1-190
    width=1344,
    height=768,
)
```

## GPT Image 2.5

`gpt-image-2.5-flare` and `gpt-image-2.5-sunburst` join `gpt-image-2`. All three
accept up to 16 `context_images` references (never trimmed) and custom sizes up to
3840px. Quality must be a concrete value: `low`, `medium` or `high`, plus `xhigh`
and `max` on 2.5; `"auto"` is rejected because every request is quoted, charged and
rendered at the quality it names. 2.5 also supports `gpt_image_background="transparent"`
(PNG or WebP output only), and `gpt_image_output_compression` (0-100) applies to JPEG
or WebP output.

To edit part of the first reference, pass a PNG alpha mask as `gpt_image_mask`
(bytes or a path) or `gpt_image_mask_url` (a URL, or a `data:image/png;base64,...`
URI under 50 MB, which is uploaded like `gpt_image_mask`). Transparent mask
regions are edited. In chat tools, `gpt-image-2.5` and `flare` select Flare;
`sunburst` selects Sunburst.

## Seedance 2.5 export options

`seedance-2-5` can deliver a MOV container (`output_format="mov"`; video
defaults to `mp4`) and export a separate image of the final frame
(`return_last_frame=True`). The frame is available as `job.last_frame_url`, and
`await job.get_last_frame_url()` mints a fresh signed URL for it, ready to use as
the first frame of a follow-up clip. Both options are Seedance 2.5 only:
`projects.create()` raises `ApiError` before sending anything for another model,
for an output format other than `mp4`/`mov`, or for a non-boolean
`return_last_frame`. Chat tool results list `lastFrameUrls` when a frame was
exported.

## Reusable subscriber uploads

On servers that support saved uploads, eligible subscribers reuse the same image,
video or audio file across projects. Pass files to `projects.create()` as usual:
the client checks the account's saved copies by SHA-256 before transferring bytes,
so a repeated reference is not uploaded again. Uploads stay private to the
signed-in account.

```python
saved = await sogni.projects.assets.upload(open("product.png", "rb").read(), "image/png", "Product")
listing = await sogni.projects.assets.list()  # {"assets": [...], "limits": {...}}
await sogni.projects.assets.remove(saved["id"])  # already-bound project inputs stay
```

Older servers and ineligible accounts keep using ordinary project uploads, but
only when saved storage cannot be prepared; a transfer, checksum or binding
failure after preparation stops project submission. Saved IDs do not replace file
parameters in `projects.create()`; `assets.bind(id, {"projectId": ..., "type": ...})`
is available for callers that manage project input slots directly. Project history
may include `byolUsed`, `personalLoras` (public-source snapshots) and
`reusedAssetCount`; missing fields on older projects mean unknown, not zero.

## Personal LoRAs

Personal imports belong to the authenticated account. Discover compatible models
and limits through the library instead of hard-coding them:

```python
library = await sogni.projects.personal_loras.list()
catalog = await sogni.projects.available_loras(include_personal=True)
ready = await sogni.projects.personal_loras.catalog(model_id="krea2_turbo_fp8_scaled")
```

Use `personal_loras.import_lora(url=..., name=..., model_id=...,
rights_confirmed=True)` only after confirming permission to use the file. Imports
are asynchronous and take minutes; check `personal_loras.get(imported["id"])`
about every 30 seconds until the status is `ready`, `rejected`, or `revoked`.
On a 429, wait `ApiError.retry_after` seconds before checking again. Remove an entry with
`personal_loras.remove(imported["id"])`.

Use ready catalog IDs with their listed model compatibility and strength ranges.
The private catalog is fetched afresh and never enters the public catalog cache.
Importing, ready-catalog discovery, and generation require the server's active
subscription entitlement; library inspection and removal remain available after
it lapses. JavaScript-style `personalLoras` and `includePersonal` aliases are also
supported; Python uses `import_lora` because `import` is a language keyword.

## Chat

Socket-backed completion:

```python
result = await sogni.chat.completions.create(
    model="qwen3.6-35b-a3b-gguf-iq4xs",
    messages=[{"role": "user", "content": "Give me three visual concepts."}],
)
print(result["content"])
```

Hosted OpenAI-compatible completion:

```python
result = await sogni.chat.hosted.create(
    model="qwen3.6-35b-a3b-gguf-iq4xs",
    messages=[{"role": "user", "content": "Describe a surreal album cover."}],
)
```

For streaming socket chat, pass `stream=True` and iterate over the returned
`ChatStream` with `async for`.

## Durable chat runs and cost approval

`sogni.chat.runs` (`create`, `get`, `cancel`, `confirm_cost`, `stream_events`)
wraps `/v1/chat/runs`, where the server drives the LLM and tool loop. A run can
pause before paid tool calls with `status == "waiting_for_user"` and
`waiting["reason"] == "cost_approval_required"`. The pause carries the paused
`toolCallId` and a `costApprovalPreview` in `run["waiting"]["details"]`, and in
`event["payload"]["details"]` on the `run_waiting_for_user` event. Show that
preview to the user, then pass it back unchanged:

```python
run = await sogni.chat.runs.get(run_id)
waiting = run.get("waiting") or {}
details = waiting.get("details") or {}

if waiting.get("reason") == "cost_approval_required" and details.get("toolCallId"):
    preview = details.get("costApprovalPreview")
    if preview and user_approved(preview):  # user_approved is your UI
        await sogni.chat.runs.confirm_cost(
            run_id,
            tool_call_id=details["toolCallId"],
            decision="confirm",
            accepted_cost_preview=preview,
            idempotency_key=f"confirm-{details['toolCallId']}",
        )
    else:
        # Declining needs no preview.
        await sogni.chat.runs.confirm_cost(
            run_id, tool_call_id=details["toolCallId"], decision="cancel"
        )
```

The server rejects a confirm without `accepted_cost_preview` (HTTP 400), and one
whose preview has expired or no longer matches (HTTP 409); read the run again
and ask the user to approve the new preview. The client never fills in the
preview for you. `idempotency_key` is sent as the `Idempotency-Key` header, so
reuse it for duplicate submissions of the same decision. `overrides` and
`reason` are also accepted.

## Durable workflows

```python
workflow = await sogni.workflows.start(
    input={"prompt": "Create a four-panel character turnaround"},
    idempotency_key="turnaround-001",
)

async for event in sogni.workflows.stream_events(workflow["id"]):
    print(event["event"], event["data"])
```

The client also exposes:

- `sogni.account` for authentication, balances, rewards, transactions, and subscriptions
- `sogni.projects` for generation, uploads, model discovery, and estimates
- `sogni.chat` for socket, hosted, tool, and durable-run APIs
- `sogni.workflows` and `sogni.workflows.templates`
- `sogni.replay` and `sogni.stats`

Python `snake_case` arguments are preferred. Common JavaScript-style aliases
remain accepted to simplify migration.

### Generation failure categories

`job.error`, `project.error`, error events, and `ProjectError.error` preserve
the optional `vendorFailureCategory` string sent by the server. Known values
include `content_policy`, `input_validation`, `timeout`, `result_storage`,
`cancelled`, `vendor_failed`, `asset_resolution`, and `vendor_transient`.
Servers may add categories; use a generic failure message for unknown strings.

### Retrying safely: `retry_after`, `details`, and idempotency keys

A refused REST request raises `ApiError` with the HTTP `status`, the server's
message, and the error body as `payload`. When the server says how long to
wait (a `429`, or a `503` while it restarts), `error.retry_after` carries that
wait **in seconds**, read from the response body or, failing that, the
`Retry-After` header. `error.details` carries any structured context the server
attached, such as the active-workflow count behind a `409`. Both are `None`
when the server sent neither. A REST chat error raised as `ChatJobError`
carries the same two attributes.

Wait at least `retry_after` seconds before trying again; a request sent sooner
is refused again. A `409` for too many active workflows clears when one of your
workflows finishes, so wait for a completion (`stream_events()` or `get()`)
rather than re-sending the start.

Pass `idempotency_key` to `start()` and `reseed()` and reuse it when you retry
a request that timed out or lost its connection. The retry returns the workflow
the first request created instead of starting, and billing, another one. A
reseed mints new random seeds, so use a new key for each take you actually
want; a replayed reseed comes back with `"idempotent": True`.

```python
import asyncio
import uuid

from sogni_client import ApiError


async def start_with_retry(sogni, attempts=5, **params):
    idempotency_key = str(uuid.uuid4())
    for attempt in range(1, attempts + 1):
        try:
            return await sogni.workflows.start(idempotency_key=idempotency_key, **params)
        except ApiError as error:
            if error.retry_after is None or attempt == attempts:
                raise
            await asyncio.sleep(error.retry_after)
```

## Resuming projects after a reconnect

Generation keeps running on the Supernet while your socket is down. A dropped
connection is a transport gap, not a failure: tracked projects stay alive, the
client reconnects with capped exponential backoff for as long as the session is
authenticated, and on every `authenticated` handshake it reconciles with the
server. The backoff resets only after the server authenticates the connection,
so repeated opens that close before authentication keep increasing the delay.
Whatever the client missed is replayed through the normal `project` /
`job` events, so listeners attached before the gap keep receiving updates and
`wait_for_completion()` still resolves.

Projects the server knows about but this client does not (a restart, a second
client sharing the account, cleared local state) are rebuilt as tracked
`Project` instances with `project.recovered is True`. Their `params` are
reconstructed from the original request; asset inputs are not recoverable.

```python
# Every reconciliation reports what changed. `snapshot` is the raw server view,
# for apps that keep their own project store.
sogni.projects.on("projectsSynced", lambda r: print(r["reason"], r["active"], r["lost"]))

# In-flight projects this client was not tracking; they are tracked now, so
# `project` / `job` events follow as usual.
sogni.projects.on("activeProjectsRecovered", lambda projects: ...)

# Projects that finished while this client was away, result URLs already resolved.
sogni.projects.on("completedProjectsRecovered", lambda projects: ...)

# Ask for a fresh reconciliation yourself, e.g. after waking from sleep.
await sogni.projects.sync()
```

A project the server no longer lists is looked up on the REST API (which only
stores finished projects) a few times before it is declared lost; it then fails
with an error where `is_project_lost_error(error)` is `True`. Apps that persist
project ids themselves can run the same lookup with
`sogni.projects.resolve_missing(ids)`.

### Socket server restarts

A Sogni platform release restarts the socket server: every connection closes
with code `1001` for a few seconds. The SDK is built so apps need no special
handling for it:

- `create()` and chat requests made during the gap wait (up to 30 seconds) for
  the reconnected, authenticated socket instead of failing.
- A project request that reached the server while it was shutting down is
  refused by id; the SDK sends the same request again after reconnecting, once.
  Projects created moments before a reconnect are re-checked when they become
  old enough to judge, rather than minutes later.
- LLM jobs are not carried across a restart. The server refunds them, and a
  stream that was open fails with a `ChatJobError` whose `retryable` is `True`
  (`error_type` `"server_restarting"` or `"transport_lost"`) rather than waiting
  forever. After a plain network blip the server keeps the job for 30 seconds
  and the stream simply continues. Re-issue retryable failures as new requests:

```python
from sogni_client import is_retryable_chat_error


async def complete_with_retry(**params):
    try:
        return await sogni.chat.completions.create(**params)
    except Exception as error:
        if not is_retryable_chat_error(error):
            raise
        return await sogni.chat.completions.create(**params)  # waits for the reconnect
```

The same snapshot answers "is anything rendering elsewhere on this account?" —
`sogni.projects.list_projects_elsewhere()` returns those in-flight projects
read-only (`appSource`, `status`, `model`, per-job step counts). The socket
rate-limits it to 20 calls per 10s per account, so poll on the order of tens of
seconds.

Recovery is per app instance: the server hands projects back to the `appId` that
created them, so persist your `appId` and reuse it across restarts.

### Results after you stopped waiting

Available in version 5.58.0 and later.

The socket holds a project that finished while its client was disconnected for
one hour. A client that restarts, or a script or agent that exits before its
projects finish, can still collect them:

- `sogni.projects.get_result(project_id)` returns a project's state and its
  renders at any time: while it is queued (with the server's `waitingReason`,
  which says whether the account's own plan concurrency is holding it or it is
  waiting for a worker) and after it finished, with signed download URLs for the
  completed renders. Pass `kind="video"` (or `"image"`, `"audio"`, `"model"`)
  when you know what it produces and the model is not in this client's catalog.
- `sogni.projects.list_recent(since=...)` lists this account's recently completed
  media projects, newest first, from the durable history (up to 7 days back, 24
  hours by default), including ones that finished while no client was
  connected. `since` is a `datetime` or milliseconds since the epoch; `limit`
  (1-100, default 50) and `app_source` narrow it.

Use `get()`, `get_status()`, and `get_result()` for one-off reads. Wait for a
tracked project with `project.wait_for_completion()` or its `completed` /
`failed` events over the socket. After a restart, reconnect with the same
`app_id` and call `projects.sync()`; `resolve_missing(ids)` covers projects the
socket no longer holds. Each read uses the per-IP request allowance, and
`get_result()` also signs a download URL for each completed render. On a 429,
wait `ApiError.retry_after` seconds before the next request.

```python
from datetime import datetime, timedelta, timezone

since = datetime.now(timezone.utc) - timedelta(hours=6)
for project in await sogni.projects.list_recent(since=since):
    result = await sogni.projects.get_result(project["id"])
    for job in result["jobs"]:
        if job.get("url"):
            print(project.get("modelName"), job["url"])

pending = await sogni.projects.get_result(project_id)
if not pending["finished"]:
    print(pending["status"], (pending.get("waitingReason") or {}).get("message"))
```

A completed render without a `url` says why in `urlUnavailable`:
`sensitiveContent` (the Sensitive Content Filter withheld it), `unknownMediaKind`
(neither the model nor the result says what media it is; pass `kind`), or
`downloadUrlFailed` (the API could not sign one; ask again later). Failed and
cancelled renders keep their `reason`.

## Queue explanations

Available in version 5.55.0 and later.

`project.waiting_reason` and `project.job_waiting_reasons` describe why queued
work is waiting. Subscribe to `sogni.projects.on("queueChanged", callback)` for
updates with `projectId`, `waitingReason`, and `jobWaitingReasons`. Existing
`project` and `job` events keep their meanings; a partial batch can be processing
while another result waits.

Each reason includes a server-provided plain-text `message` and a `reason` code:
`concurrency_limit`, `model_concurrency_limit`, `payment_pending`, `no_workers`,
or `queued`. These describe the current wait, not progress or an ETA. Each
per-result entry has a zero-based `jobIndex`; `imgID` is optional until a worker
starts. Reading queue entries does not create placeholder jobs. Known pending
jobs also expose `job.waiting_reason`. Camel-case aliases and JSON serialization
use `waitingReason` and `jobWaitingReasons`.

Queue state clears when results start or finish, on explicit null/empty updates,
and when an entry is omitted from the server's complete current list. Older
servers simply leave these fields empty. The SDK automatically requests the
`projectQueue` socket subscription; integrations may explicitly disable it with
`socket_event_subscriptions={"projectQueue": False}`.

## Announcements

Admin-authored in-app announcements — maintenance notices, launches — arrive on
the `appAlert` socket event. It is opt-in, so an integration that does not ask
for it is unaffected:

```python
sogni = await SogniClient.create(
    api_key=os.environ["SOGNI_API_KEY"],
    app_id="my-announcements-app",
    app_source="my-app",
    socket_event_subscriptions={"appAlert": True},
)

sogni.api_client.on("appAlert", lambda announcement: print(announcement["title"]))

# What is live right now, for a client that just started up.
for announcement in await sogni.announcements.active("my-app"):
    print(announcement["title"], announcement["bodyMarkdown"])

# Dismissal is stored per ACCOUNT, so it sticks across the user's devices.
await sogni.announcements.dismiss(announcement["id"])
```

`appAlert` is **not** at-most-once: a live pinned announcement is re-sent on
every reconnect, so a user who was offline when it published still receives it.
Deduplicate on `id`.

## Segmentation and 3D models

These workflows transform a source image instead of generating from a prompt, so
each needs a `starting_image`. Ask the SDK rather than hardcoding model ids:
`requires_starting_image()`, `is_segmentation_model()`,
`is_model_artifact_model()`, `is_pixal3d_model()` and
`is_pixal3d_multiview_model()`, alongside the `SAM3_IMAGE_SEGMENT_MODEL_ID`,
`PIXAL3D_IMAGE_TO_3D_MODEL_ID` and `PIXAL3D_MULTIVIEW_IMAGE_TO_3D_MODEL_ID`
constants.

SAM 3 returns one lossless mask PNG the same size as the source. The request
carries a bounded `sam3_prompt`: `points` (`label` `positive`/`negative`),
`boxes` (a `negative` box excludes one instance of a text-prompted concept and
requires `text`), `text`, `threshold`, `multimask` (point prompts only),
`apply_mask` (return the selection cut out as RGBA instead of the bare mask),
and `max_instances` (1 to 16). Coordinates are normalized from 0 to 1.

```python
from sogni_client import SAM3_IMAGE_SEGMENT_MODEL_ID

project = await sogni.projects.create(
    type="image",
    model_id=SAM3_IMAGE_SEGMENT_MODEL_ID,
    positive_prompt="",
    number_of_media=1,
    starting_image="room.png",
    sam3_prompt={"text": "the teapot", "apply_mask": True, "max_instances": 1},
)
```

Pixal3D returns a binary glTF, so `job.type` is `"model"` and the artifact
downloads as `model/gltf-binary`. Four options — `texture_size`,
`mesh_target_faces`, `normal_map_size`, and `ambient_occlusion_size` — are
reduce-only and default to their maximum. `shape_resolution` defaults to 1024
and can be raised to the priced 1536 maximum-detail step.
`mesh_target_faces` is the one worth setting: the 700,000-triangle default is
far heavier than a real-time engine wants.

`pixal3d_int8_i23d` reconstructs from `starting_image` alone.
`PIXAL3D_MULTIVIEW_IMAGE_TO_3D_MODEL_ID` (`pixal3d_multiview_int8_i23d`) takes
`starting_image` as the required FRONT view plus any subset of three optional
orbit views, each uploaded in a fixed slot. The views must show the same object
at the same height, 90 degrees apart around it at eye level, like a character
turnaround sheet. Name them from the subject's own point of view, not the
viewer's:

| Keyword | What the image shows | Upload slot |
|---------|----------------------|-------------|
| `starting_image` | Front view (required) | `startingImage` |
| `left_view_image` | The subject turned so **its own left side** faces the camera (it faces screen-left) | `contextImage1` |
| `back_view_image` | The subject seen from behind | `contextImage2` |
| `right_view_image` | The subject turned so **its own right side** faces the camera (it faces screen-right) | `contextImage3` |

Swapping left and right builds a model turned 180 degrees. Some turnaround
templates label the photo of the subject's right side "left"; follow the table,
not those labels. The single-view model refuses orbit views, both models refuse
`context_images`, and only the single-view model accepts `template_variant`.
Both take the options above.

```python
from sogni_client import PIXAL3D_MULTIVIEW_IMAGE_TO_3D_MODEL_ID

project = await sogni.projects.create(
    type="image",
    model_id=PIXAL3D_MULTIVIEW_IMAGE_TO_3D_MODEL_ID,
    positive_prompt="",
    number_of_media=1,
    starting_image="front.png",
    left_view_image="left.png",  # optional
    back_view_image="back.png",  # optional
    right_view_image="right.png",  # optional
    mesh_target_faces=200_000,
)
```

When a workflow attests its inputs and outputs, `job.provenance` carries the
worker-signed receipt. Like `job.error` and `project.params`, it is the wire
record, so its keys stay camelCase: lowercase SHA-256 digests (`sha256`,
`sourceImageSha256`, `samPromptSha256`, `maskRleSha256`) plus, for SAM 3,
`maskBox`, `maskCoverage`, and the per-selection report (`maskDetectedCount`,
`maskReturnedCount`, `maskSelections`) that tells a confident selection from a
marginal one. Malformed entries are dropped rather than surfaced half-valid.

## Sensitive content

`job.is_nsfw` means the server **withheld** the media: the render ran with the
Sensitive Content Filter on, a signal fired, and there is nothing to download.
When the artist turns the filter off the media is delivered and merely labelled
— that case reports `job.nsfw_detected` with `job.nsfw_sources` (`prompt`
and/or `image`), has a `result_url` like any other result, and leaves
`job.is_nsfw` false. Use `job.has_result_media` (or `job.is_withheld`) to decide
whether media exists, and the viewer's own filter setting to decide whether to
blur it.

## Compatibility

This release tracks the TypeScript SDK at `5.60.3`. The
REST, WebSocket, and SSE contracts are covered by credential-free protocol
tests, including authentication refresh, uploads, project state recovery,
streaming chat, workflows, templates, replay, and the canonical 30 hosted-tool
schemas.

Current model and transport coverage includes LTX 2.5, MiniMax H3 in all four
tiers (Standard, 8-step Balanced, 4-step LightX2V Turbo, and the separate
FastH3 `fastvideo-int8` Turbo engine with its audio-guide ia2v/flfa2v/a2v modes and
Two-Stage 720p/1080p/2K ids, plus intermediate keyframes on every mode but t2v),
Seedance 2.5, Wan 3 and Wan 3.0 Enhanced,
RTX VSR, MiniMax Music 3, Qwen3-TTS speech and voice cloning, SAM 3 image
segmentation, Pixal3D image-to-3D, FlashVSR v1.1 promptless video upscaling,
LoRA catalog discovery, queue start estimates,
live-benchmarked render/total time on cost quotes, in-flight project recovery
across reconnects, results by project id and recently completed projects from
the durable history, confirmed cancellation, connection/workload attribution, and
admin announcements (`appAlert` plus the announcements read/dismiss pair).

The Python API is async-first; `AsyncSogniClient` is an alias of
`SogniClient`, not a synchronous wrapper. Browser-only cookie coordination and
multi-tab behavior have no Python equivalent. Local image references are
uploaded with their detected MIME type, but the TypeScript client's optional
browser-side image resizing is not reproduced. All 30 canonical tool schemas
are exposed; the local project-backed executor handles the six direct media
generation tools, while the remaining tools run through the hosted or durable
chat APIs. Live, credentialed smoke tests are intentionally separate from the
default test suite.

## Token authentication

```python
sogni = await SogniClient.create(app_id="my-token-app", auth_type="token")
await sogni.set_tokens(token=access_token, refresh_token=refresh_token)
```

Username/password login and signing are available through `sogni.account.login`.
API-key use does not require storing a wallet password.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check sogni_client tests
ruff format --check sogni_client tests
python -m build
```

Live integration tests require explicit credentials and are not run by default.

## Documentation

- [Python SDK quickstart](https://docs.sogni.ai/sogni-sdk/python/)
- [Sogni SDK overview](https://docs.sogni.ai/sogni-sdk/)
- [REST API reference](https://docs.sogni.ai/api-reference/)
