# Image Generation

TextualGrok supports image generation and editing via xAI's Grok Imagine service. Images can be produced in two ways: through the **Grok Imagine tool** (a function tool the model calls autonomously) and through **`<grok:render>` tags** (inline generation instructions embedded in a model response).

---

## Enabling image generation

Image generation is disabled by default. To enable it:

1. Open Settings (`Ctrl+P` → **Settings**).
2. Go to the **Image** tab.
3. Toggle **Enable Grok Imagine Tool** on.
4. Adjust other settings as needed (see below).
5. Click **Save**.

Once enabled, the `generate_image` function is registered in every API request. Grok will call it when you ask for visual output.

---

## Generating an image

With image generation enabled, ask Grok for an image in natural language:

```
Draw a minimalist logo for a terminal app called Grok.
```

```
Generate a photo-realistic image of a mountain lake at sunset, 16:9 aspect ratio.
```

Grok will call the `generate_image` function automatically. The result appears in the chat log as a URL or a note about base64 data, and the image thumbnail appears in the session image panel below the chat log.

---

## Editing an existing image

To edit an image you have already generated, describe the change you want:

```
Change the color of the sky to purple.
```

When **Use Last Generated Image For Edits** is enabled (the default), the most recently generated image is automatically passed as the source image. You do not need to provide a URL manually.

To edit a specific image, either:
- Set a source URL in **Settings → Image → Edit Source URL** (this persists for the session), or
- Reference the URL explicitly in your prompt.

---

## Image settings

Open Settings → **Image** tab to configure the following:

| Setting | Description | Default |
|---|---|---|
| Enable Grok Imagine Tool | Registers the `generate_image` function tool. | Off |
| Return Image As Base64 | Returns image data as base64 JSON instead of a URL. Useful when URLs might expire. | Off |
| Use Last Generated Image For Edits | Automatically passes the most recent image as the source for edit requests. | On |
| Grok Imagine Model | Which Grok Imagine model to use. Refresh models in the Chat tab to see available options. | `grok-imagine-image` |
| Image Count | Number of images to generate per call. Must be between 1 and 10. | 1 |
| Aspect Ratio | Output aspect ratio. See supported values below. | `1:1` |

### Supported aspect ratios

`1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`, `2:1`, `1:2`, `19.5:9`, `9:19.5`, `20:9`, `9:20`

Select `auto` to let the model choose based on the prompt.

---

## Viewing and saving generated images

Generated images appear in the **Session Image** panel below the chat log. Each image shows:
- A thumbnail (rendered inline in the terminal using `textual-image`)
- A numbered label and truncated source reference
- An **Open** button

Click a thumbnail or click **Open** to view the image full-screen in the image gallery. Press `Escape` or `Q` to return to the chat.

To save an image, use the export function from within the gallery. Images are saved to `exports/images/image-YYYYMMDD-HHMMSS.png`.

> **Temporary files:** Images are written to temporary files during the session. These files are deleted when you quit the app. Save any images you want to keep before quitting.

---

## `<grok:render>` tags

Some Grok models can emit `<grok:render>` tags directly in their response text to request inline image generation. TextualGrok detects these tags and executes the image generation call before rendering the response.

You do not need to do anything to handle these tags. If a response contains them, the app processes them automatically.

### Tag format

The model may use several formats inside a `<grok:render>` tag. TextualGrok supports:

**JSON body:**

```xml
<grok:render type="generate_image">
{"prompt": "a red panda", "n": 1, "aspect_ratio": "1:1"}
</grok:render>
```

**Line-based key-value pairs:**

```
<grok:render>
prompt: a red panda
n: 1
aspect_ratio: 16:9
</grok:render>
```

**XML sub-elements:**

```xml
<grok:render>
  <prompt>a red panda</prompt>
  <aspect_ratio>1:1</aspect_ratio>
</grok:render>
```

**Plain text body (treated as the prompt):**

```xml
<grok:render>a red panda</grok:render>
```

### Tag attributes

Parameters can also be set as HTML attributes on the tag itself:

```xml
<grok:render type="generate_image" prompt="a red panda" aspect_ratio="16:9">
</grok:render>
```

Body values take precedence over attribute values for the same parameter.

### Parameters recognized in `<grok:render>` tags

| Parameter | Type | Description |
|---|---|---|
| `prompt` | string | Image description (required). |
| `model` | string | Override the image model for this call. |
| `n` | integer | Number of images to generate (1-10). |
| `response_format` | `url` or `b64_json` | Override the output format. |
| `aspect_ratio` | string | Aspect ratio (see supported values above). |
| `source_image_url` | string | Source image URL for edit/variation requests. |

### Render tag behavior

- The tag content is replaced in the response text by a summary of what was generated (URL or a note about base64 data).
- If the tag has no recognizable prompt and no plain-text body, it is silently removed.
- Generated images are added to the session image panel alongside images from tool calls.

---

## How image generation works internally

When the `generate_image` function is called (either by the model or via a `<grok:render>` tag), TextualGrok:

1. Constructs a payload for the xAI `/images/generations` endpoint.
2. Sends the request with a 180-second timeout and up to 1 automatic retry on transient failures.
3. Extracts URLs or base64 data from the response.
4. If URLs were returned, downloads them to temporary files so they can be rendered inline.
5. Appends the images to the session image panel.

The model's configured settings (count, aspect ratio, format, source image) serve as defaults. Parameters passed in the function call or `<grok:render>` tag override those defaults on a per-call basis, with the exception of `response_format`: if the user has selected URL format in Settings, the app always forces URL output regardless of what the model requests.
