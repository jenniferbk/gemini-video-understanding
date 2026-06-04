# Installing Gemini Video Understanding (v1.2.0)

For Mac users (Apple Silicon — M1/M2/M3/M4).

## Step 1 — Download and open the DMG

Double-click `Gemini Video Understanding-1.2.0-arm64.dmg`. A window opens
showing the app icon and an Applications folder shortcut. Drag the app
icon onto the Applications folder.

## Step 2 — First launch (this is the unusual step)

The app is not signed with an Apple Developer ID, so the first time you
launch it macOS will show a warning that it "cannot be opened because
Apple cannot check it for malicious software." This is expected.

To get past it:

1. Open **Applications** in Finder.
2. **Right-click** (or Control-click) on **Gemini Video Understanding**.
3. Choose **Open** from the menu.
4. A dialog appears with an **Open** button. Click it.

You only need to do this once. After that, the app launches normally
from the Dock or Spotlight.

If macOS does not show an **Open** option (only "Move to Trash"), open
**System Settings → Privacy & Security**, scroll to the "Security"
section, and click **Open Anyway** next to the app's name.

## Step 3 — Add your Gemini API key

1. Launch the app.
2. Open **Settings** (gear icon, top right).
3. Paste your API key into the "Gemini API Key" field.
4. The key is stored in your macOS Keychain — it does not leave your
   machine.

You can get an API key at <https://aistudio.google.com/app/apikey>.

## Step 4 — Transcribe a video

1. Drag a video file onto the app, or click "Browse" to pick one.
2. On the Configuration screen, leave the defaults unless you have a
   reason to change them. The pipeline is tuned for classroom video.
3. Click **Detect Speakers** and review the visual descriptions.
4. Click **Start Transcription**.

A typical 1-hour video takes about 1 hour to transcribe and costs
about $0.19 in Gemini API credits.

## What's new in 1.2.0

- **De-identify Names** option (Configuration → Advanced). When enabled,
  the pipeline runs a second pass that replaces real student and teacher
  names in the transcript with realistic pseudonyms. An audit file
  (`transcript_name_map.json`) is written next to the transcript so you
  can recover the original mapping if you have legitimate access.
  - **Important:** the audit file contains the real-name ↔ pseudonym
    mapping. If your IRB requires deidentified storage, keep the audit
    file in a separately access-controlled location.
- Burn-in timestamps are always on (corrects intra-chunk timing drift).
- Chunk-1 timestamps now correct on videos where audio starts before
  the first video frame (a common YouTube re-encode pattern).

## Help

If you hit a problem, email Jennifer with:

1. Which video file you were transcribing.
2. What screen you were on when it failed.
3. Any error message text (screenshot is fine).
