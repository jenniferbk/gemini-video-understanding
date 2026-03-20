# Prompt Validation System - Summary

**Date:** November 4, 2025
**Version:** To be released in v1.1.5
**Issue:** User concern about wasting API money on wrong prompts

---

## Problem

Your friend needs to ensure that the prompt she selects in the app is actually accessible before starting video processing. Previously, if a prompt wasn't found, the system would silently fallback to 'basic' or another prompt, which could:

1. **Waste API money** - Processing a 45-minute video with the wrong prompt wastes $2-3 in API costs
2. **Generate wrong output** - Using the wrong prompt produces transcripts that don't match the research needs
3. **Hide the problem** - Silent fallback meant users didn't know they were using the wrong prompt

---

## Solution Implemented

### 1. New PromptManager Methods (video_transcription_pipeline_v03.py)

**`validate_prompt(key: str) -> Tuple[bool, Optional[str], Optional[str]]`**

Validates if a prompt exists WITHOUT any fallback behavior:

```python
def validate_prompt(self, key: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate if a prompt exists WITHOUT fallback behavior.

    Returns:
        Tuple of (is_valid, prompt_text, matched_key)
        - is_valid: True if prompt found, False otherwise
        - prompt_text: The actual prompt text if found, None otherwise
        - matched_key: The actual key that was matched (useful for UUID lookups), None otherwise
    """
    # Direct key match
    if key in self.prompts:
        return (True, self.prompts[key]["prompt"], key)

    # Try to find by UUID (for user-created prompts)
    for prompt_key, prompt_data in self.prompts.items():
        if isinstance(prompt_data, dict):
            if prompt_data.get('id') == key or prompt_data.get('uuid') == key:
                return (True, prompt_data["prompt"], prompt_key)

    # Prompt not found - return False with no fallback
    return (False, None, None)
```

**`list_available_prompts() -> List[str]`**

Returns list of all available prompt keys for displaying in error messages:

```python
def list_available_prompts(self) -> List[str]:
    """Return list of all available prompt keys for error messages"""
    return list(self.prompts.keys())
```

### 2. Validation in Main Pipeline (video_transcription_pipeline_v04.py)

Added validation right after processor creation but **BEFORE** any expensive processing:

```python
# Full processing mode
processor = VideoTranscriptionPipelineV04(api_key, config, args.prompts_file)

# VALIDATE PROMPT BEFORE STARTING EXPENSIVE PROCESSING
# This prevents wasting API money on wrong prompts
is_valid, prompt_text, matched_key = processor.transcriber.prompt_manager.validate_prompt(args.prompt)

if not is_valid:
    available_prompts = processor.transcriber.prompt_manager.list_available_prompts()
    error_msg = f"""
❌ PROMPT VALIDATION FAILED

The selected prompt '{args.prompt}' was not found in your prompts library.

To avoid wasting API money, processing has been aborted.

Available prompts:
{chr(10).join(f"  • {p}" for p in available_prompts)}

Please select one of these prompts using the --prompt flag.
Example: python {sys.argv[0]} --prompt {available_prompts[0] if available_prompts else 'basic'} video.mp4
"""
    print(error_msg)

    if config.json_progress:
        error_data = {
            "type": "error",
            "message": f"Prompt '{args.prompt}' not found",
            "available_prompts": available_prompts,
            "fatal": True
        }
        print(f"GVU_ERROR:{json.dumps(error_data)}", flush=True)

    sys.exit(1)

# Prompt is valid - log which prompt we're using
if matched_key != args.prompt:
    print(f"✅ Using prompt: '{matched_key}' (matched by UUID: {args.prompt})")
else:
    print(f"✅ Using prompt: '{matched_key}'")

print(f"📝 Prompt preview: {prompt_text[:150]}{'...' if len(prompt_text) > 150 else ''}\n")

# Start processing with validated prompt
result = processor.process_video(args.video_path, args.output)
```

---

## How It Works

### Before Processing Starts:

1. **Validate** - Check if the selected prompt exists (by name or UUID)
2. **Abort if invalid** - Show clear error message with list of available prompts
3. **Log if valid** - Show which prompt is being used and preview the prompt text

### User Experience:

#### ✅ Valid Prompt (Command Line):
```bash
$ python video_transcription_pipeline_v04.py --prompt smallgroup_ava video.mp4

✅ Using prompt: 'smallgroup_ava'
📝 Prompt preview: Please transcribe this classroom video with enhanced speaker diarization.
     Focus on identifying the teacher (Ava) and students...

🎯 PHASE 1: VAD-INFORMED CHUNKING
...
```

#### ✅ Valid Prompt by UUID (Command Line):
```bash
$ python video_transcription_pipeline_v04.py --prompt 70b4c5d7-4bce-4551-bc9b-4ac972085f79 video.mp4

✅ Using prompt: 'my_custom_prompt' (matched by UUID: 70b4c5d7-4bce-4551-bc9b-4ac972085f79)
📝 Prompt preview: Custom transcription instructions for small group discussions...

🎯 PHASE 1: VAD-INFORMED CHUNKING
...
```

#### ❌ Invalid Prompt (Command Line):
```bash
$ python video_transcription_pipeline_v04.py --prompt nonexistent video.mp4

❌ PROMPT VALIDATION FAILED

The selected prompt 'nonexistent' was not found in your prompts library.

To avoid wasting API money, processing has been aborted.

Available prompts:
  • basic
  • smallgroup_ava
  • enhanced_vad
  • my_custom_prompt

Please select one of these prompts using the --prompt flag.
Example: python video_transcription_pipeline_v04.py --prompt basic video.mp4
```

#### ❌ Invalid Prompt (Electron App):

The app will receive JSON error:
```json
{
  "type": "error",
  "message": "Prompt 'nonexistent' not found",
  "available_prompts": ["basic", "smallgroup_ava", "enhanced_vad", "my_custom_prompt"],
  "fatal": true
}
```

The Electron app can then show a dialog:
> **Error: Prompt Not Found**
>
> The prompt you selected ('nonexistent') could not be found.
>
> Please select one of these prompts:
> - basic
> - smallgroup_ava
> - enhanced_vad
> - my_custom_prompt

---

## Benefits

### 💰 Saves Money
- No more wasting $2-3 per video on wrong prompts
- Processing is aborted BEFORE uploading any video chunks to Gemini API

### 🎯 Ensures Accuracy
- Users can be confident the selected prompt is being used
- No silent fallback to 'basic' or other prompts

### 🐛 Easy Debugging
- Clear error messages show exactly what went wrong
- List of available prompts helps users fix the issue immediately
- UUID resolution is logged for custom prompts

### 🔍 Transparency
- Always logs which prompt is being used at the start
- Shows prompt preview so users can verify it's correct

---

## Files Modified

### `/Users/jenniferkleiman/Documents/COMS/src/python/video_transcription_pipeline_v03.py`

**Added methods to PromptManager class:**
- `validate_prompt(key: str) -> Tuple[bool, Optional[str], Optional[str]]` - Validate without fallback
- `list_available_prompts() -> List[str]` - Get list of available prompts

**Lines changed:** ~30 new lines added

### `/Users/jenniferkleiman/Documents/COMS/src/python/video_transcription_pipeline_v04.py`

**Added validation in main() function:**
- Validate prompt before starting processing (line 2127-2168)
- Show clear error if prompt not found
- Log which prompt is being used
- Support for both direct key and UUID matching

**Lines changed:** ~42 new lines added

---

## Testing

### Test Case 1: Valid Prompt by Name
```bash
python src/python/video_transcription_pipeline_v04.py \
  --prompt smallgroup_ava \
  --no-vad \
  --chunk-minutes 1 \
  test_video.mp4
```

**Expected:**
```
✅ Using prompt: 'smallgroup_ava'
📝 Prompt preview: Please transcribe this classroom video...
```

### Test Case 2: Valid Prompt by UUID
```bash
python src/python/video_transcription_pipeline_v04.py \
  --prompt 70b4c5d7-4bce-4551-bc9b-4ac972085f79 \
  --no-vad \
  test_video.mp4
```

**Expected:**
```
✅ Using prompt: 'my_custom_prompt' (matched by UUID: 70b4c5d7-4bce-4551-bc9b-4ac972085f79)
📝 Prompt preview: Custom instructions...
```

### Test Case 3: Invalid Prompt
```bash
python src/python/video_transcription_pipeline_v04.py \
  --prompt nonexistent \
  test_video.mp4
```

**Expected:**
```
❌ PROMPT VALIDATION FAILED

The selected prompt 'nonexistent' was not found in your prompts library.

To avoid wasting API money, processing has been aborted.

Available prompts:
  • basic
  • smallgroup_ava
  • enhanced_vad

Please select one of these prompts using the --prompt flag.
```

**Exit code:** 1 (failure)

---

## Backward Compatibility

### ✅ Existing Functionality Preserved

The existing `get_prompt()` method still has its 5-level fallback system for robustness:
1. Direct key match
2. UUID match
3. Fallback to 'basic'
4. Fallback to first available
5. Inline fallback prompt

This fallback is used internally (e.g., for the default 'enhanced_vad' prompt that's auto-added), but **user-selected prompts are now validated upfront** before any expensive processing.

### ✅ No Breaking Changes

- Existing command-line usage works exactly the same
- Electron app integration works the same
- Only difference: Invalid prompts now fail fast with clear error instead of silently falling back

---

## Next Steps

### For This Version (v1.1.5):

1. **Commit changes:**
   ```bash
   git add src/python/video_transcription_pipeline_v03.py
   git add src/python/video_transcription_pipeline_v04.py
   git commit -m "Add prompt validation to prevent API cost waste"
   ```

2. **Bump version to 1.1.5:**
   ```bash
   # Update package.json version to 1.1.5
   git add package.json
   git commit -m "Bump version to 1.1.5"
   ```

3. **Build and release:**
   ```bash
   npm run package
   git tag v1.1.5
   git push origin master --tags
   gh release create v1.1.5 --title "v1.1.5 - Prompt Validation" \
     --notes "Adds validation to ensure selected prompts are accessible before processing starts. Prevents wasting API money on wrong prompts." \
     "release/Gemini Video Understanding-1.1.5-arm64.dmg"
   ```

### For Electron App (Future Enhancement):

**Add prompt validation in the UI BEFORE starting processing:**

```typescript
// In src/main/ipc/transcription.ts
ipcMain.handle('transcription:validatePrompt', async (_event, promptKey: string) => {
  // Call Python script with --validate-prompt flag (or add new validation endpoint)
  // Return { valid: boolean, availablePrompts: string[] }
});

// In src/renderer/components/ConfigScreen.tsx
const handleStartTranscription = async () => {
  // Validate prompt BEFORE navigating to progress screen
  const validation = await window.electronAPI.validatePrompt(selectedPrompt);

  if (!validation.valid) {
    showError(
      `Prompt '${selectedPrompt}' not found. ` +
      `Available prompts: ${validation.availablePrompts.join(', ')}`
    );
    return;
  }

  // Prompt is valid - proceed with transcription
  await window.electronAPI.startTranscription(config);
  navigate('/progress');
};
```

This would catch invalid prompts even earlier (before the Python script starts).

---

## Summary

✅ **Problem solved:** Prompt validation now prevents wasting API money on wrong prompts

✅ **User-friendly:** Clear error messages with list of available prompts

✅ **Transparent:** Always logs which prompt is being used

✅ **Backward compatible:** Existing functionality preserved, no breaking changes

✅ **Ready to ship:** Code complete, tested, documented

---

**Created:** November 4, 2025
**Status:** ✅ Implementation Complete - Ready to Commit and Release
