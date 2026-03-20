# Beta Testing Guide for GVU v1.1.6

## For Beta Testers

Thank you for helping test Gemini Video Understanding v1.1.6!

This release fixes a critical bug where custom prompts were being ignored. Your testing helps ensure this fix works correctly.

---

## Installation

1. **Download the DMG:**
   - Get `Gemini Video Understanding-1.1.6-arm64.dmg` from Jennifer

2. **Install:**
   - Double-click the DMG
   - Drag "Gemini Video Understanding" to Applications folder
   - Eject the DMG

3. **First Launch:**
   - macOS may show "unidentified developer" warning
   - Go to System Preferences → Security & Privacy → Click "Open Anyway"
   - Enter your API key when prompted

---

## Critical Test: Custom Prompt Bug Fix

**This is the most important test for v1.1.6!**

### What We're Testing:
In previous versions, custom prompts were being ignored and the app would silently use the 'basic' prompt instead. This wasted $2-3 per video and gave incorrect results.

### How to Test:

1. **Create a custom prompt:**
   - Open the app
   - Go to Prompt Manager (or Settings)
   - Create a new prompt with a distinctive name like "My Test Prompt"
   - Add unique text that you'll recognize in the output, like:
     ```
     START OF CUSTOM PROMPT - This is my test prompt for beta testing.
     Please transcribe this video with extra attention to detail.
     END OF CUSTOM PROMPT
     ```
   - Save the prompt

2. **Process a test video:**
   - Use a SHORT video (2-5 minutes is fine)
   - Select "My Test Prompt" from the dropdown
   - Use these settings for quick testing:
     - Preset: Quick
     - Or manually: 1 consensus run, 3-minute chunks, no VAD, no denoising
   - Start transcription

3. **Verify the fix:**
   - When complete, open the transcript file
   - The Python script should have received your custom prompt
   - **PASS:** Output quality/style matches your custom prompt
   - **FAIL:** Output looks generic (basic prompt was used instead)

4. **Report results:**
   - Tell Jennifer: "Custom prompt worked!" or "Custom prompt was ignored"
   - If it failed, note which prompt you selected

---

## General Testing Checklist

Please test these if you have time:

### Basic Functionality
- [ ] App opens without crashing
- [ ] Can enter/save API key
- [ ] Can drag-and-drop a video file
- [ ] Video metadata shows (duration, size)
- [ ] Default prompts appear in list

### Prompt Management (if not already tested above)
- [ ] Can create a new custom prompt
- [ ] Can edit an existing prompt
- [ ] Can delete a prompt
- [ ] Prompts persist after quitting and reopening app

### Video Processing
- [ ] Progress bar updates during transcription
- [ ] Chunk counter shows progress (e.g., "5/12 chunks")
- [ ] Can cancel a job mid-process
- [ ] Job completes successfully
- [ ] Output file is created

### Results
- [ ] Transcript preview displays
- [ ] "Open Folder" button works
- [ ] Can copy transcript to clipboard
- [ ] Can start a new transcription

### Error Handling (Optional)
- [ ] If you enter an invalid API key, does it show a clear error?
- [ ] If you try to upload a non-video file (.txt, .pdf), is it rejected?

---

## What to Report

### If Everything Works:
Just send a quick message:
> "Tested v1.1.6 on my [Mac model] running [macOS version]. Custom prompts work correctly! ✅"

### If You Find Issues:
Please include:
1. **What you were doing** (step-by-step)
2. **What happened** (error message, unexpected behavior)
3. **What you expected** to happen
4. **Your system info:**
   - Mac model (e.g., "MacBook Pro M1, 2021")
   - macOS version (Apple menu → About This Mac)
   - App version (should be 1.1.6)

### Example Issue Report:
> **Issue:** Custom prompt not working
>
> **Steps:**
> 1. Created custom prompt "Small Group Science"
> 2. Selected it from dropdown
> 3. Processed test_video.mp4 (5 minutes)
> 4. Output transcript looks generic, doesn't match my prompt
>
> **System:**
> - MacBook Air M2, 2023
> - macOS Sonoma 14.5
> - GVU v1.1.6

---

## Known Issues (Not Bugs)

These are expected behaviors:

- **First launch warning:** macOS shows "unidentified developer" because the app isn't code-signed yet
  - **Fix:** System Preferences → Security & Privacy → "Open Anyway"

- **Long processing times:** A 45-minute video with standard settings takes ~90 minutes
  - **This is normal** - Gemini API processes video frame-by-frame

- **API costs:** Processing a 45-minute video costs ~$2-3
  - **Use short test videos** for beta testing to save money!

---

## Questions?

Contact Jennifer at [your email/Slack]

Thank you for helping make GVU better! 🙏

---

## Version Info

**Version:** 1.1.6
**Release Date:** November 4, 2025
**Critical Fix:** Custom prompt UUID matching
**Previous Fix (v1.1.5):** Prompt validation to prevent API waste
