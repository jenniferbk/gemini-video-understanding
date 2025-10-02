# Gemini Video Understanding - Installation Guide

## For Research Team Members

This guide will help you install and set up the Gemini Video Understanding application on your Mac.

---

## System Requirements

- **macOS:** 11.0 (Big Sur) or later
- **Mac Type:** Apple Silicon (M1, M2, M3) or Intel
- **Disk Space:** ~200 MB for the application
- **Internet:** Required for video transcription (uses Google Gemini API)

---

## Installation Steps

### Step 1: Download the Application

You should have received the file:
- **`Gemini Video Understanding-1.0.0-arm64.dmg`**

Save this file to your Downloads folder.

### Step 2: Install the Application

1. **Double-click** the `.dmg` file to open it
2. A window will appear showing the application icon and an Applications folder
3. **Drag** the "Gemini Video Understanding" icon to the Applications folder
4. **Close** the installer window
5. You can now **delete** the `.dmg` file from Downloads (optional)

### Step 3: First Launch

#### On First Open:

1. Open your **Applications** folder
2. Find **"Gemini Video Understanding"**
3. **Right-click** (or Control+click) on the app
4. Select **"Open"** from the menu
5. You'll see a security warning because the app isn't code-signed by Apple
6. Click **"Open"** to confirm

**Why this step?** macOS requires you to explicitly approve apps from unidentified developers on first launch. After this one-time approval, you can open the app normally by double-clicking.

#### Alternative Method (if Right-click doesn't work):

1. Try to open the app normally (double-click)
2. If blocked, go to **System Settings** → **Privacy & Security**
3. Scroll down to find a message about "Gemini Video Understanding"
4. Click **"Open Anyway"**
5. Confirm by clicking **"Open"**

---

## Initial Setup

### Get Your Google Gemini API Key

The app requires a Google Gemini API key to transcribe videos.

1. Visit: **https://aistudio.google.com/apikey**
2. Sign in with your Google account (use your UGA account if you have one)
3. Click **"Create API Key"**
4. **Copy** the key (it looks like: `AIzaSy...`)

### Enter Your API Key in the App

1. When you first open Gemini Video Understanding, you'll see a settings prompt
2. **Paste** your API key into the "Gemini API Key" field
3. Click **"Save"**

The app will securely store your key in your Mac's Keychain.

---

## Quick Start Guide

### Transcribe Your First Video

1. **Launch** Gemini Video Understanding
2. **Drag and drop** a classroom video onto the upload area, or click **"Choose Video"**
3. **Select a prompt** from the dropdown (e.g., "Small Group Discussion")
4. **Choose quality preset:**
   - **Quick:** ~30 minutes processing (1 consensus run)
   - **Standard:** ~90 minutes processing (3 consensus runs) - **Recommended**
   - **High Quality:** ~150 minutes processing (5 consensus runs)
5. Click **"Start Transcription"**
6. **Wait** for processing to complete (you can monitor progress)
7. When done, click **"View Results"** to see your transcript

### Supported Video Formats

- MP4
- MOV
- AVI
- MKV
- WebM

---

## Tips & Best Practices

### Cost Management

- Each transcription costs approximately **$2-4** depending on video length
- The app shows an estimate before you start
- Use **Quick** preset for testing, **Standard** for production work

### Video Quality

- Longer videos take more time and cost more
- For best results, use videos with clear audio
- Speaker diarization works best with 2-5 speakers

### Output Files

- Transcripts are saved to: **`~/Documents/VideoTranscripts/`**
- Each video gets its own folder with timestamped results
- Files are in plain text format, compatible with Transana

---

## Troubleshooting

### "App is damaged and can't be opened"

This happens if macOS security settings are strict:

1. Open **Terminal** (in Applications → Utilities)
2. Type: `xattr -cr "/Applications/Gemini Video Understanding.app"`
3. Press Enter
4. Try opening the app again

### "Invalid API Key" Error

1. Check that you copied the **entire** key from Google AI Studio
2. Make sure there are no spaces before or after the key
3. Try generating a **new** API key and entering it again

### Video Won't Upload

- Check that your video file is under **2 GB**
- Verify the file format is supported (MP4, MOV, AVI, MKV, WebM)
- Make sure you have internet connection

### Transcription Fails Mid-Process

- Check your internet connection
- Verify your API key hasn't hit quota limits
- Try reducing quality to "Quick" preset
- Contact Jennifer if the problem persists

---

## Getting Help

### Contact

- **Lead Developer:** Jennifer Kleiman
- **Project PIs:** AnnaMarie Conner, Xiaoming Zhai

### Feedback

Please share any issues, bugs, or feature requests with Jennifer. Your feedback helps improve the tool for everyone!

---

## About This Software

**Gemini Video Understanding** was developed at the University of Georgia's Department of Mathematics and Science Education as part of the C4OMS project and AI4STEM Center, funded by the National Science Foundation.

**Version:** 1.0.0
**Last Updated:** January 2025

---

## Appendix: Advanced Settings

### Custom Prompts

You can create custom transcription prompts for different video types:

1. Click the **ℹ️ (info)** icon on the home screen
2. Navigate to **Prompt Manager** (if available in future versions)
3. Or contact Jennifer to request custom prompts for your research needs

### Output Location

To change where transcripts are saved:

1. Go to **Settings** (gear icon)
2. Change **"Default Output Path"**
3. Choose your preferred folder

---

**Happy Transcribing! 🎬→📝**
