# Hooks System for Auto-Skill Activation

This directory contains hooks that automatically activate Skills based on the context of your work.

## How It Works

Claude Code supports hooks that run on specific events:
- **UserPromptSubmit** - Runs before Claude processes your prompt
- **Stop** - Runs when you stop Claude mid-task

Hooks can analyze your prompt and file context to automatically load relevant Skills.

## Configuration

### Option 1: Claude Code Settings (Recommended)

Configure hooks via Claude Code's settings UI:

1. Open Claude Code settings
2. Navigate to Hooks section
3. Enable "User Prompt Submit" hook
4. Add logic to detect skill activation patterns

**Example Hook Logic:**
```typescript
// Check if working on Electron main process
if (context.activeFiles.some(f => f.includes('src/main/'))) {
  loadSkill('electron-dev-guidelines');
}

// Check if working on React components
if (context.activeFiles.some(f => f.includes('src/renderer/components/'))) {
  loadSkill('react-typescript-patterns');
}

// Check if working on Python integration
if (context.activeFiles.some(f => f.includes('src/main/python/'))) {
  loadSkill('python-integration');
}

// Check if working on database
if (context.activeFiles.some(f => f.includes('src/main/database/'))) {
  loadSkill('database-operations');
}
```

### Option 2: Manual Skill Invocation

If hooks aren't working, you can manually invoke skills:

```
@skill electron-dev-guidelines
@skill python-integration
@skill react-typescript-patterns
@skill database-operations
```

## Skill Activation Rules

The `skill-rules.json` file documents when each skill should activate:

### electron-dev-guidelines
**Activates when:**
- Working in `src/main/**/*.ts`
- Prompt mentions: IPC, Electron, main process, preload, browser window

### python-integration
**Activates when:**
- Working in `src/main/python/**/*.ts` or `src/python/**/*.py`
- Prompt mentions: Python, spawn, child process, venv, stdout

### react-typescript-patterns
**Activates when:**
- Working in `src/renderer/**/*.tsx`
- Prompt mentions: React, component, hooks, useState, Material UI

### database-operations
**Activates when:**
- Working in `src/main/database/**/*.ts`
- Prompt mentions: SQLite, database, SQL, query, transaction

## Troubleshooting

**Skills not auto-activating?**
1. Check that hooks are enabled in Claude Code settings
2. Verify file paths match patterns in `skill-rules.json`
3. Use manual invocation as fallback

**Too many skills loading?**
- Adjust priority levels in `skill-rules.json`
- Disable auto-activation for less-used skills
- Set `maxConcurrentSkills` to limit simultaneous skills

## Files

- `skill-rules.json` - Configuration reference for skill activation patterns
- `README.md` - This file
- `user-prompt-submit.ts` - UserPromptSubmit hook (if implemented)

## Notes

The hooks system helps maintain context and ensures you always have the right patterns and guidelines available when working on different parts of the codebase.

---

**Last Updated:** November 4, 2025
