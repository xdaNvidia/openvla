# Tmux Configuration Guide

## Overview
This tmux configuration is optimized for use in Cursor's integrated terminal with mouse support, proper scrolling, and clipboard integration.

## Features
- Mouse support (scrolling, selecting, resizing, copy/paste)
- Proper scroll behavior in copy mode
- Clipboard integration with system clipboard
- Vi-style copy mode
- 50,000 line scrollback buffer
- True color support
- Fast key response (no escape delay)

## Essential Shortcuts

### Pane Management

**Create Panes:**
- `Ctrl+b` then `|` - Split horizontally (left/right)
- `Ctrl+b` then `-` - Split vertically (top/bottom)

**Navigate Between Panes:**
- `Ctrl+Arrow keys` - Switch panes directly (no prefix needed)
- `Ctrl+b` then `h` - Move to left pane
- `Ctrl+b` then `j` - Move to bottom pane
- `Ctrl+b` then `k` - Move to top pane
- `Ctrl+b` then `l` - Move to right pane

**Close Panes:**
- `Ctrl+d` or type `exit` - Close current pane
- `Ctrl+b` then `x` - Kill pane (asks for confirmation)

**Resize Panes:**
- Use mouse to drag pane borders
- `Ctrl+b` then `z` - Toggle zoom (fullscreen) current pane

### Window Management

**Switch Windows:**
- `Ctrl+b` then `1`-`9` - Jump to window number
- `Ctrl+b` then `n` - Next window
- `Ctrl+b` then `p` - Previous window
- `Ctrl+b` then `w` - List all windows (interactive)

**Create/Close Windows:**
- `Ctrl+b` then `c` - Create new window
- `Ctrl+d` or `exit` - Close current window

**Rename Window:**
- `Ctrl+b` then `,` - Rename current window

### Copy/Paste

**Using Mouse:**
- Click and drag to select text - automatically copies to clipboard
- Middle-click or `Ctrl+Shift+V` to paste

**Using Keyboard (Vi Mode):**
1. `Ctrl+b` then `[` - Enter copy mode
2. Navigate with arrow keys or `h`/`j`/`k`/`l`
3. Press `v` to start selection
4. Navigate to select text
5. Press `y` to yank (copy) to clipboard
6. Press `q` or `Escape` to exit copy mode

**Paste:**
- `Ctrl+b` then `p` - Paste from tmux buffer
- `Ctrl+Shift+V` - Paste from system clipboard

### Scrolling

**Using Mouse:**
- Scroll up/down with mouse wheel (works like normal terminal)

**Using Keyboard:**
1. `Ctrl+b` then `[` - Enter copy mode
2. Use arrow keys, `Page Up`, `Page Down` to scroll
3. Press `q` or `Escape` to exit

### Session Management

**Detach/Attach:**
- `Ctrl+b` then `d` - Detach from session (tmux keeps running)
- `tmux attach` or `tmux a` - Reattach to last session
- `tmux attach -t <session-name>` - Attach to specific session

**List Sessions:**
- `tmux ls` - List all sessions
- `Ctrl+b` then `s` - Interactive session list

**Create Named Session:**
- `tmux new -s <name>` - Start new session with name

**Kill Session:**
- `tmux kill-session -t <name>` - Kill specific session

### System Commands

**Reload Config:**
- `Ctrl+b` then `r` - Reload tmux configuration

**Show All Keybindings:**
- `Ctrl+b` then `?` - Display all keybindings

**Command Prompt:**
- `Ctrl+b` then `:` - Open tmux command prompt

## Mouse Features

The mouse is fully enabled for:
- **Selecting panes** - Click to switch focus
- **Resizing panes** - Drag borders
- **Selecting text** - Click and drag (auto-copies)
- **Scrolling** - Mouse wheel up/down
- **Clicking links** - If terminal supports it

## Tips & Tricks

### Quick Start
```bash
# Start new session
tmux

# Split into panes
Ctrl+b then | (horizontal split)
Ctrl+b then - (vertical split)

# Navigate with Ctrl+Arrows or click with mouse
```

### Common Workflow
1. Start tmux: `tmux`
2. Split into multiple panes for different tasks
3. Use `Ctrl+b z` to zoom into one pane when needed
4. Use mouse to select and copy output
5. Detach with `Ctrl+b d` to leave it running
6. Reattach later with `tmux a`

### Scrollback
- Mouse wheel scrolling works automatically
- Or press `Ctrl+b [` then use `Page Up`/`Page Down`
- Search in copy mode: `Ctrl+b [` then `/` to search forward, `?` to search backward

### Copy Long Output
1. Press `Ctrl+b [` to enter copy mode
2. Navigate to start of text
3. Press `v` to start selection
4. Navigate to end of text
5. Press `y` to copy to clipboard
6. Paste anywhere with `Ctrl+Shift+V`

## Configuration Location

- Config file: `~/.tmux.conf`
- Reload after changes: `tmux source-file ~/.tmux.conf` or `Ctrl+b r`

## Requirements

- **xclip** - Required for clipboard integration (already installed)
- **tmux** - Terminal multiplexer

## Troubleshooting

### Clipboard not working?
- Make sure `xclip` is installed: `which xclip`
- Install if missing: `sudo apt-get install xclip`
- Reload config: `tmux source-file ~/.tmux.conf`

### Shortcuts not working in Cursor?
- Alt-based shortcuts may conflict with IDE shortcuts
- Use Ctrl+Arrow or Ctrl+b combinations instead
- Check Cursor's keyboard settings if conflicts occur

### Colors look wrong?
- Make sure your terminal supports 256 colors
- Check `echo $TERM` - should show `screen-256color` inside tmux

### Mouse not working?
- Verify mouse mode is enabled: `tmux show -g mouse` should show `on`
- Try clicking directly in the terminal area
- Some terminal emulators may need additional configuration

## Quick Reference Card

```
PREFIX = Ctrl+b

Panes:                    Windows:                Copy Mode:
  |     split horizontal    c     new window       [     enter copy mode
  -     split vertical      1-9   switch window    v     start selection
  arrows switch panes       n     next window      y     copy selection
  z     zoom/unzoom        p     prev window      q     exit copy mode
  x     kill pane          ,     rename window

System:                   Mouse:
  r     reload config       Click   switch pane
  d     detach session      Drag    resize pane
  ?     help keybindings    Scroll  scroll/copy mode
  :     command prompt      Select  copy to clipboard
```

## Advanced Usage

### Named Windows for Organization
```bash
# Rename current window
Ctrl+b , then type name

# Create window with name
tmux new-window -n "name"
```

### Save Terminal Output
Everything in the scrollback is saved, so you can:
1. Enter copy mode: `Ctrl+b [`
2. Select all with mouse or keyboard
3. Copy with `y`
4. Paste into a file

### Run Long Tasks
```bash
# Start tmux
tmux new -s training

# Run your long task
python train.py

# Detach (task keeps running)
Ctrl+b d

# Come back later
tmux attach -s training
```

This is perfect for ML training, downloads, or any long-running process!
