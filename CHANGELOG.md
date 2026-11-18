# Changelog

All notable changes to GestureSense, newest first.

## 6.1.0 – one-click start
- `GestureSense.bat`: starts the app with no console window; the first run sets up the virtual
  environment and installs packages.
- `create_desktop_shortcut.bat`: puts a GestureSense icon on the Desktop.
- Effects tab → *Start the camera when the app opens*.
- Code reorganised into a package (`gesturesense/config`, `core`, `render`, `system`, `ui`); the
  3D renderer is split into focused modules.
- Hand model download is atomic (`.part` file, size check, then rename), so an interrupted
  download can no longer leave a broken model behind.
- Repo hygiene: `.gitattributes` keeps `.bat` files CRLF, runtime files are ignored, ruff
  configuration (line length 100), `CHANGELOG.md`.

## 6.0.0 – two hands, more hand styles, statistics, auto profiles
- **Two-hand mode**: the main hand keeps volume and gestures, a second hand pinches to set the
  screen brightness (`screen-brightness-control`); the main hand side is configurable.
- New hand styles: Gold, Carbon and Hologram.
- **Statistics** window: gesture counts, busiest hours, gestures per day, active time (7 or 30
  days), stored locally in `gesturesense_stats.json`.
- **Auto-switch profile by active app** (Windows): built-in rules for PowerPoint, Zoom / Teams /
  Webex / Skype / Slack and media players; your own rules go in `gesturesense_autorules.json`.

## 5.0.0 – sounds, pause hotkey and tray, tutorial
- Sound effects for every action plus a volume tick that rises in pitch (Windows `winsound`).
- Global pause hotkey **Ctrl+Alt+G** and a Pause button.
- Optional system-tray mode (`pystray`).
- Seven-step first-run tutorial that checks each gesture live.

## 4.3.1 – android palm
- Rebuilt palm: armoured shell, recessed panel, glowing core, tendon grooves, thumb pad.

## 4.3.0 – more realistic android hand
- Multi-step gloss shading, panel seams, glossy joint collars, ribbed wrist collar, soften pass.

## 4.2.0 – white android hand (new default)
- Modelled on a white robotic-arm reference: white plates, black joint collars, dark back of the
  hand, blue camera-lens knuckles.

## 4.1.0 – human hand
- Skin tones (light, medium, tan, deep), joined palm and thumb pad, knuckle creases, nails, sleeve.

## 4.0.0 – faster, animated, ready to ship
- Renderer about 2.5× faster; the camera keeps only the newest frame; speed-adaptive smoothing.
- Motion animations: fingertip streaks, sparks and floor ripples (adjustable strength).
- Webcam picker, app icon, error log, data folder (`%APPDATA%\GestureSense` when frozen,
  `GESTURESENSE_HOME` override), `build_exe.bat` for a standalone Windows app.

## 3.4 – robot hand
- Gunmetal capsule fingers with glowing LED strips, ball joints and a plated palm; *Glass* style
  kept as an option.

## 3.3 – themes and capture
- Colour themes: Aurora, Neon, Ocean, Sunset, Mono.
- Screenshot (PNG) and recording (MP4) buttons; files go to a `captures` folder.

## 3.2 – profiles and custom gestures
- Profiles (Music, Presentation, Video call, Gaming, plus your own) saved in
  `gesturesense_profiles.json`.
- Choose what Fist, Peace, Swipe left/right and Thumbs up/down do; includes an "open any app"
  action.

## 3.1 – smoother, steadier tracking
- One-Euro jitter filter, hand-lost recovery with a re-acquire delay, calibration wizard.

## 3.0 – 3D view and dark UI
- Software 3D renderer (NumPy + OpenCV): hand from MediaPipe world landmarks, volume tower, floor
  grid, orbit camera, bloom, shadows.
- Resizable window, fullscreen, DPI scaling, dark custom widgets, 3D View tab.

## 2.0 – effects and safety
- Glowing skeleton, fingertip trail, pinch line, volume arc, hold rings, action popups.
- Auto-lock after N seconds without a hand, background detection thread, volume steps capped at
  5 % per frame.

## 1.0 – first version
- Webcam hand tracking with MediaPipe; pinch for volume, fist to mute, peace sign for play/pause,
  swipe for next / previous track, pinky to lock.
