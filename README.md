# GestureSense (v3 - 3D view and dark UI)

Control your Windows volume and media with hand gestures through your webcam.
Python - OpenCV - MediaPipe Tasks (HandLandmarker) - NumPy - Pycaw - Tkinter.
No new packages in v3: the 3D renderer is written with NumPy + OpenCV only.

## Run
1. Install Python 3.10 - 3.12 from https://python.org (tick "Add python.exe to PATH").
2. Double-click `run.bat` (or in PowerShell: `.\run.bat`).
   First run: creates `.venv`, installs packages, downloads the hand model (internet needed).
3. Click **Start**.

Manual: `python -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, `pip install -r requirements.txt`, `python main.py`.

## Gestures
| Gesture | Action |
|---|---|
| Pinch / spread thumb + index | Volume down / up |
| Raise pinky | Lock volume (lower it to unlock) |
| Fist, hold ~0.7 s | Toggle mute |
| Peace sign, hold ~0.7 s | Play / pause |
| Open palm, swipe right / left | Next / previous track |

## What is new in v3.4 - robot hand
- The 3D hand is now a **robot hand**: gunmetal capsule fingers with a glowing LED strip, ball joints, glowing fingertips, a plated palm with a glowing core and a forearm cuff. Its accent colour follows the state (green active, red locked, violet muted) and its metal is tinted by the theme
- *3D View > Hand style* switches back to the earlier **glass hand** if you prefer it

## What is new in v3.3 - themes and capture
- **Colour themes** for the 3D scene (*3D View > Colour theme*): Aurora, Neon, Ocean, Sunset and Mono change the tower gradient, hand skin, backdrop and floor grid
- **Screenshot** button: saves what you see in the view as a PNG
- **Record** button: records the view to an MP4 (steady 25 fps, shows a live timer; press again to stop - closing the window or pressing Stop also finishes the file)
- Files are saved in a `captures` folder next to the program; the status line shows the file name
- New files: `capture.py`, `tests/test_capture_themes.py` (73 tests in total)

## What is new in v3.2 - profiles and custom gestures
- **Profiles** (Music, Presentation, Video call, Gaming + your own): pick one in the top bar; each profile remembers what every gesture does. Saved in `gesturesense_profiles.json`
- **Gesture actions** (*Gestures* tab): choose what Fist, Peace sign, Swipe right / left and the new **Thumbs up / Thumbs down** do - mute, volume +/- 10 %, play/pause, next / previous track, slide next / previous, start / end slideshow, Teams / Zoom / Meet mic mute, screenshot, Alt+Tab, show desktop, or **open any app**
- Hold thumbs up / down to step the volume; keep holding and it keeps stepping
- *New profile* copies the current one, *Reset* restores a built-in profile, *Delete* removes your own profiles (built-ins are protected)
- Pinch / spread volume and the pinky lock work in every profile
- New files: `actions.py`, `profiles.py`, `tests/test_profiles.py` (62 tests in total)
- Notes: thumbs gestures assume an upright hand; Teams / Zoom / Meet shortcuts are the apps' default keys and need the app to be in focus or have global shortcuts on

## What is new in v3.1 - smoother, steadier tracking
- **Jitter filter** (One-Euro filter): the hand stops shaking when you hold still, yet follows fast movements without lag. *Tuning > Jitter filter* (0 = off, higher = smoother; 0.5 is a good start)
- **Hand-lost recovery**: if detection blinks out for a moment (default 0.3 s) the last pose is held; when the hand really disappears and comes back, the volume stays frozen briefly (*Re-acquire delay*) and then eases in, so it can never jump
- **Calibration wizard** (*Tuning > Calibration wizard...*): hold a pinch, then a spread; it measures your real range, rejects shaky or too-similar results, and applies the pinched / spread ratios. Press *Save settings* to keep them
- New files: `filters.py`, `calibration.py`, `wizard.py`, `tests/test_tracking.py` (40 tests in total)

## What is new in v3
**3D scene** (`render3d.py`, pure NumPy + OpenCV)
- A 3D hand built from MediaPipe's real *world landmarks* (metres, with true finger depth): shaded bones, lit joints, glowing fingertips, depth-sorted
- A segmented 3D volume tower on a perspective floor grid; its colour runs azure -> violet -> coral and it turns violet when muted / red when locked
- A glowing tether from your pinch to the tower height, a ground shadow + floor marker under your hand, and floor ripples on every action
- An orbiting camera: it sweeps by itself, or **drag** to orbit, **scroll** to zoom, **double-click** to go back to auto-orbit
- Three views, switchable at the top of the viewport: *3D + Camera* (camera inset), *3D Scene*, *Camera* (the v2 look)
- Try it without a webcam: `python demo3d.py` (animated synthetic hand; keys: q quit, space pause, a auto-orbit, m mute)

**Fully adjustable window**
- The 3D / camera view fills the whole window and the 3D scene renders at the view's size
- Drag the divider between the view and the settings panel to resize them, press **Hide panel** for a big view, **F11** for fullscreen (Esc to leave)
- Everything scales with Windows display scaling (125 %, 150 %...)
- *3D View* tab: auto-orbit, speed, sweep, camera height, zoom, **camera inset size** (or scroll the mouse wheel while hovering over the inset), **render quality** (lower = faster) and the grid / shadow / glow switches

**Modern dark UI** (`ui.py`, `gui.py`)
- Rebuilt window: top bar with live status, volume card with a segmented LED meter, MUTED / LOCKED chips, and tabbed settings (Gestures, Tuning, Effects, 3D View)
- Custom toggles, sliders and buttons; all colours live in one dict (`ui.C`) if you want to re-skin it
- Orbit speed / sweep / height / zoom and grid / shadow / glow switches are in the *3D View* tab and are saved with your settings

## What was new in v2
**Visual effects** (each can be switched off in the *Effects* tab)
- Glowing hand skeleton that changes colour: green = active, red = locked, violet = muted
- Fingertip trail that fades out while you pinch
- Pinch line that turns from blue to red as volume rises, with a live percentage
- Animated circular volume arc (bottom-left of the preview)
- Hold-progress ring around your hand while you hold a fist or peace sign
- Action popups (muted, play/pause, next, previous, locked, unlocked) that fade in and out

**Features**
- Auto-lock: after N seconds without a hand the volume locks itself. To unlock, show your hand,
  raise your pinky, then lower it. Set N in *Tuning* (0 = off).
- Detection runs in a background thread, so the window stays responsive.
- Safety: volume changes at most 5 % per frame, so a sudden gesture can never blast your speakers.
- Tuning tab with live values, calibration buttons, save and reset.

## Performance
The 3D scene takes ~18 ms per frame on a typical CPU. If your frame rate drops, switch the view to *Camera*, turn off *Bloom / glow* in the *3D View* tab, or lower `VIEW_W` / `VIEW_H` in `render3d.py`.

## Tips
- Good front lighting, hand upright and facing the camera, 30 - 60 cm away.
- Click **Set pinched** while pinching and **Set spread** while spreading, then **Save settings**.
- Wrong camera? Edit `camera_index` in `gesturesense_settings.json` (0, 1, 2...), saved after you click Save.

## Troubleshooting
- *Scripts disabled when activating .venv*: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
- *MediaPipe will not install*: use Python 3.10 - 3.12 (`py -3.12 -m venv .venv`).
- *Could not open webcam*: close Zoom / Teams / Camera app, or change `camera_index`.
- *Model download fails*: download `hand_landmarker.task` (URL is in `gui.py`) and put it next to `main.py`.
- Pycaw is Windows-only. macOS / Linux would need a different `audio.py`.

## Project layout
| File | Purpose |
|---|---|
| `main.py` | Entry point |
| `gui.py` | Dark Tkinter window, viewport, tabs, mouse orbit |
| `ui.py` | Dark widget kit: buttons, toggles, sliders, tabs, LED meter |
| `capture.py` | Screenshot (PNG) and recording (MP4) |
| `actions.py` | Action catalogue and runner (keys, hotkeys, volume steps, open app) |
| `profiles.py` | Profiles: gesture -> action mappings, load / save |
| `filters.py` | One-Euro jitter filter for the landmarks |
| `calibration.py` | Calibration maths (sample collection, range, quality check) |
| `wizard.py` | Calibration wizard window |
| `render3d.py` | Software 3D renderer: camera, hand model, volume tower, floor grid |
| `demo3d.py` | Webcam-free demo with a synthetic animated hand |
| `worker.py` | Background thread: camera -> MediaPipe -> gestures -> effects |
| `gestures.py` | Pose detection, hold timers, swipe detection, volume logic (no GUI / audio imports) |
| `effects.py` | OpenCV overlay effects and animations |
| `audio.py` | Windows volume / mute through Pycaw |
| `settings.py` | Defaults, load, save |
| `tests/` | Simulated-hand tests (logic + 3D renderer): `python -m unittest discover -s tests -v` |

## Roadmap
Stage 2: custom gesture mapping, profiles (Music / Presentation / Video call), brightness control.
Stage 3: system tray mode, usage statistics, light theme option, settings import/export,
start with Windows and a global pause hotkey.


## v4.0 - faster, animated, ready to ship

**Speed.** The 3D renderer is ~2.5x faster (about 9 ms per frame at the default size, 12 ms at full
window size). The camera keeps only the newest frame (no buffered lag), the window refreshes every
12 ms, and the hand follows with speed-adaptive smoothing: steady when still, tight when moving fast.

**Motion animations** (3D View tab -> *Motion animations*, strength slider): glowing streaks follow
your fingertips, sparks fly off them, and fast sweeps send a ripple across the floor. All scale with
how quickly the hand moves; switch them off any time.

**Ready to ship.**
- A webcam picker (3D View tab), an app icon, error logging to `gesturesense.log`.
- Files live next to the program when run from source, or in `%APPDATA%\GestureSense` when built as
  an .exe. Set `GESTURESENSE_HOME` to choose another folder.
- Build a standalone Windows app: double-click `build_exe.bat`, then share `dist\GestureSense`.
  Put `hand_landmarker.task` next to it first if you want the model inside the build (otherwise it is
  downloaded on first Start). Not tested on Windows in the build environment - run it once yourself.

### Human hand (v4.1)
3D View tab -> *Hand style* -> **Human hand**, then pick a *Skin tone* (light, medium, tan, deep).
Shaded skin, a joined palm and thumb pad, knuckle creases, nails and a sleeve cuff in the status
colour. It is drawn in software, so it looks like a clean 3D-animated hand, not a photograph.

### White android hand (v4.2, new default)
Modelled on a white robotic-arm reference: glossy white plates, black joint collars, a dark
mechanical back of the hand and blue camera-lens knuckles, with a white forearm and black wrist
collar. Pick it under 3D View -> *Hand style* -> **White android hand** (Robot, Human and Glass
are still there).

### v4.3
More realistic white android hand: smoother multi-step gloss shading with a cool rim and a specular
streak, panel seams and grooves on each finger segment, glossy black joint collars, a ribbed wrist
collar, and a soften pass so the shapes blend into one surface.

### v4.3.1
Rebuilt android palm: white armoured shell with a recessed dark panel, glowing blue core, tendon grooves, black knuckle gap and a thumb pad.

## v5.0 - sounds, pause hotkey + tray, first-run tutorial

**Sound effects.** Short tones for mute / unmute, play / pause, next / previous, lock / unlock, volume
steps and pause / resume. While you pinch, a soft tick plays each time the volume crosses a 10 % step,
and its pitch rises with the volume. Turn them off, or set the level, in the Effects tab. (Windows
plays them through the built-in `winsound`; other systems stay silent.)

**Pause from anywhere.** Press **Ctrl+Alt+G** (or the Pause button) to stop all gestures without
closing anything - tracking and the 3D view keep running, but nothing changes your volume or presses
keys. Press it again to resume.

**System tray.** Effects tab -> *Keep running in the tray when closed*. Closing the window then hides
it; use the tray icon to Show, Pause / resume or Quit. Needs the `pystray` package (installed by
`run.bat`).

**Tutorial.** A 7-step guided tour (show hand, pinch, fist, peace, swipe, thumbs, pinky lock) that
checks each gesture live from your camera. It opens on first run and any time from the **Tutorial**
button. While it is open, gestures are paused, so practising is safe.

## v6.0 - two hands, more hands, statistics, auto profiles

**Two-hand mode** (Tuning tab -> *Two hands*). Your main hand (right by default) keeps doing volume
and every gesture. A second hand pinches to set the **screen brightness**; it appears in 3D with a glowing
sun that grows and brightens with the level. Pick which side is the main hand, or set the second hand
to "show only". Brightness uses the `screen-brightness-control` package (installed by `run.bat`) and works
on laptop panels and most monitors with DDC/CI; if your display does not support it the sun still shows
the level but the screen will not change. The camera is restarted internally when you switch the mode.

**More hands** (3D View -> Hand style): Gold, Carbon and Hologram join White android, Robot, Human
and Glass.

**Statistics** (Stats button). Counts of each gesture, the busiest hours, gestures per day and active
time, for 7 or 30 days. Stored only on your computer in `gesturesense_stats.json`; Reset deletes it.

**Auto-switch profile** (Gestures tab -> *Auto-switch profile by active app*, Windows). PowerPoint
switches to Presentation, Zoom / Teams / Webex / Skype / Slack to Video call, Spotify / VLC /
media players to Music; leaving the app puts your previous profile back (unless you changed it
yourself). Add your own rules in `gesturesense_autorules.json` next to the program:

    {"obs64.exe": "Gaming", "chrome.exe": "Music"}

Not tested on a real Windows PC in the build environment: the brightness control, the foreground-app
lookup, the global hotkey, sounds and the tray icon need a quick try on your machine.

### One-click start (v6.1)
- **Double-click `GestureSense.bat`**: starts the app with no console window. The very first time it
  sets everything up (a few minutes), then opens GestureSense.
- **Double-click `create_desktop_shortcut.bat`** once to put a GestureSense icon on your Desktop; after
  that, click the icon.
- Effects tab -> *Start the camera when the app opens* makes it start working immediately, with no
  need to press Start.
