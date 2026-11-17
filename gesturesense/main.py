"""GestureSense entry point:  python main.py"""
import logging
import sys
import tkinter as tk
import traceback


def _setup_logging() -> None:
    from paths import data_dir

    try:
        logging.basicConfig(filename=str(data_dir() / "gesturesense.log"), level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")
    except OSError:
        logging.basicConfig(level=logging.INFO)


def main() -> None:
    _setup_logging()
    try:                                   # crisp text on high-DPI Windows screens
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    from gui import App
    from version import APP_NAME, __version__

    logging.info("%s %s starting (python %s)", APP_NAME, __version__, sys.version.split()[0])
    root = tk.Tk()

    def report(exc, val, tb):              # never die silently: log it and tell the user once
        logging.error("Unhandled error:\n%s", "".join(traceback.format_exception(exc, val, tb)))
        try:
            from tkinter import messagebox
            messagebox.showerror(APP_NAME, f"Something went wrong:\n{val}\n\nDetails were saved to "
                                 "gesturesense.log")
        except Exception:
            pass

    root.report_callback_exception = report
    try:
        from PIL import Image, ImageTk
        from paths import resource_dir
        icon = resource_dir() / "icon.png"
        if icon.exists():
            root._icon = ImageTk.PhotoImage(Image.open(icon))
            root.iconphoto(True, root._icon)
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
