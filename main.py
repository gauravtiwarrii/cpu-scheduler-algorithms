# main.py
# Entry point for the CPU Scheduler Simulator application.

import tkinter as tk
from ttkbootstrap import Style
from cpu_scheduler_gui import SchedulerGUI
import traceback

if __name__ == "__main__":
    try:
        root = tk.Tk()
        style = Style(theme="darkly")  # Start with dark theme
        app = SchedulerGUI(root)
        root.mainloop()
    except Exception as e:
        print("An error occurred:")
        print(traceback.format_exc())
