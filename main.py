from pathlib import Path
import queue
import subprocess
import sys
import threading
from tkinter import Tk, StringVar, filedialog, messagebox, ttk

from cardio_sketch.pipeline import (
    run_pipeline_for_record,
    run_pipeline_for_folder,
)


APP_TITLE = "CardioSketch"


def _resolve_input(path: Path) -> Path:
    """Resolve a selected ECG file to the actual input file used by the pipeline.

    XML is used directly. For WFDB, either .dat or .hea may be selected;
    the matching partner is checked automatically.
    """
    suffix = path.suffix.lower()

    if suffix == ".xml":
        return path

    if suffix in {".dat", ".hea"}:
        dat_path = path.with_suffix(".dat")
        hea_path = path.with_suffix(".hea")

        missing = []
        if not dat_path.exists():
            missing.append(dat_path.name)
        if not hea_path.exists():
            missing.append(hea_path.name)

        if missing:
            raise FileNotFoundError(
                "This WFDB record is incomplete. Missing: "
                + ", ".join(missing)
            )

        return dat_path

    raise ValueError(
        f"Unsupported file type: {path.suffix or '(no extension)'}"
    )


def _open_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.Popen(["explorer", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def main():
    root = Tk()
    root.title(APP_TITLE)
    root.geometry("620x510")
    root.resizable(False, False)

    selected_folder = {"value": None}
    selected_path = {"value": None}
    status = StringVar(value="No ECG file selected.")
    progress_text = StringVar(value="Ready.")
    progress_queue = queue.Queue()
    operation_running = {"value": False}

    frame = ttk.Frame(root, padding=26)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="CardioSketch",
        font=("Segoe UI", 22, "bold"),
    ).pack(pady=(2, 4))

    ttk.Label(
        frame,
        text="ECG waveform analysis",
        font=("Segoe UI", 11),
    ).pack(pady=(0, 18))

    ttk.Label(
        frame,
        text="Select an ECG file. XML and WFDB records are supported.\n"
             "For WFDB, you may select either the .dat or .hea file.",
        justify="center",
    ).pack(pady=(0, 16))

    operation_frame = ttk.Frame(frame)
    operation_frame.pack(pady=(0, 12))

    single_frame = ttk.LabelFrame(
        operation_frame,
        text="Single Record",
        padding=16,
    )
    single_frame.pack(side="left", padx=8)

    batch_frame = ttk.LabelFrame(
        operation_frame,
        text="Batch Analysis",
        padding=16,
    )
    batch_frame.pack(side="left", padx=8)

    def choose_file():
        path = filedialog.askopenfilename(
            parent=root,
            title="Select ECG File",
            filetypes=[
                ("ECG files", "*.xml *.dat *.hea"),
                ("XML files", "*.xml"),
                ("WFDB files", "*.dat *.hea"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            resolved = _resolve_input(Path(path))
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc), parent=root)
            return

        selected_path["value"] = resolved
        status.set(f"Selected: {resolved.name}")
        progress_text.set("Ready.")

    def choose_folder():
        folder = filedialog.askdirectory(
            parent=root,
            title="Select ECG Folder",
        )
        if not folder:
            return

        selected_folder["value"] = Path(folder)
        status.set(f"Selected folder: {Path(folder).name}")
        progress_text.set("Ready.")

    choose_button = ttk.Button(
        single_frame,
        text="Select ECG File",
        command=choose_file,
    )
    choose_button.pack(ipadx=10, ipady=6)

    run_button = ttk.Button(
        single_frame,
        text="Run CardioSketch",
    )
    run_button.pack(ipadx=10, ipady=6, pady=(9, 0))

    choose_folder_button = ttk.Button(
        batch_frame,
        text="Select ECG Folder",
        command=choose_folder,
    )
    choose_folder_button.pack(ipadx=10, ipady=6)

    run_batch_button = ttk.Button(
        batch_frame,
        text="Run Batch",
    )
    run_batch_button.pack(ipadx=10, ipady=6, pady=(9, 0))

    ttk.Label(
        frame,
        textvariable=status,
        wraplength=540,
        justify="center",
    ).pack(pady=(0, 8))

    progress_bar = ttk.Progressbar(
        frame,
        orient="horizontal",
        mode="determinate",
        maximum=100,
        length=490,
    )
    progress_bar.pack(pady=(0, 5), fill="x")

    ttk.Label(
        frame,
        textvariable=progress_text,
        wraplength=540,
        justify="center",
    ).pack(pady=(0, 12))

    output_frame = ttk.Frame(frame)
    output_frame.pack()

    ttk.Button(
        output_frame,
        text="Open ECG plots Output",
        command=lambda: _open_folder(Path("output") / "pqrst"),
    ).pack(side="left", padx=5)

    ttk.Button(
        output_frame,
        text="Open Measurement Output",
        command=lambda: _open_folder(Path("output") / "measurement"),
    ).pack(side="left", padx=5)

    ttk.Button(
        output_frame,
        text="Open Summary Output",
        command=lambda: _open_folder(Path("output") / "measurement_summary"),
    ).pack(side="left", padx=5)

    def set_controls_enabled(enabled):
        state = "normal" if enabled else "disabled"
        for button in (
            choose_button,
            run_button,
            choose_folder_button,
            run_batch_button,
        ):
            button.config(state=state)

    def start_operation(mode):
        if operation_running["value"]:
            return

        if mode == "single":
            path = selected_path["value"]
            if path is None:
                messagebox.showwarning(
                    APP_TITLE,
                    "Please select an ECG file first.",
                    parent=root,
                )
                return
            operation_description = f"Analyzing: {path.name}"
            progress_bar.stop()
            progress_bar.configure(mode="indeterminate", maximum=100)
            progress_bar.start(10)
            progress_text.set("Analyzing ECG record... Progress is being updated.")
        else:
            folder = selected_folder["value"]
            if folder is None:
                messagebox.showwarning(
                    APP_TITLE,
                    "Please select an ECG folder first.",
                    parent=root,
                )
                return
            operation_description = f"Batch analyzing: {folder.name}"
            progress_bar.stop()
            progress_bar.configure(mode="determinate", maximum=100)
            progress_bar["value"] = 0
            progress_text.set("Preparing batch analysis...")

        operation_running["value"] = True
        status.set(operation_description)
        set_controls_enabled(False)

        def worker():
            try:
                if mode == "single":
                    result = run_pipeline_for_record(path)
                else:
                    def report_progress(info):
                        # The worker thread only puts data in the queue.
                        # All Tkinter updates happen in poll_progress().
                        progress_queue.put(("progress", info))

                    result = run_pipeline_for_folder(
                        folder,
                        progress_callback=report_progress,
                    )
                progress_queue.put(("done", {
                    "mode": mode,
                    "result": result,
                    "error": None,
                }))
            except Exception as exc:
                progress_queue.put(("done", {
                    "mode": mode,
                    "result": None,
                    "error": str(exc),
                }))

        threading.Thread(target=worker, daemon=True).start()
        root.after(100, poll_progress)

    def poll_progress():
        try:
            while True:
                message_type, payload = progress_queue.get_nowait()

                if message_type == "progress":
                    completed = payload["completed"]
                    total = payload["total"]
                    percent = completed / total * 100 if total else 0

                    progress_bar["value"] = percent
                    progress_text.set(
                        f'{completed} / {total} records '
                        f'({percent:.0f}%)  |  '
                        f'Successful: {payload["success"]}  |  '
                        f'Failed: {payload["failed"]}'
                    )
                    status.set(f'Currently processed: {payload["current"]}')

                elif message_type == "done":
                    finish_operation(payload)
                    return

        except queue.Empty:
            pass

        if operation_running["value"]:
            root.after(100, poll_progress)

    def finish_operation(payload):
        mode = payload["mode"]
        result = payload["result"]
        error = payload["error"]

        progress_bar.stop()
        operation_running["value"] = False
        set_controls_enabled(True)

        if error is not None:
            status.set("Analysis failed." if mode == "single" else "Batch analysis failed.")
            progress_text.set("The operation did not complete.")
            messagebox.showerror(
                APP_TITLE,
                f"Analysis failed:\n\n{error}\n\n"
                "See the terminal for the detailed traceback.",
                parent=root,
            )
            return

        if mode == "single":
            if result is None:
                status.set("Analysis failed.")
                progress_text.set("The operation did not complete.")
                messagebox.showerror(
                    APP_TITLE,
                    "The CardioSketch pipeline did not complete successfully.\n\n"
                    "Please check the terminal for the detailed error.",
                    parent=root,
                )
                return

            progress_bar.configure(mode="determinate")
            progress_bar["value"] = 100
            status.set("Analysis completed successfully.")
            progress_text.set("1 / 1 record (100%)")
            messagebox.showinfo(
                APP_TITLE,
                "ECG analysis completed successfully.\n\n"
                "The ECG plot and measurement Excel file have been saved "
                "under the output folder.",
                parent=root,
            )
        else:
            if result is None:
                status.set("Batch analysis failed.")
                progress_text.set("The operation did not complete.")
                messagebox.showerror(
                    APP_TITLE,
                    "The batch pipeline did not complete successfully.\n\n"
                    "Please check the terminal for the detailed error.",
                    parent=root,
                )
                return

            total = result["total"]
            success = len(result["results"])
            failed = len(result["failed"])
            progress_bar.configure(mode="determinate")
            progress_bar["value"] = 100
            status.set(f"Batch completed: {success}/{total} successful.")
            progress_text.set(
                f"{total} / {total} records (100%)  |  "
                f"Successful: {success}  |  Failed: {failed}"
            )
            messagebox.showinfo(
                APP_TITLE,
                "Batch ECG analysis completed.\n\n"
                f"Total records: {total}\n"
                f"Successful: {success}\n"
                f"Failed: {failed}\n\n"
                "The output files have been saved under the output folder.",
                parent=root,
            )

    run_button.config(command=lambda: start_operation("single"))
    run_batch_button.config(command=lambda: start_operation("batch"))

    root.mainloop()


if __name__ == "__main__":
    main()
