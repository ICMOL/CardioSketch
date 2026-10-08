from pathlib import Path
import subprocess
import sys
from tkinter import Tk, StringVar, filedialog, messagebox, ttk

from cardio_sketch.pipeline import (
    run_pipeline_for_record,
    run_pipeline_for_folder,
)


APP_TITLE = "CardioSketch"


def _resolve_input(path: Path) -> Path:
    """Resolve a user-selected ECG file to the actual input file used by the pipeline.

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

        # wfdb.rdrecord() receives the record base path; the pipeline's
        # loader will handle the .dat/.hea pair.
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
    root.geometry("560x430")
    root.resizable(False, False)
    selected_folder = {"value": None}
    selected_path = {"value": None}
    status = StringVar(value="No ECG file selected.")

    frame = ttk.Frame(root, padding=28)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="CardioSketch",
        font=("Segoe UI", 22, "bold"),
    ).pack(pady=(4, 4))

    ttk.Label(
        frame,
        text="ECG waveform analysis",
        font=("Segoe UI", 11),
    ).pack(pady=(0, 24))

    ttk.Label(
        frame,
        text="Select an ECG file. XML and WFDB records are supported.\n"
             "For WFDB, you may select either the .dat or .hea file.",
        justify="center",
    ).pack(pady=(0, 18))

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
    def choose_folder():
        folder = filedialog.askdirectory(
            parent=root,
            title="Select ECG Folder",
        )
        if not folder:
            return

        selected_folder["value"] = Path(folder)
        status.set(f"Selected folder: {Path(folder).name}")
    def run_analysis():
        path = selected_path["value"]
        if path is None:
            messagebox.showwarning(
                APP_TITLE,
                "Please select an ECG file first.",
                parent=root,
            )
            return

        run_button.config(state="disabled")
        choose_button.config(state="disabled")
        status.set(f"Analyzing: {path.name}")
        root.update_idletasks()

        try:
            result = run_pipeline_for_record(path)
            if result is None:
                raise RuntimeError(
                    "The CardioSketch pipeline did not complete successfully. "
                    "Please check the terminal for the detailed error."
                )

            status.set("Analysis completed successfully.")
            messagebox.showinfo(
                APP_TITLE,
                "ECG analysis completed successfully.\n\n"
                "The PQRST plot and measurement Excel file have been saved "
                "under the output folder.",
                parent=root,
            )
        except Exception as exc:
            status.set("Analysis failed.")
            messagebox.showerror(
                APP_TITLE,
                f"Analysis failed:\n\n{exc}\n\n"
                "See the terminal for the detailed traceback.",
                parent=root,
            )
        finally:
            run_button.config(state="normal")
            choose_button.config(state="normal")
    def run_batch():
        folder = selected_folder["value"]

        if folder is None:
            messagebox.showwarning(
                APP_TITLE,
                "Please select an ECG folder first.",
                parent=root,
            )
            return

        run_batch_button.config(state="disabled")
        choose_folder_button.config(state="disabled")
        choose_button.config(state="disabled")
        run_button.config(state="disabled")

        status.set(f"Batch analyzing: {folder.name}")
        root.update_idletasks()

        try:
            result = run_pipeline_for_folder(folder)

            if result is None:
                raise RuntimeError(
                    "The batch pipeline did not complete successfully."
                )

            total = result["total"]
            success = len(result["results"])
            failed = len(result["failed"])

            status.set(
                f"Batch completed: {success}/{total} successful."
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

        except Exception as exc:
            status.set("Batch analysis failed.")

            messagebox.showerror(
                APP_TITLE,
                f"Batch analysis failed:\n\n{exc}\n\n"
                "See the terminal for the detailed traceback.",
                parent=root,
            )

        finally:
            run_batch_button.config(state="normal")
            choose_folder_button.config(state="normal")
            choose_button.config(state="normal")
            run_button.config(state="normal")

    operation_frame = ttk.Frame(frame)
    operation_frame.pack(pady=(0, 14))

    single_frame = ttk.LabelFrame(
        operation_frame,
        text="Single Record",
        padding=18,
    )
    single_frame.pack(side="left", padx=8)

    batch_frame = ttk.LabelFrame(
        operation_frame,
        text="Batch Analysis",
        padding=18,
    )
    batch_frame.pack(side="left", padx=8)

    choose_button = ttk.Button(
        single_frame,
        text="Select ECG File",
        command=choose_file,
    )
    choose_button.pack(ipadx=12, ipady=7)

    run_button = ttk.Button(
        single_frame,
        text="Run CardioSketch",
        command=run_analysis,
    )
    run_button.pack(ipadx=12, ipady=7, pady=(10, 0))

    choose_folder_button = ttk.Button(
        batch_frame,
        text="Select ECG Folder",
        command=choose_folder,
    )
    choose_folder_button.pack(ipadx=12, ipady=7)

    run_batch_button = ttk.Button(
        batch_frame,
        text="Run Batch",
        command=run_batch,
    )
    run_batch_button.pack(ipadx=12, ipady=7, pady=(10, 0))

    ttk.Label(
        frame,
        textvariable=status,
        wraplength=500,
        justify="center",
    ).pack(pady=(2, 14))

    output_frame = ttk.Frame(frame)
    output_frame.pack()

    ttk.Button(
        output_frame,
        text="Open PQRST Output",
        command=lambda: _open_folder(Path("output") / "pqrst"),
    ).pack(side="left", padx=5)

    ttk.Button(
        output_frame,
        text="Open Measurement Output",
        command=lambda: _open_folder(Path("output") / "measurement"),
    ).pack(side="left", padx=5)

    root.mainloop()


if __name__ == "__main__":
    main()
