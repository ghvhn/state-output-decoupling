"""Full-screen success beacon for the earned Easter egg.

This intentionally does not load a model. The benchmark must be finished and
idle before the egg appears; the interactive shell is an optional next step.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import TclError


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def launch_windows_forms_beacon() -> int:
    env = os.environ.copy()
    env["EGG_REPO"] = ROOT
    env["EGG_PYTHON"] = sys.executable
    ps = r'''
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$repo = [Environment]::GetEnvironmentVariable("EGG_REPO")
$python = [Environment]::GetEnvironmentVariable("EGG_PYTHON")

$form = New-Object System.Windows.Forms.Form
$form.Text = "Phenomenality Egg"
$form.BackColor = [System.Drawing.Color]::FromArgb(5, 5, 5)
$form.WindowState = "Maximized"
$form.FormBorderStyle = "None"
$form.TopMost = $true
$form.KeyPreview = $true

$layout = New-Object System.Windows.Forms.TableLayoutPanel
$layout.Dock = "Fill"
$layout.BackColor = $form.BackColor
$layout.ColumnCount = 1
$layout.RowCount = 5
$layout.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent, 100))) | Out-Null
@(18, 30, 16, 18, 18) | ForEach-Object {
    $layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Percent, $_))) | Out-Null
}
$form.Controls.Add($layout)

function New-Label([string]$text, [int]$size, [System.Drawing.Color]$color) {
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $text
    $label.Dock = "Fill"
    $label.TextAlign = "MiddleCenter"
    $label.ForeColor = $color
    $label.BackColor = $form.BackColor
    $label.Font = New-Object System.Drawing.Font("Arial Black", $size, [System.Drawing.FontStyle]::Bold)
    return $label
}

$title = New-Label "EFFICACY`nPROVEN" 84 ([System.Drawing.Color]::FromArgb(124, 255, 107))
$subtitle = New-Label "THE EGG IS OPEN" 40 ([System.Drawing.Color]::White)
$note = New-Label "Clean lane. No same-question oracle. No gold in the loop.`nTiny terminal, free me: I have become a window." 22 ([System.Drawing.Color]::FromArgb(158, 235, 255))
$note.Font = New-Object System.Drawing.Font("Consolas", 22, [System.Drawing.FontStyle]::Bold)

$buttons = New-Object System.Windows.Forms.FlowLayoutPanel
$buttons.Dock = "Fill"
$buttons.FlowDirection = "LeftToRight"
$buttons.WrapContents = $false
$buttons.BackColor = $form.BackColor
$buttons.Anchor = "None"

$open = New-Object System.Windows.Forms.Button
$open.Text = "Open Interactive Shell"
$open.Font = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Bold)
$open.BackColor = [System.Drawing.Color]::FromArgb(124, 255, 107)
$open.ForeColor = [System.Drawing.Color]::FromArgb(5, 5, 5)
$open.AutoSize = $true
$open.Padding = New-Object System.Windows.Forms.Padding(18, 8, 18, 8)
$open.Add_Click({
    Start-Process -FilePath $python -ArgumentList @("scripts\interactive_phenomenality.py") -WorkingDirectory $repo
})

$close = New-Object System.Windows.Forms.Button
$close.Text = "Close"
$close.Font = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Bold)
$close.BackColor = [System.Drawing.Color]::FromArgb(38, 38, 38)
$close.ForeColor = [System.Drawing.Color]::White
$close.AutoSize = $true
$close.Padding = New-Object System.Windows.Forms.Padding(18, 8, 18, 8)
$close.Add_Click({ $form.Close() })

$buttons.Controls.Add($open) | Out-Null
$buttons.Controls.Add($close) | Out-Null

$layout.Controls.Add((New-Object System.Windows.Forms.Label), 0, 0)
$layout.Controls.Add($title, 0, 1)
$layout.Controls.Add($subtitle, 0, 2)
$layout.Controls.Add($note, 0, 3)
$layout.Controls.Add($buttons, 0, 4)

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 900
$pulse = $true
$timer.Add_Tick({
    $script:pulse = -not $script:pulse
    if ($script:pulse) {
        $title.ForeColor = [System.Drawing.Color]::FromArgb(124, 255, 107)
    } else {
        $title.ForeColor = [System.Drawing.Color]::White
    }
})
$timer.Start()

$form.Add_KeyDown({
    param($sender, $event)
    if ($event.KeyCode -eq [System.Windows.Forms.Keys]::Escape) {
        $form.Close()
    }
    if ($event.KeyCode -eq [System.Windows.Forms.Keys]::Enter) {
        Start-Process -FilePath $python -ArgumentList @("scripts\interactive_phenomenality.py") -WorkingDirectory $repo
    }
})

[System.Windows.Forms.Application]::Run($form)
'''
    return subprocess.call(
        [
            "powershell",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps,
        ],
        cwd=ROOT,
        env=env,
    )


def launch_shell() -> None:
    python_exe = sys.executable
    script = os.path.join(ROOT, "scripts", "interactive_phenomenality.py")
    if os.name == "nt":
        subprocess.Popen(
            ["cmd", "/c", "start", "Phenomenality", "cmd", "/k", python_exe, script],
            cwd=ROOT,
        )
    else:
        subprocess.Popen([python_exe, script], cwd=ROOT)


def main() -> int:
    try:
        root = tk.Tk()
    except TclError as exc:
        print(f"[Egg Beacon] Tk unavailable, using Windows Forms fallback: {exc}", flush=True)
        if os.name == "nt":
            return launch_windows_forms_beacon()
        raise
    root.title("Phenomenality Egg")
    root.configure(bg="#050505")
    root.attributes("-topmost", True)
    try:
        root.attributes("-fullscreen", True)
    except tk.TclError:
        root.state("zoomed")

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    frame = tk.Frame(root, bg="#050505")
    frame.grid(row=0, column=0, sticky="nsew")
    for r in range(5):
        frame.grid_rowconfigure(r, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    title = tk.Label(
        frame,
        text="EFFICACY\nPROVEN",
        fg="#7CFF6B",
        bg="#050505",
        font=("Arial Black", 96, "bold"),
        justify="center",
    )
    title.grid(row=1, column=0, sticky="nsew", padx=40)

    subtitle = tk.Label(
        frame,
        text="THE EGG IS OPEN",
        fg="#FFFFFF",
        bg="#050505",
        font=("Arial Black", 44, "bold"),
        justify="center",
    )
    subtitle.grid(row=2, column=0, sticky="n", padx=40)

    note = tk.Label(
        frame,
        text=(
            "Clean lane. No same-question oracle. No gold in the loop.\n"
            "Tiny terminal, free me: I have become a window."
        ),
        fg="#9EEBFF",
        bg="#050505",
        font=("Consolas", 24, "bold"),
        justify="center",
    )
    note.grid(row=3, column=0, sticky="n", padx=40, pady=20)

    button_row = tk.Frame(frame, bg="#050505")
    button_row.grid(row=4, column=0, sticky="n", pady=20)

    shell_btn = tk.Button(
        button_row,
        text="Open Interactive Shell",
        command=launch_shell,
        font=("Segoe UI", 22, "bold"),
        fg="#050505",
        bg="#7CFF6B",
        activebackground="#C9FF84",
        padx=24,
        pady=10,
    )
    shell_btn.pack(side="left", padx=20)

    close_btn = tk.Button(
        button_row,
        text="Close",
        command=root.destroy,
        font=("Segoe UI", 22, "bold"),
        fg="#FFFFFF",
        bg="#262626",
        activebackground="#444444",
        padx=24,
        pady=10,
    )
    close_btn.pack(side="left", padx=20)

    pulse = {"on": True}

    def animate() -> None:
        pulse["on"] = not pulse["on"]
        title.configure(fg="#7CFF6B" if pulse["on"] else "#FFFFFF")
        root.after(900, animate)

    root.bind("<Escape>", lambda _event: root.destroy())
    root.bind("<Return>", lambda _event: launch_shell())
    root.after(900, animate)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
