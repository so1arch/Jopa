"""
Voice Changer - изменение голоса в реальном времени.
Микрофон -> смена тона (pitch shift) -> выбранное устройство вывода.
"""
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
WINDOW = 2048  # размер окна задержки для pitch shift (в сэмплах)


class PitchShifter:
    """Pitch shifter на двух плавающих линиях задержки с кроссфейдом.
    Работает с низкой задержкой, подходит для реального времени."""

    def __init__(self):
        self.hist_len = WINDOW + 4
        self.hist = np.zeros(self.hist_len, dtype=np.float32)
        self.phase = 0.0
        self.ratio = 1.0
        self.robot = False
        self._robot_pos = 0

    def set_semitones(self, semitones: float):
        self.ratio = 2.0 ** (semitones / 12.0)

    def process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        ratio = self.ratio
        buf = np.concatenate((self.hist, block))
        i = np.arange(n, dtype=np.float64)

        step = (1.0 - ratio) / WINDOW
        p1 = (self.phase + i * step) % 1.0
        p2 = (p1 + 0.5) % 1.0

        out = np.zeros(n, dtype=np.float64)
        for p in (p1, p2):
            delay = p * WINDOW + 1.0
            pos = self.hist_len + i - delay
            idx = np.floor(pos).astype(np.int64)
            frac = pos - idx
            sample = buf[idx] * (1.0 - frac) + buf[idx + 1] * frac
            gain = np.sin(np.pi * p) ** 2
            out += sample * gain

        self.phase = (self.phase + n * step) % 1.0
        self.hist = buf[-self.hist_len:].copy()

        if self.robot:
            t = (self._robot_pos + np.arange(n)) / SAMPLE_RATE
            out *= np.sin(2 * np.pi * 50.0 * t)
            self._robot_pos += n

        return np.clip(out, -1.0, 1.0).astype(np.float32)


class App(tk.Tk):
    PRESETS = {
        "Обычный": 0,
        "Мужской": -4,
        "Женский": 4,
        "Ребёнок": 8,
        "Монстр": -9,
    }

    def __init__(self):
        super().__init__()
        self.title("Voice Changer")
        self.geometry("460x360")
        self.resizable(False, False)

        self.shifter = PitchShifter()
        self.stream = None

        devices = sd.query_devices()
        self.inputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        self.outputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"] > 0]

        pad = {"padx": 12, "pady": 6}

        ttk.Label(self, text="Микрофон (вход):").pack(anchor="w", **pad)
        self.in_box = ttk.Combobox(self, state="readonly", width=60,
                                   values=[f"{i}: {n}" for i, n in self.inputs])
        self.in_box.pack(**pad)

        ttk.Label(self, text="Вывод (наушники / виртуальный кабель):").pack(anchor="w", **pad)
        self.out_box = ttk.Combobox(self, state="readonly", width=60,
                                    values=[f"{i}: {n}" for i, n in self.outputs])
        self.out_box.pack(**pad)

        try:
            default_in, default_out = sd.default.device
            self._select_default(self.in_box, self.inputs, default_in)
            self._select_default(self.out_box, self.outputs, default_out)
        except Exception:
            pass

        ttk.Label(self, text="Тон (полутона):").pack(anchor="w", **pad)
        self.pitch_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(self, from_=-12, to=12, variable=self.pitch_var,
                                command=self.on_pitch, length=420)
        self.slider.pack(**pad)
        self.pitch_label = ttk.Label(self, text="0.0")
        self.pitch_label.pack()

        presets = ttk.Frame(self)
        presets.pack(**pad)
        for name, val in self.PRESETS.items():
            ttk.Button(presets, text=name, width=9,
                       command=lambda v=val: self.set_pitch(v)).pack(side="left", padx=2)

        self.robot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Эффект робота", variable=self.robot_var,
                        command=self.on_robot).pack(**pad)

        self.btn = ttk.Button(self, text="▶ Старт", command=self.toggle)
        self.btn.pack(pady=10)

        ttk.Label(self, text="Совет: используйте наушники, чтобы не было эха.",
                  foreground="gray").pack()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    @staticmethod
    def _select_default(box, items, dev_id):
        for k, (i, _) in enumerate(items):
            if i == dev_id:
                box.current(k)
                return
        if items:
            box.current(0)

    def set_pitch(self, value):
        self.pitch_var.set(value)
        self.on_pitch(value)

    def on_pitch(self, _=None):
        v = round(self.pitch_var.get(), 1)
        self.pitch_label.config(text=f"{v:+.1f}")
        self.shifter.set_semitones(v)

    def on_robot(self):
        self.shifter.robot = self.robot_var.get()

    def callback(self, indata, outdata, frames, time, status):
        mono = indata[:, 0]
        outdata[:, 0] = self.shifter.process(mono)

    def toggle(self):
        if self.stream is None:
            if self.in_box.current() < 0 or self.out_box.current() < 0:
                messagebox.showwarning("Ошибка", "Выберите устройства ввода и вывода")
                return
            in_id = self.inputs[self.in_box.current()][0]
            out_id = self.outputs[self.out_box.current()][0]
            try:
                self.stream = sd.Stream(
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    device=(in_id, out_id),
                    channels=1,
                    dtype="float32",
                    callback=self.callback,
                )
                self.stream.start()
            except Exception as e:
                self.stream = None
                messagebox.showerror("Ошибка аудио", str(e))
                return
            self.btn.config(text="■ Стоп")
        else:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.btn.config(text="▶ Старт")

    def on_close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
