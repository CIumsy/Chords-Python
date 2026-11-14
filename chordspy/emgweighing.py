"""
EMG Weighing Scale - Estimate the weight of a held object from the EMG envelope after a quick calibration.

Calibration records the envelope with the arm supported and the hand fully relaxed (E_relaxed, the noise floor of the
electrodes), then in the lifting position with an empty hand or empty container (E_rest), then while holding 3 to 5
objects of known mass M_i (E_i). The model
    W = A * (E - E_rest) ** p
is fitted to those points with a least-squares log-log fit. E_relaxed is not part of the model: it is used to check the
signal quality (heaviest known weight against the noise floor).

Weighing an object takes two recordings: first the empty hand in position again (tare, E_tare), then the object (E).
The weight is A * (E - E_tare) ** p, so the zero point follows any drift of the baseline since calibration.

The signal pipeline reproduces Chords-Web, which the weight formula was validated on: the exact biquad filters of
src/components/filters.tsx (mains band-stop notch, then the EMG 70 Hz high-pass) run sample by sample, followed by the
RepForge EnvelopeFilter (moving average of |EMG| over 64 samples). Everything is causal, so recorded envelope values
never change retroactively.

Each hold is analysed in 1-second chunks after a settling period: the step value is the median chunk average and the
stability is the spread of the chunk averages. Every recording is saved to a CSV in ChordsPy/weighing_logs in the
user's home folder for debugging.

The side panel shows every step as a row with its own Record / Retake button. Steps are recorded in order, any recorded
step can be retaken, and cancelling a recording only stops that step. The banner at the top of the panel shows what to
do next, the countdown and progress while recording, and the weighing result.
"""
import csv
import math
from datetime import datetime
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QMainWindow, QWidget, QLabel, QComboBox, QPushButton, QLineEdit, QProgressBar, QFrame,
                             QMessageBox)
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import pylsl
import sys
import time

# Constants
WEIGHT_EXPONENT = 0.8          # Default exponent p (used when the fitted exponent is implausible)
P_MIN = 0.5                    # Smallest fitted exponent accepted
P_MAX = 1.5                    # Largest fitted exponent accepted
COUNTDOWN_SECONDS = 3          # Countdown before each recording so the user can get into position
RECORD_SECONDS = 20            # Seconds of envelope recorded for each calibration/weighing step
SETTLE_SECONDS = 8             # First seconds of each recording discarded (pickup and settling): seconds 8-20 are used
CHUNK_SECONDS = 1              # The analysed part is averaged in chunks of this length
MOVING_AVERAGE_SECONDS = 2     # Window of the live moving average drawn over the envelope
STABILITY_LIMIT = 0.15         # Maximum std/mean of the 1-second chunk averages before asking the user to repeat
REST_MOVEMENT_FACTOR = 2.0     # No-load steps: a chunk above this many times the median chunk means the arm moved
ENVELOPE_SAMPLES = 64          # Envelope window in samples (Chords-Web RepForge EnvelopeFilter)
MIN_KNOWN_RATIO = 1.5          # Every E_i must be at least this many times E_rest
MIN_KNOWN_MASS_KG = 0.5        # Smallest accepted known calibration mass
MIN_KNOWN_WEIGHTS = 3          # Fewest known weights: with only two, the fit passes through them exactly and has no error check
MAX_KNOWN_WEIGHTS = 5          # Most known weights used in one calibration
MIN_SIGNAL_TO_NOISE = 5.0      # Warn when the heaviest known weight's envelope is below this many times E_relaxed
TARE_CHANGE_LIMIT = 0.3        # Warn when the tare baseline differs from the calibration E_rest by more than this fraction
ABOVE_RANGE_FACTOR = 1.5       # Warn when a prediction exceeds this many times the largest calibrated mass
ESTIMATE_RANGE = 0.15          # The weight is shown as a range: estimate minus/plus this fraction of it
REST_SECONDS = 60              # Suggested rest between holds, counted by the rest timer in the banner
DISPLAY_DURATION = 10          # Seconds of signal shown in the plots (same as EMG with Envelope)
HIGH_PASS_CUTOFF = 70.0        # EMG high-pass cutoff in Hz (Chords-Web EMG filter)
LOG_DIR = Path.home() / "ChordsPy" / "weighing_logs"   # Debug CSVs of every calibration step and weigh (user folder: always writable)
CHORDS_STREAM_NAME = "BioAmpDataStream"             # LSL stream name published by Chords Python (connection.py)

# Filter coefficients copied from Chords-Web src/components/filters.tsx, as second-order sections:
# ((b0, b1, b2), (a1, a2)) for each section, in the order they are applied.
EMG_HIGH_PASS = {    # EXGFilter type 4 (EMG): high-pass 70 Hz, order 2
    500: [((0.52996723, -1.05993445, 0.52996723), (-0.82523238, 0.29463653))],
    250: [((0.23976966, -0.47953932, 0.23976966), (0.22115344, 0.18023207))],
}
MAINS_NOTCH = {      # Notch types 1 and 2: band-stop 48-52 Hz (50 Hz mains) and 58-62 Hz (60 Hz mains), order 2
    (500, 50): [((0.96508099, -1.56202714, 0.96508099), (-1.56858163, 0.96424138)),
                ((1.0, -1.61854514, 1.0), (-1.61100358, 0.96592171))],
    (500, 60): [((0.96508099, -1.40747202, 0.96508099), (-1.40810535, 0.96443153)),
                ((1.0, -1.45839783, 1.0), (-1.45687509, 0.96573127))],
    (250, 50): [((0.93137886, -0.57635175, 0.93137886), (-0.53127491, 0.93061518)),
                ((1.0, -0.61881558, 1.0), (-0.66243374, 0.93214913))],
    (250, 60): [((0.93137886, -0.11711144, 0.93137886), (-0.05269865, 0.93123336)),
                ((1.0, -0.12573985, 1.0), (-0.18985625, 0.93153034))],
}

# Session phases
PHASE_IDLE = "idle"            # Waiting for the user to record a step
PHASE_COUNTDOWN = "countdown"  # Countdown before recording
PHASE_RECORDING = "recording"  # Envelope is being recorded

# Step kinds
STEP_RELAXED = "relaxed"       # Calibration: arm supported, hand fully relaxed (noise floor)
STEP_REST = "rest"             # Calibration: empty hand / empty container in position (zero point)
STEP_KNOWN = "known"           # Calibration: one known weight
STEP_TARE = "tare"             # Weighing: empty hand / empty container in position, just before the object
STEP_WEIGH = "weigh"           # Weighing an unknown object
NO_LOAD_STEPS = (STEP_RELAXED, STEP_REST, STEP_TARE)   # Steps recorded without any object in the hand
HOLD_STEPS = (STEP_KNOWN, STEP_WEIGH)                   # Steps holding a weight: the rest timer is shown before them

TIPS = "Same grip · same arm posture · keep still · rest about 1 min between holds"

# Window look: one set of colours, sizes and control styles used by every element. Like the other Chords apps, the window
# keeps the standard background and the plots and side panel are flat white areas. Colour is only used for accents
# (banner header bar, active-row rail, status circles, buttons and links). No gradients anywhere.
COLOR_TEXT = "#1f2328"
COLOR_OK = "#1e8449"           # Done steps and results
COLOR_ACCENT = "#2471a3"       # Next step, recording, links and primary buttons
COLOR_ACCENT_DARK = "#1f618d"  # Primary button hover
COLOR_MUTED = "#6b7280"        # Secondary text
COLOR_WARN = "#b9770e"         # Steps that need a retake
COLOR_ERROR = "#c0392b"        # Invalid mass entries
COLOR_BORDER = "#c8ccd0"       # Buttons, entries and drop-downs
COLOR_BORDER_DARK = "#a6acb2"  # Control hover
COLOR_HOVER = "#f4f6f7"        # Secondary button hover background
COLOR_DISABLED_BG = "#f4f6f7"
COLOR_DISABLED_BORDER = "#e1e4e7"
COLOR_DISABLED_TEXT = "#b3b7bb" # Disabled controls, locked steps and their status circles
COLOR_TRACK = "#d6e4f0"        # Progress bar track
COLOR_BANNER_BORDER = "#dcdfe3"  # Outline of the banner
COLOR_DIVIDER = "#e8eaed"      # Lines between banner zones, section headers and step rows
COLOR_FOOTER = "#f8f9fa"       # Banner footer strip
COLOR_ROW_ACTIVE = "#e8f0f8"   # Background of the next / active step row (with an accent rail)
BANNER_COLORS = {              # Banner header bar (white text on it)
    "info": COLOR_ACCENT,
    "warn": "#a8660a",         # Darker than COLOR_WARN so white text stays readable
    "ok": COLOR_OK,
}
FONT_TITLE = 14                # Banner title and instructions
FONT_SMALL = 12                # Secondary text: values, counts, hints, fit line and tips
FONT_CAPS = 11                 # Upper-case labels: banner state label and section headers
FONT_BIG = 32                  # Banner countdown, progress and result
CONTROL_HEIGHT = 26            # Every button, mass entry and drop-down
BUTTON_WIDTH = 80              # Every Record / Retake / Cancel button (rows and banner)
GAP = 12                       # Space around and between the window's cards
PAD = 14                       # Inner left/right padding of the banner, section headers and step rows
ROW_HEIGHT = 38                # Every step row
RAIL_WIDTH = 3                 # Accent rail on the left of the next / active step row
STATUS_SIZE = 16               # Status circle at the start of each row
ROW_SPACING = 8                # Space between the items of a row
NAME_INDENT = PAD + STATUS_SIZE + ROW_SPACING   # Left edge of the step names: links and the fit line line up with it
BANNER_HEIGHT = 184            # Fixed banner height, so the step list below never moves
BANNER_HEADER_HEIGHT = 40
BANNER_FOOTER_HEIGHT = 44
CAPS_STYLE = f"font-size: {FONT_CAPS}px; font-weight: bold; letter-spacing: 1px;"
WINDOW_STYLE = f"""
QWidget {{ font-size: 13px; color: {COLOR_TEXT}; }}
QLabel {{ background: transparent; }}
QComboBox {{ background: white; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 0 28px 0 8px;
    min-height: {CONTROL_HEIGHT - 2}px; max-height: {CONTROL_HEIGHT - 2}px; min-width: 70px; }}
QComboBox:hover {{ border-color: {COLOR_BORDER_DARK}; }}
QComboBox:disabled {{ background: {COLOR_DISABLED_BG}; color: {COLOR_DISABLED_TEXT}; border-color: {COLOR_DISABLED_BORDER}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{ background: white; border: 1px solid {COLOR_BORDER}; outline: none;
    selection-background-color: #e8f1f8; selection-color: {COLOR_TEXT}; }}
"""
PANEL_STYLE = f"""
QFrame#panel {{ background: white; border: none; }}
QFrame#section {{ background: transparent; border: 1px solid transparent; }}
QFrame#divider {{ background: {COLOR_DIVIDER}; border: none; }}
QPushButton[variant="primary"], QPushButton[variant="secondary"] {{
    min-height: {CONTROL_HEIGHT - 2}px; max-height: {CONTROL_HEIGHT - 2}px; padding: 0 10px; border-radius: 4px; }}
QPushButton[variant="primary"] {{ background: {COLOR_ACCENT}; color: white; border: 1px solid {COLOR_ACCENT}; }}
QPushButton[variant="primary"]:hover {{ background: {COLOR_ACCENT_DARK}; border-color: {COLOR_ACCENT_DARK}; }}
QPushButton[variant="secondary"] {{ background: white; color: {COLOR_TEXT}; border: 1px solid {COLOR_BORDER}; }}
QPushButton[variant="secondary"]:hover {{ background: {COLOR_HOVER}; border-color: {COLOR_BORDER_DARK}; }}
QPushButton[variant="primary"]:disabled, QPushButton[variant="secondary"]:disabled {{
    background: {COLOR_DISABLED_BG}; color: {COLOR_DISABLED_TEXT}; border: 1px solid {COLOR_DISABLED_BORDER}; }}
QPushButton[variant="link"] {{ background: transparent; border: none; padding: 0; color: {COLOR_ACCENT}; font-size: {FONT_SMALL}px; }}
QPushButton[variant="link"]:hover {{ text-decoration: underline; }}
QPushButton[variant="link"]:disabled {{ color: {COLOR_DISABLED_TEXT}; }}
QPushButton[variant="icon"] {{ background: transparent; border: none; padding: 0; color: {COLOR_DISABLED_TEXT}; }}
QPushButton[variant="icon"]:hover {{ color: {COLOR_TEXT}; }}
QLineEdit {{ background: white; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 0 6px;
    min-height: {CONTROL_HEIGHT - 2}px; max-height: {CONTROL_HEIGHT - 2}px; }}
QLineEdit:focus {{ border-color: {COLOR_ACCENT}; }}
QLineEdit:disabled {{ background: {COLOR_DISABLED_BG}; color: {COLOR_DISABLED_TEXT}; border-color: {COLOR_DISABLED_BORDER}; }}
QLineEdit[error="true"] {{ border-color: {COLOR_ERROR}; }}
QProgressBar {{ background: {COLOR_TRACK}; border: none; border-radius: 3px; }}
QProgressBar::chunk {{ background: {COLOR_ACCENT}; border-radius: 3px; }}
"""

class ChevronComboBox(QComboBox):
    """Drop-down styled like the entries. A styled QComboBox loses the native arrow, so a chevron is drawn instead."""
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = COLOR_MUTED if self.isEnabled() else COLOR_DISABLED_TEXT
        painter.setPen(QPen(QColor(color), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        x, y = self.width() - 14, self.height() / 2
        painter.drawPolyline(QPointF(x - 4, y - 2), QPointF(x, y + 2), QPointF(x + 4, y - 2))
        painter.end()

def status_icon(kind):
    # Round status icon for a step row, drawn (not a font glyph) so every state has the same shape and size.
    # kind: done (green tick), problem (amber !), active (blue dot), next (blue ring), locked (grey ring)
    ratio = 2   # Drawn at twice the size for high-DPI screens
    pixmap = QPixmap(STATUS_SIZE * ratio, STATUS_SIZE * ratio)
    pixmap.fill(Qt.transparent)
    pixmap.setDevicePixelRatio(ratio)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    color = QColor({"done": COLOR_OK, "problem": COLOR_WARN, "active": COLOR_ACCENT, "next": COLOR_ACCENT,
                    "locked": COLOR_DISABLED_TEXT}[kind])
    circle = QRectF(1.5, 1.5, STATUS_SIZE - 3, STATUS_SIZE - 3)
    if kind in ("next", "locked"):
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(circle)
    else:
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(circle)
        white = QPen(QColor("white"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(white)
        painter.setBrush(Qt.NoBrush)
        middle = STATUS_SIZE / 2
        if kind == "done":
            tick = QPainterPath(QPointF(middle - 3.2, middle + 0.2))
            tick.lineTo(middle - 0.8, middle + 2.6)
            tick.lineTo(middle + 3.4, middle - 2.2)
            painter.drawPath(tick)
        elif kind == "problem":
            painter.drawLine(QPointF(middle, middle - 3.5), QPointF(middle, middle + 0.8))
            painter.drawPoint(QPointF(middle, middle + 3.4))
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("white"))
            painter.drawEllipse(QPointF(middle, middle), 2.6, 2.6)
    painter.end()
    return pixmap

def set_style_property(widget, name, value):
    # Set a property used by PANEL_STYLE selectors and re-apply the style when it changes
    if widget.property(name) != value:
        widget.setProperty(name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

class Biquad:
    """One second-order filter section, in the same direct form as Chords-Web filters.tsx:
    x = input - a1*z1 - a2*z2;  output = b0*x + b1*z1 + b2*z2;  z2 = z1;  z1 = x"""
    def __init__(self, b, a):
        self.b0, self.b1, self.b2 = b
        self.a1, self.a2 = a
        self.z1 = 0.0
        self.z2 = 0.0

    def process(self, value):
        x = value - self.a1 * self.z1 - self.a2 * self.z2
        output = self.b0 * x + self.b1 * self.z1 + self.b2 * self.z2
        self.z2 = self.z1
        self.z1 = x
        return output

class EnvelopeFilter:
    """Chords-Web RepForge EnvelopeFilter: moving average of |EMG| over a circular buffer (not RMS)."""
    def __init__(self, size):
        self.size = size
        self.buffer = [0.0] * size
        self.sum = 0.0
        self.index = 0

    def process(self, abs_value):
        self.sum -= self.buffer[self.index]
        self.sum += abs_value
        self.buffer[self.index] = abs_value
        self.index = (self.index + 1) % self.size
        return self.sum / self.size

class Step:
    """One recording step, shown as a row in the side panel."""
    def __init__(self, kind, title):
        self.kind = kind               # STEP_RELAXED / STEP_REST / STEP_KNOWN / STEP_TARE / STEP_WEIGH
        self.title = title             # Row label (known weights show a mass entry instead)
        self.value = None              # Median envelope of the accepted recording (None until recorded)
        self.stability = None
        # Row widgets (set by EMGWeighingScale.build_row)
        self.row = self.rail = self.icon = self.name_label = self.mass_input = None
        self.value_label = self.button = self.remove_button = None   # remove_button only on removable known weights

    def mass(self):
        # Entered mass in kg, or None when the entry is empty or not a number
        if self.mass_input is None:
            return None
        try:
            mass = float(self.mass_input.text().strip())
        except ValueError:
            return None
        return mass if math.isfinite(mass) else None

class EMGWeighingScale(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EMG Weighing Scale (Experimental)")
        self.setGeometry(100, 100, 1240, 720)

        self.stream_active = True   # Flag to check if the stream is active
        self.last_data_time = None  # Variable to store the last data time

        # Set up LSL stream inlet
        print("Searching for available LSL streams...")
        available_streams = pylsl.resolve_streams()

        if not available_streams:
            print("No LSL streams found! Exiting...")
            sys.exit(0)

        # Only streams at a supported rate can be used (Chords-Web filters exist for 250 and 500 Hz). Prefer the Chords
        # stream, then any other EXG stream, so an unrelated stream that happens to appear first is not picked.
        candidates = [s for s in available_streams if int(s.nominal_srate()) in EMG_HIGH_PASS]
        candidates.sort(key=lambda s: (s.name() != CHORDS_STREAM_NAME, s.type() != "EXG"))
        if not candidates:
            found = ", ".join(f"{s.name()} ({s.nominal_srate():g} Hz)" for s in available_streams)
            print(f"No LSL stream at a supported rate (250 or 500 Hz). Found: {found}. Exiting...")
            sys.exit(0)

        self.inlet = None
        for stream in candidates:
            try:
                self.inlet = pylsl.StreamInlet(stream)
                print(f"Connected to LSL stream: {stream.name()}")
                break
            except Exception as e:
                print(f"Failed to connect to {stream.name()}: {e}")

        if self.inlet is None:
            print("Unable to connect to any LSL stream! Exiting...")
            sys.exit(0)

        self.sampling_rate = int(self.inlet.info().nominal_srate())
        self.num_channels = self.inlet.info().channel_count()
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"Number of channels: {self.num_channels}")

        # Signal settings
        self.channel = 0       # Selected channel index (Channel 1)
        self.notch_freq = 50   # Selected mains frequency in Hz
        self.avg_window_size = MOVING_AVERAGE_SECONDS * self.sampling_rate

        # Display buffers (circular, like EMG with Envelope)
        self.buffer_size = self.sampling_rate * DISPLAY_DURATION
        self.time_data = np.linspace(0, DISPLAY_DURATION, self.buffer_size)
        self.emg_data = np.zeros(self.buffer_size)
        self.envelope_data = np.zeros(self.buffer_size)
        self.average_data = np.zeros(self.buffer_size)

        # Steps, in the order they are recorded (known weight steps are added by add_known_step)
        self.cal_steps = [Step(STEP_RELAXED, "Relaxed hand"), Step(STEP_REST, "Empty hand")]
        self.weigh_steps = [Step(STEP_TARE, "Tare (empty hand)"), Step(STEP_WEIGH, "Object")]

        # Calibration result (set by evaluate_calibration)
        self.calibrated = False
        self.cal_masses = None         # Known masses in ascending order
        self.cal_envelopes = None      # Envelope value E_i for each known mass
        self.fit_p = None
        self.fit_a = None
        self.fit_reliable = True
        self.problems = {}             # Step -> message, for recorded known weights that need a retake
        self.mass_errors = {}          # Step -> message, for invalid mass entries

        # Session state
        self.phase = PHASE_IDLE
        self.active = None             # Step being counted down or recorded
        self.countdown_end = 0.0
        self.notice = None             # (style, label, title, text, button text, action) shown in the banner until the next action
        self.last_hold_end = None      # When the last recording finished (for the rest timer)
        self.recorded = []             # Envelope chunks recorded in the current step
        self.recorded_raw = []         # Raw and filtered chunks of the same recording (for the debug CSV)
        self.recorded_filtered = []
        self.recorded_count = 0
        self.record_target = RECORD_SECONDS * self.sampling_rate
        self.banner_action = None      # Called by the banner button
        self.banner_step = None        # Step whose rest timer is shown in the banner

        self.init_ui()
        for _ in range(MIN_KNOWN_WEIGHTS):
            self.add_known_step()
        self.reset_signal_state()
        self.refresh()

        # Timer for plot update
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(15)

    # UI
    def init_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(WINDOW_STYLE)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(GAP, GAP, GAP, GAP)
        main_layout.setSpacing(GAP)

        # Top controls: channel and notch
        controls = QHBoxLayout()
        controls.setSpacing(ROW_SPACING)
        controls.addWidget(QLabel("Channel:"))
        self.channel_combo = ChevronComboBox()
        for i in range(self.num_channels):
            self.channel_combo.addItem(f"Channel {i+1}")
        self.channel_combo.setCurrentIndex(self.channel)
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        controls.addWidget(self.channel_combo)

        controls.addSpacing(20)
        controls.addWidget(QLabel("Mains notch:"))
        self.notch_combo = ChevronComboBox()
        self.notch_combo.addItems(["50 Hz", "60 Hz"])
        self.notch_combo.currentIndexChanged.connect(self.on_notch_changed)
        controls.addWidget(self.notch_combo)
        controls.addStretch()
        main_layout.addLayout(controls)

        body = QHBoxLayout()

        # Left: plots (same styling as EMG with Envelope)
        plots_layout = QVBoxLayout()
        self.emg_plot = PlotWidget(self)
        self.emg_plot.setBackground('w')
        self.emg_plot.showGrid(x=True, y=True)
        self.emg_plot.setMouseEnabled(x=False, y=False)  # Disable zoom
        self.update_emg_title()

        self.envelope_plot = PlotWidget(self)
        self.envelope_plot.setBackground('w')
        self.envelope_plot.showGrid(x=True, y=True)
        self.envelope_plot.setMouseEnabled(x=False, y=False)  # Disable zoom
        self.envelope_plot.setTitle(f"EMG Envelope (average of |EMG| over {ENVELOPE_SAMPLES} samples) with {MOVING_AVERAGE_SECONDS} s Moving Average")
        self.envelope_plot.addLegend(offset=(10, 10), labelTextColor='k', brush=pg.mkBrush(255, 255, 255, 200))

        # Set fixed axis ranges
        self.emg_plot.setXRange(0, DISPLAY_DURATION, padding=0)
        self.envelope_plot.setXRange(0, DISPLAY_DURATION, padding=0)

        # Set y-axis limits based on sampling rate (same as EMG with Envelope)
        if self.sampling_rate == 250:
            self.emg_plot.setYRange(-((2**10)/2), ((2**10)/2), padding=0)  # for R3
            self.envelope_plot.setYRange(0, ((2**10)/2), padding=0)       # for R3
        elif self.sampling_rate == 500:
            self.emg_plot.setYRange(-((2**14)/2), ((2**14)/2), padding=0)  # for R4
            self.envelope_plot.setYRange(0, ((2**14)/2), padding=0)       # for R4

        self.emg_curve = self.emg_plot.plot(self.time_data, self.emg_data, pen=pg.mkPen('b', width=1))
        self.envelope_curve = self.envelope_plot.plot(self.time_data, self.envelope_data, pen=pg.mkPen('r', width=2), name="Envelope")
        self.average_curve = self.envelope_plot.plot(self.time_data, self.average_data, pen=pg.mkPen('k', width=2), name=f"{MOVING_AVERAGE_SECONDS} s average")

        # Numeric readout of the current moving average
        self.average_label = QLabel()
        self.average_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.average_label.setStyleSheet(f"font-size: {FONT_TITLE}px; font-weight: bold;")

        plots_layout.addWidget(self.emg_plot)
        plots_layout.addWidget(self.average_label)
        plots_layout.addWidget(self.envelope_plot)
        body.setSpacing(GAP)
        body.addLayout(plots_layout, stretch=3)

        # Right: banner, calibration steps and weighing steps
        self.status_icons = {kind: status_icon(kind) for kind in ("done", "problem", "active", "next", "locked")}
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(440)
        panel.setStyleSheet(PANEL_STYLE)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(GAP, GAP, GAP, GAP)
        panel_layout.setSpacing(GAP)
        panel_layout.addWidget(self.build_banner())

        # Calibrate: header, step rows, then the Add weight link, fit line and quality warnings
        self.cal_count_label = QLabel()
        self.cal_count_label.setStyleSheet(f"color: {COLOR_MUTED}; font-size: {FONT_SMALL}px;")
        self.reset_button = self.link_button("Reset")
        self.reset_button.clicked.connect(self.reset_calibration)
        calibrate, calibrate_layout = self.section("Calibrate", self.cal_count_label, self.reset_button)
        self.cal_rows_layout = QVBoxLayout()
        self.cal_rows_layout.setSpacing(0)
        calibrate_layout.addLayout(self.cal_rows_layout)
        for step in self.cal_steps:
            self.cal_rows_layout.addWidget(self.build_row(step))
        calibrate_layout.addWidget(self.divider())

        footer = QVBoxLayout()    # Starts at the same left edge as the step names
        footer.setContentsMargins(NAME_INDENT, 8, PAD, 0)
        footer.setSpacing(4)
        self.add_weight_button = self.link_button("+ Add weight")
        self.add_weight_button.clicked.connect(lambda: (self.add_known_step(), self.refresh()))
        footer.addWidget(self.add_weight_button, alignment=Qt.AlignLeft)
        self.fit_label = QLabel()   # Fit error and signal to noise (per-weight table in the tooltip)
        self.fit_label.setStyleSheet(f"color: {COLOR_MUTED}; font-size: {FONT_SMALL}px;")
        footer.addWidget(self.fit_label)
        self.quality_label = QLabel()   # Signal quality warnings
        self.quality_label.setWordWrap(True)
        self.quality_label.setStyleSheet(f"color: {COLOR_WARN}; font-size: {FONT_SMALL}px;")
        footer.addWidget(self.quality_label)
        calibrate_layout.addLayout(footer)
        panel_layout.addWidget(calibrate)

        # Weigh: header and the tare and object rows
        self.weigh_another_button = self.link_button("Weigh another")
        self.weigh_another_button.clicked.connect(self.weigh_another)
        weigh, weigh_layout = self.section("Weigh", None, self.weigh_another_button)
        for step in self.weigh_steps:
            weigh_layout.addWidget(self.build_row(step))
        panel_layout.addWidget(weigh)

        panel_layout.addStretch()
        tips = QLabel(TIPS)
        tips.setWordWrap(True)
        tips.setStyleSheet(f"color: {COLOR_MUTED}; font-size: {FONT_SMALL}px;")
        tips.setContentsMargins(PAD + 1, 0, PAD + 1, 0)   # +1: sections have a 1 px (transparent) border like the banner
        panel_layout.addWidget(tips)

        body.addWidget(panel, stretch=2)
        main_layout.addLayout(body)
        self.setCentralWidget(central_widget)

    @staticmethod
    def link_button(text):
        button = QPushButton(text)
        button.setProperty("variant", "link")
        button.setCursor(Qt.PointingHandCursor)
        return button

    @staticmethod
    def action_button():
        # Record / Retake / Cancel button: the same size everywhere (rows and banner)
        button = QPushButton()
        button.setProperty("variant", "secondary")
        button.setFixedWidth(BUTTON_WIDTH)
        button.setCursor(Qt.PointingHandCursor)
        return button

    @staticmethod
    def divider():
        line = QFrame()
        line.setObjectName("divider")
        line.setFixedHeight(1)
        return line

    @classmethod
    def section(cls, title, info_label, button):
        # Section of the panel: upper-case header (with optional count and link) over a divider. Returns the frame and
        # the layout the rows go into. Header text, row content and links share the PAD left/right edges.
        frame = QFrame()
        frame.setObjectName("section")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(PAD, 10, PAD, 10)
        header.setSpacing(12)
        label = QLabel(title.upper())
        label.setStyleSheet(f"{CAPS_STYLE} color: {COLOR_MUTED};")
        header.addWidget(label)
        header.addStretch()
        if info_label is not None:
            header.addWidget(info_label)
        header.addWidget(button)
        layout.addLayout(header)
        layout.addWidget(cls.divider())
        return frame, layout

    def build_banner(self):
        # Fixed-height box at the top of the panel, in three zones split by dividers: header bar (step name and state
        # label), body (instruction, countdown, progress or result) and footer (hint and button)
        self.banner = QFrame()
        self.banner.setObjectName("banner")
        self.banner.setFixedHeight(BANNER_HEIGHT)
        layout = QVBoxLayout(self.banner)
        layout.setContentsMargins(1, 1, 1, 1)   # Inside the 1 px border
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("bannerHeader")
        header.setFixedHeight(BANNER_HEADER_HEIGHT)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(PAD, 0, PAD, 0)
        header_layout.setSpacing(10)
        self.banner_title = QLabel()
        self.banner_title.setStyleSheet(f"font-size: {FONT_TITLE}px; font-weight: bold; color: white;")
        self.banner_label = QLabel()
        self.banner_label.setStyleSheet(f"{CAPS_STYLE} color: rgba(255, 255, 255, 0.78);")
        header_layout.addWidget(self.banner_title)
        header_layout.addStretch()
        header_layout.addWidget(self.banner_label)
        layout.addWidget(header)

        middle = QVBoxLayout()
        middle.setContentsMargins(PAD, 8, PAD, 8)
        middle.setSpacing(6)
        self.banner_big = QLabel()
        self.banner_big.setAlignment(Qt.AlignCenter)
        self.banner_body = QLabel()
        self.banner_body.setWordWrap(True)
        self.banner_progress = QProgressBar()
        self.banner_progress.setRange(0, self.record_target)
        self.banner_progress.setTextVisible(False)
        self.banner_progress.setFixedHeight(6)
        middle.addStretch()
        middle.addWidget(self.banner_big)
        middle.addWidget(self.banner_body)
        middle.addWidget(self.banner_progress)
        middle.addStretch()
        layout.addLayout(middle, stretch=1)

        self.banner_footer_line = self.divider()
        layout.addWidget(self.banner_footer_line)
        self.banner_footer = QFrame()
        self.banner_footer.setObjectName("bannerFooter")
        self.banner_footer.setFixedHeight(BANNER_FOOTER_HEIGHT)
        footer_layout = QHBoxLayout(self.banner_footer)
        footer_layout.setContentsMargins(PAD, 0, PAD, 0)
        footer_layout.setSpacing(10)
        self.banner_hint = QLabel()
        self.banner_hint.setWordWrap(True)
        self.banner_button = self.action_button()
        self.banner_button.clicked.connect(lambda: self.banner_action and self.banner_action())
        footer_layout.addWidget(self.banner_hint, stretch=1)
        footer_layout.addWidget(self.banner_button)
        layout.addWidget(self.banner_footer)
        return self.banner

    def build_row(self, step, removable=False):
        # [rail] status  name (or mass entry, "kg" and a remove ✕)  ...  value  button
        row = QFrame()
        row.setObjectName("stepRow")
        row.setFixedHeight(ROW_HEIGHT)
        outer = QHBoxLayout(row)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        step.rail = QFrame()     # Accent rail, shown on the next / active step
        step.rail.setFixedWidth(RAIL_WIDTH)
        outer.addWidget(step.rail)
        layout = QHBoxLayout()
        layout.setContentsMargins(PAD - RAIL_WIDTH, 0, PAD, 0)
        layout.setSpacing(ROW_SPACING)
        outer.addLayout(layout)

        step.icon = QLabel()
        step.icon.setFixedSize(STATUS_SIZE, STATUS_SIZE)
        layout.addWidget(step.icon)
        if step.kind == STEP_KNOWN:
            step.mass_input = QLineEdit()
            step.mass_input.setFixedWidth(64)
            step.mass_input.setAlignment(Qt.AlignRight)
            step.mass_input.setPlaceholderText("mass")
            step.mass_input.textChanged.connect(self.refresh)
            layout.addWidget(step.mass_input)
            layout.addWidget(QLabel("kg"))
            if removable:    # Known weights beyond the required ones can be removed
                step.remove_button = QPushButton("✕")
                step.remove_button.setProperty("variant", "icon")
                step.remove_button.setFixedSize(STATUS_SIZE, CONTROL_HEIGHT)
                step.remove_button.setCursor(Qt.PointingHandCursor)
                step.remove_button.setToolTip("Remove this weight")
                step.remove_button.clicked.connect(lambda: self.remove_known_step(step))
                layout.addWidget(step.remove_button)
        else:
            step.name_label = QLabel(step.title)
            layout.addWidget(step.name_label)
        layout.addStretch()

        step.value_label = QLabel()
        layout.addWidget(step.value_label)
        step.button = self.action_button()
        step.button.clicked.connect(lambda: self.on_row_button(step))
        layout.addWidget(step.button)

        step.row = row
        return row

    def add_known_step(self):
        known = self.known_steps()
        if len(known) >= MAX_KNOWN_WEIGHTS:
            return
        step = Step(STEP_KNOWN, f"Known weight {len(known) + 1}")
        self.cal_steps.append(step)
        self.cal_rows_layout.addWidget(self.build_row(step, removable=len(known) >= MIN_KNOWN_WEIGHTS))

    def remove_known_step(self, step):
        if self.phase != PHASE_IDLE or step not in self.cal_steps:
            return
        self.cal_steps.remove(step)
        self.cal_rows_layout.removeWidget(step.row)
        step.row.setParent(None)     # Detach now: a removed row must not stay drawn until it is deleted
        step.row.deleteLater()
        self.refresh()

    def known_steps(self):
        return [step for step in self.cal_steps if step.kind == STEP_KNOWN]

    def update_emg_title(self):
        self.emg_plot.setTitle(f"Filtered EMG Signal (High Pass:{int(HIGH_PASS_CUTOFF)} Hz, Notch:{self.notch_freq} Hz)")

    # Step texts
    def step_name(self, step):
        if step.kind == STEP_KNOWN:
            mass = step.mass()
            return f"Known weight {mass:.2f} kg" if mass is not None else f"Known weight {self.known_steps().index(step) + 1}"
        return {STEP_RELAXED: "Relaxed hand", STEP_REST: "Empty hand", STEP_TARE: "Tare", STEP_WEIGH: "Object"}[step.kind]

    def step_position(self, step):
        # Banner state label for a step that is next: its place in the calibration or weighing sequence
        if step in self.cal_steps:
            return f"Step {self.cal_steps.index(step) + 1} of {len(self.cal_steps)}"
        return f"Weigh {self.weigh_steps.index(step) + 1} of {len(self.weigh_steps)}"

    def step_instruction(self, step):
        if step.kind == STEP_RELAXED:
            return "Rest your forearm on a table or your lap, hand open and fully relaxed."
        if step.kind == STEP_REST:
            return "Get into position with an empty hand (or the empty container)."
        if step.kind == STEP_KNOWN:
            return f"Pick up the {step.mass():.2f} kg weight and hold it in position."
        if step.kind == STEP_TARE:
            return "Get into position with an empty hand (or the same empty container)."
        return "Pick up the object and hold it in position."

    def step_hint(self, step):
        if step.kind == STEP_RELAXED:
            return "Measures the electrode noise floor"
        if step.kind == STEP_REST:
            return "Using a container? Results will be the contents only"
        if step.kind == STEP_TARE:
            return "Re-zeros the scale before weighing"
        return self.rest_text()

    def rest_text(self):
        # Rest timer since the last recording (shown before steps that hold a weight)
        if self.last_hold_end is None:
            return ""
        rested = int(time.time() - self.last_hold_end)
        target = f"{REST_SECONDS // 60}:{REST_SECONDS % 60:02d}"
        if rested >= REST_SECONDS:
            return f"Rested {target} ✓"
        return f"Rested {rested // 60}:{rested % 60:02d} of {target}"

    # Calibration state
    def evaluate_calibration(self):
        # Validate the mass entries and recorded known weights, and fit the model once every calibration step is
        # recorded and valid. Sets mass_errors, problems, calibrated and the fit.
        known = self.known_steps()
        self.mass_errors = {}
        masses = {}
        for step in known:
            text = step.mass_input.text().strip()
            mass = step.mass()
            masses[step] = None
            if not text:
                continue          # Not filled in yet: blocks the step without flagging it
            if mass is None:
                self.mass_errors[step] = "Enter a number in kg"
            elif mass < MIN_KNOWN_MASS_KG:
                self.mass_errors[step] = f"At least {MIN_KNOWN_MASS_KG} kg"
            else:
                masses[step] = mass
        by_mass = {}
        for step, mass in masses.items():
            if mass is not None:
                by_mass.setdefault(round(mass, 3), []).append(step)
        for group in by_mass.values():
            if len(group) > 1:
                for step in group:
                    self.mass_errors[step] = "Known weights must all be different"
                    masses[step] = None

        # Recorded known weights must be clearly above the empty hand and increase with mass
        self.problems = {}
        e_rest = self.cal_steps[1].value
        if e_rest is not None:
            for step in known:
                if step.value is not None and step.value < MIN_KNOWN_RATIO * e_rest:
                    self.problems[step] = (f"The signal did not rise enough (E = {step.value:.1f}, needs at least "
                                           f"{MIN_KNOWN_RATIO:g}× the empty hand: {MIN_KNOWN_RATIO * e_rest:.1f}). "
                                           "Use a heavier weight or check the electrodes.")
        ordered = sorted((s for s in known if s.value is not None and masses[s] is not None and s not in self.problems),
                         key=lambda s: masses[s])
        for lighter, heavier in zip(ordered, ordered[1:]):
            if heavier.value <= lighter.value:
                self.problems[heavier] = (f"The signal did not rise above the {masses[lighter]:.2f} kg weight. "
                                          "Rest your arm, check your grip and posture, then retake.")

        self.calibrated = (all(step.value is not None for step in self.cal_steps) and not self.problems
                           and all(masses[step] is not None for step in known))
        if self.calibrated:
            ordered = sorted(known, key=lambda s: masses[s])
            self.cal_masses = np.array([masses[s] for s in ordered])
            self.cal_envelopes = np.array([s.value for s in ordered])
            self.fit_p, self.fit_a, self.fit_reliable = self.fit_model(self.cal_masses, self.cal_envelopes - self.e_rest)

    @property
    def e_relaxed(self):
        return self.cal_steps[0].value

    @property
    def e_rest(self):
        return self.cal_steps[1].value

    def next_step(self):
        # The first step not recorded yet (steps are recorded in order; weighing needs a complete calibration)
        for step in self.cal_steps:
            if step.value is None:
                return step
        if not self.calibrated:
            return None
        for step in self.weigh_steps:
            if step.value is None:
                return step
        return None

    def quality_warnings(self):
        # Signal quality checks against the relaxed noise floor (warnings only: the model does not use E_relaxed)
        warnings = []
        if self.e_rest <= self.e_relaxed:
            warnings.append("The empty hand was not above the relaxed hand: make sure your hand is fully relaxed and "
                            "supported in step 1.")
        snr = self.cal_envelopes.max() / self.e_relaxed if self.e_relaxed > 0 else float('inf')
        if snr < MIN_SIGNAL_TO_NOISE:
            warnings.append(f"Weak signal ({snr:.1f}× the noise floor, at least {MIN_SIGNAL_TO_NOISE:g}× "
                            "recommended): check the electrodes or use heavier weights.")
        if not self.fit_reliable:
            warnings.append(f"Unreliable fit (exponent outside {P_MIN} to {P_MAX}): p = {WEIGHT_EXPONENT} was used.")
        return snr, warnings

    def fit_table_html(self):
        # Tooltip of the fit line: each known weight with its envelope, predicted mass and error
        rows = ""
        errors = []
        for mass, envelope in zip(self.cal_masses, self.cal_envelopes):
            predicted = self.predict_weight(envelope)
            error = (predicted - mass) / mass * 100
            errors.append(abs(error))
            rows += (f"<tr><td align='right'>{mass:.2f}</td><td align='right'>{envelope:.1f}</td>"
                     f"<td align='right'>{predicted:.2f}</td><td align='right'>{error:+.1f}%</td></tr>")
        table = ("<table cellspacing='0' cellpadding='3'>"
                 "<tr><th>Mass (kg)</th><th>Envelope</th><th>Predicted (kg)</th><th>Error</th></tr>"
                 f"{rows}</table><p>E_relaxed = {self.e_relaxed:.1f}, E_rest = {self.e_rest:.1f}<br>"
                 f"p = {self.fit_p:.3f}, A = {self.fit_a:.4g}</p>")
        return float(np.mean(errors)), table

    # Refreshing the panel
    def refresh(self):
        # Rebuild every row and the banner from the current state
        self.evaluate_calibration()
        busy = self.phase != PHASE_IDLE
        upcoming = None if busy else self.next_step()
        for step in self.cal_steps + self.weigh_steps:
            self.update_row(step, busy, upcoming)

        done = sum(step.value is not None for step in self.cal_steps)
        self.cal_count_label.setText(f"{done} of {len(self.cal_steps)} ✓" if self.calibrated else f"{done} of {len(self.cal_steps)} done")
        if self.calibrated:
            average_error, table = self.fit_table_html()
            snr, warnings = self.quality_warnings()
            self.fit_label.setText(f"Fit error {average_error:.1f}%  ·  Signal {snr:.1f}×")
            self.fit_label.setToolTip(table)
            self.quality_label.setText("<br>".join(warnings))
        self.fit_label.setVisible(self.calibrated)
        self.quality_label.setVisible(self.calibrated and bool(self.quality_label.text()))

        self.reset_button.setEnabled(not busy)
        self.add_weight_button.setEnabled(not busy and len(self.known_steps()) < MAX_KNOWN_WEIGHTS)
        self.weigh_another_button.setVisible(not busy and self.weigh_steps[1].value is not None)
        self.channel_combo.setEnabled(not busy)
        self.notch_combo.setEnabled(not busy)
        self.update_banner()

    def update_row(self, step, busy, upcoming):
        locked = step in self.weigh_steps and not self.calibrated
        highlight = False
        name_color = COLOR_TEXT
        variant = "secondary"
        if step is self.active:
            icon = "active"
            value, value_color = ("get ready…" if self.phase == PHASE_COUNTDOWN else "recording…"), COLOR_ACCENT
            button, enabled, highlight = "Cancel", True, True
        elif step.value is not None:
            problem = step in self.problems
            icon = "problem" if problem else "done"
            value, value_color = f"E = {step.value:.1f}", (COLOR_WARN if problem else COLOR_MUTED)
            button, enabled = "Retake", not busy and not locked
        elif step is upcoming:
            icon = "next"
            value, value_color = "", COLOR_MUTED
            ready = step.kind != STEP_KNOWN or (step.mass() is not None and step not in self.mass_errors)
            button, enabled, highlight, variant = "Record", ready, True, "primary"
        else:
            icon = "locked"
            value, value_color = "", COLOR_MUTED
            button, enabled, name_color = "Record", False, COLOR_DISABLED_TEXT

        step.icon.setPixmap(self.status_icons[icon])
        step.value_label.setText(value)
        step.value_label.setStyleSheet(f"color: {value_color}; font-size: {FONT_SMALL}px;")
        step.button.setText(button)
        step.button.setEnabled(enabled)
        set_style_property(step.button, "variant", variant)
        # Next / active step: light background and an accent rail. Rows are split by dividers (none under the last row).
        group = self.cal_steps if step in self.cal_steps else self.weigh_steps
        divider = "none" if step is group[-1] else f"1px solid {COLOR_DIVIDER}"
        step.row.setStyleSheet(f"QFrame#stepRow {{ background: {COLOR_ROW_ACTIVE if highlight else 'transparent'}; "
                               f"border: none; border-bottom: {divider}; }}")
        step.rail.setStyleSheet(f"background: {COLOR_ACCENT if highlight else 'transparent'}; border: none;")
        if step.name_label is not None:
            step.name_label.setStyleSheet(f"color: {name_color};" + (" font-weight: bold;" if highlight else ""))
        if step.mass_input is not None:
            error = self.mass_errors.get(step)
            step.mass_input.setEnabled(not busy)
            set_style_property(step.mass_input, "error", bool(error))
            step.mass_input.setToolTip(error or f"Mass in kg (at least {MIN_KNOWN_MASS_KG} kg)")
        if step.remove_button is not None:
            step.remove_button.setEnabled(not busy)

    def set_banner(self, style, label, title, big="", body="", hint="", button=None, action=None, primary=True,
                   enabled=True, progress=False, big_color=COLOR_TEXT, secondary_body=False, hint_color=COLOR_MUTED):
        # Every state uses the same pieces. style: info (blue), warn (amber) or ok (green) header bar; label: upper-case
        # state on the right of the bar; title: step name on the left. Body: optional big text (with the body centred
        # under it), body text and progress bar. Footer (hidden when empty): hint and one action button.
        # secondary_body shows the body as small muted text (the result's estimate line).
        color = BANNER_COLORS[style]
        self.banner.setStyleSheet(f"""
            QFrame#banner {{ background: white; border: 1px solid {COLOR_BANNER_BORDER}; border-radius: 4px; }}
            QFrame#bannerHeader {{ background: {color}; border: none; border-top-left-radius: 3px; border-top-right-radius: 3px; }}
            QFrame#bannerFooter {{ background: {COLOR_FOOTER}; border: none; border-bottom-left-radius: 3px;
                                   border-bottom-right-radius: 3px; }}""")
        self.banner_title.setText(title)
        self.banner_label.setText(label.upper())
        self.banner_big.setText(big)
        self.banner_big.setStyleSheet(f"font-size: {FONT_BIG}px; font-weight: bold; color: {big_color};")
        self.banner_big.setVisible(bool(big))
        self.banner_body.setText(body)
        body_style = (f"font-size: {FONT_SMALL}px; color: {COLOR_MUTED};" if secondary_body
                      else f"font-size: {FONT_TITLE}px; color: {COLOR_TEXT};")
        self.banner_body.setStyleSheet(body_style)
        self.banner_body.setAlignment(Qt.AlignCenter if big else Qt.AlignLeft | Qt.AlignVCenter)
        self.banner_body.setVisible(bool(body))
        self.banner_progress.setVisible(progress)
        self.banner_hint.setText(hint)
        self.banner_hint.setStyleSheet(f"color: {hint_color}; font-size: {FONT_SMALL}px;")
        self.banner_button.setVisible(button is not None)
        self.banner_button.setText(button or "")
        self.banner_button.setEnabled(enabled)
        set_style_property(self.banner_button, "variant", "primary" if primary else "secondary")
        footer = bool(hint) or button is not None
        self.banner_footer.setVisible(footer)
        self.banner_footer_line.setVisible(footer)
        self.banner_action = action
        self.banner_step = None

    def update_banner(self):
        step = self.active
        if self.phase == PHASE_COUNTDOWN:
            self.set_banner("info", "Get ready", self.step_name(step), str(COUNTDOWN_SECONDS), self.step_instruction(step),
                            button="Cancel", action=self.cancel_step, primary=False)
            return
        if self.phase == PHASE_RECORDING:
            self.set_banner("info", "Recording", self.step_name(step), f"0 / {RECORD_SECONDS} s",
                            hint="Keep your hand and arm still", button="Cancel", action=self.cancel_step,
                            primary=False, progress=True)
            self.banner_progress.setValue(0)
            return
        if self.notice:
            style, label, title, text, button, action = self.notice
            self.set_banner(style, label, title, body=text, button=button, action=action)
            return

        # A recorded known weight that needs a retake
        problem = next((s for s in self.cal_steps if s in self.problems), None)
        if problem:
            self.set_banner("warn", "Retake needed", self.step_name(problem), body=self.problems[problem],
                            button="Retake", action=lambda: self.start_step(problem))
            return

        upcoming = self.next_step()
        if upcoming is not None and upcoming.kind == STEP_KNOWN and (upcoming.mass() is None or upcoming in self.mass_errors):
            error = self.mass_errors.get(upcoming)
            self.set_banner("info", self.step_position(upcoming), self.step_name(upcoming),
                            body=f"Enter the mass of this known weight in kg (at least {MIN_KNOWN_MASS_KG} kg).",
                            hint=error or "", hint_color=COLOR_ERROR, button="Record", enabled=False)
            return
        if upcoming is not None:
            self.set_banner("info", self.step_position(upcoming), self.step_name(upcoming), body=self.step_instruction(upcoming),
                            hint=self.step_hint(upcoming), button="Record", action=lambda: self.start_step(upcoming))
            if upcoming.kind in HOLD_STEPS:
                self.banner_step = upcoming     # Keep the rest timer ticking
            return

        if not self.calibrated:
            # Everything is recorded but a mass entry is missing or invalid
            bad = next((s for s in self.known_steps() if s.mass() is None or s in self.mass_errors), None)
            message = self.mass_errors.get(bad, "Enter the mass of every known weight.") if bad else ""
            self.set_banner("warn", "Calibrate", "Check the known weights", body=message)
            return
        self.show_result()

    def show_result(self):
        tare, obj = self.weigh_steps[0].value, self.weigh_steps[1].value
        weight = self.predict_weight(obj, tare)
        notes = []
        if obj <= tare:
            notes.append("No load detected.")
        elif weight > ABOVE_RANGE_FACTOR * self.cal_masses.max():
            notes.append("Above the calibrated range: less reliable.")
        change = tare / self.e_rest - 1
        if abs(change) > TARE_CHANGE_LIMIT:
            notes.append(f"Empty hand changed {change * 100:+.0f}% since calibration: recalibrate if posture or electrodes changed.")
        if weight > 0:
            low, high = weight * (1 - ESTIMATE_RANGE), weight * (1 + ESTIMATE_RANGE)
            big, body = f"{low:.2f} – {high:.2f} kg", f"Estimate {weight:.2f} kg (±{ESTIMATE_RANGE * 100:.0f}%)"
        else:
            big, body = "0.00 kg", ""
        self.set_banner("ok", "Result", "Estimated weight", big, body, hint=" ".join(notes), big_color=COLOR_OK,
                        secondary_body=True, hint_color=COLOR_WARN)

    # Signal processing
    def build_filters(self):
        # Fresh Chords-Web filters (zero state, as in Chords-Web): mains notch, EMG high-pass and envelope
        self.notch_filter = [Biquad(b, a) for b, a in MAINS_NOTCH[(self.sampling_rate, self.notch_freq)]]
        self.high_pass_filter = [Biquad(b, a) for b, a in EMG_HIGH_PASS[self.sampling_rate]]
        self.envelope_filter = EnvelopeFilter(ENVELOPE_SAMPLES)

    def reset_signal_state(self):
        self.build_filters()
        self.avg_history = np.zeros(0)                       # Last window-1 envelope samples for the moving average
        self.emg_data[:] = 0
        self.envelope_data[:] = 0
        self.average_data[:] = 0
        self.current_index = 0
        self.average_label.setText(f"{MOVING_AVERAGE_SECONDS} s average: --")

    def process_chunk(self, raw):
        # Chords-Web pipeline, sample by sample with stateful filters: notch, then EMG high-pass, then envelope of |x|
        filtered = np.empty(len(raw))
        envelope = np.empty(len(raw))
        for i, value in enumerate(raw):
            for section in self.notch_filter:
                value = section.process(value)
            for section in self.high_pass_filter:
                value = section.process(value)
            filtered[i] = value
            envelope[i] = self.envelope_filter.process(abs(value))

        # Causal moving average of the envelope (uses fewer samples until the window has filled)
        history = np.concatenate([self.avg_history, envelope])
        cumsum = np.concatenate([[0.0], np.cumsum(history)])
        ends = np.arange(len(self.avg_history) + 1, len(history) + 1)
        starts = np.maximum(ends - self.avg_window_size, 0)
        moving_avg = (cumsum[ends] - cumsum[starts]) / (ends - starts)
        self.avg_history = history[-(self.avg_window_size - 1):]

        return filtered, envelope, moving_avg

    def update_plot(self):
        if self.inlet is None or not self.stream_active:
            return
        samples, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=30)
        if samples:
            self.last_data_time = time.time()
            raw = np.array([sample[self.channel] for sample in samples], dtype=float)
            filtered, envelope, moving_avg = self.process_chunk(raw)

            # Overwrite the oldest data points in the circular buffers
            indices = (self.current_index + np.arange(len(raw))) % self.buffer_size
            self.emg_data[indices] = filtered
            self.envelope_data[indices] = envelope
            self.average_data[indices] = moving_avg
            self.current_index = (self.current_index + len(raw)) % self.buffer_size

            if self.phase == PHASE_RECORDING:
                self.recorded.append(envelope)
                self.recorded_raw.append(raw)
                self.recorded_filtered.append(filtered)
                self.recorded_count += len(envelope)

            # Update curves and readout
            self.emg_curve.setData(self.time_data, self.emg_data)
            self.envelope_curve.setData(self.time_data, self.envelope_data)
            self.average_curve.setData(self.time_data, self.average_data)
            self.average_label.setText(f"{MOVING_AVERAGE_SECONDS} s average: {moving_avg[-1]:.1f}")
        else:
            if self.last_data_time and (time.time() - self.last_data_time) > 2:
                self.stream_active = False
                print("LSL stream disconnected!")
                self.timer.stop()
                self.close()
                return

        self.update_session()

    # Weight model
    @staticmethod
    def fit_model(masses, deltas):
        # Fit W = A * delta ** p, where delta = E - E_rest (at least MIN_KNOWN_WEIGHTS points). Returns (p, A, reliable).
        p, log_a = np.polyfit(np.log(deltas), np.log(masses), 1)   # ln(W) = p * ln(delta) + ln(A)
        if P_MIN <= p <= P_MAX:
            return float(p), float(np.exp(log_a)), True
        # Fitted exponent is implausible: keep the default exponent and refit only A (geometric mean)
        a = np.exp(np.mean(np.log(masses / deltas ** WEIGHT_EXPONENT)))
        return WEIGHT_EXPONENT, float(a), False

    def predict_weight(self, envelope_value, baseline=None):
        # baseline: the no-load level to measure from (the tare when weighing, E_rest by default)
        baseline = self.e_rest if baseline is None else baseline
        if envelope_value <= baseline:
            return 0.0
        return self.fit_a * (envelope_value - baseline) ** self.fit_p

    # Session flow
    def on_row_button(self, step):
        if step is self.active:
            self.cancel_step()
        else:
            self.start_step(step)

    def start_step(self, step):
        # Record (or retake) one step: countdown, then recording
        if self.phase != PHASE_IDLE:
            return
        self.notice = None
        self.active = step
        self.phase = PHASE_COUNTDOWN
        self.countdown_end = time.time() + COUNTDOWN_SECONDS
        self.refresh()

    def begin_recording(self):
        self.phase = PHASE_RECORDING
        self.recorded = []
        self.recorded_raw = []
        self.recorded_filtered = []
        self.recorded_count = 0
        self.refresh()

    def cancel_step(self):
        # Stop only the step in progress: every recorded step keeps its value
        if self.phase == PHASE_IDLE:
            return
        self.phase = PHASE_IDLE
        self.active = None
        self.refresh()

    def update_session(self):
        if self.phase == PHASE_COUNTDOWN:
            remaining = self.countdown_end - time.time()
            if remaining > 0:
                self.banner_big.setText(str(math.ceil(remaining)))
            else:
                self.begin_recording()
        elif self.phase == PHASE_RECORDING:
            if self.recorded_count >= self.record_target:
                self.finish_recording()
                return
            self.banner_big.setText(f"{int(self.recorded_count / self.sampling_rate)} / {RECORD_SECONDS} s")
            self.banner_progress.setValue(self.recorded_count)
        elif self.banner_step is not None:
            hint = self.rest_text()
            if self.banner_hint.text() != hint:
                self.banner_hint.setText(hint)

    def analyse_segment(self, envelope):
        # Use seconds SETTLE_SECONDS..RECORD_SECONDS, averaged in 1-second chunks. Returns (value, stability, peak):
        # value = median chunk average, stability = std / mean of the chunk averages,
        # peak = highest chunk average relative to the median (a large peak means the arm moved)
        segment = envelope[SETTLE_SECONDS * self.sampling_rate:]
        size = int(CHUNK_SECONDS * self.sampling_rate)
        chunks = segment[:(len(segment) // size) * size].reshape(-1, size).mean(axis=1)
        value = float(np.median(chunks))
        mean = float(np.mean(chunks))
        stability = float(np.std(chunks) / mean) if mean > 0 else float('inf')
        peak = float(np.max(chunks) / value) if value > 0 else float('inf')
        return value, stability, peak

    def save_log(self, step, raw, filtered, envelope):
        # Debug CSV of this recording (the whole hold, including the settling period). Returns (path, error).
        if step.kind in NO_LOAD_STEPS:
            mass = 0.0                                          # Relaxed hand, or empty hand / empty container
        elif step.kind == STEP_KNOWN:
            mass = step.mass()
        else:
            mass = None                                         # Unknown object: left blank
        stem = datetime.now().strftime('%Y%m%d-%H%M%S') + f"_{step.kind}" + (f"_{mass:.2f}kg" if step.kind == STEP_KNOWN else "")
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            path = LOG_DIR / f"{stem}.csv"
            count = 2
            while path.exists():                                # Never overwrite an earlier log
                path = LOG_DIR / f"{stem}_{count}.csv"
                count += 1
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time_s', 'raw', 'filtered', 'envelope', 'step', 'mass_kg'])
                mass_text = '' if mass is None else f"{mass:g}"
                for i in range(len(envelope)):
                    writer.writerow([f"{i / self.sampling_rate:.4f}", raw[i], filtered[i], envelope[i], step.kind, mass_text])
            return path, None
        except OSError as e:
            return None, str(e)

    def finish_recording(self):
        step = self.active
        n = self.record_target
        envelope = np.concatenate(self.recorded)[:n]
        log_path, log_error = self.save_log(step, np.concatenate(self.recorded_raw)[:n], np.concatenate(self.recorded_filtered)[:n], envelope)
        if log_error:
            print(f"Could not save the log: {log_error}")
        value, stability, peak = self.analyse_segment(envelope)
        self.phase = PHASE_IDLE
        self.active = None
        self.last_hold_end = time.time()

        # No-load levels are small, so their relative spread means little: only a clear movement fails them
        if step.kind in NO_LOAD_STEPS:
            unstable = peak > REST_MOVEMENT_FACTOR
            reason = f"one second was {peak:.1f}× the typical level, so the arm moved"
        else:
            unstable = stability > STABILITY_LIMIT
            reason = f"stability {stability:.2f}, limit {STABILITY_LIMIT}"
        if unstable:
            kept = " The previous recording was kept." if step.value is not None else ""
            self.notice = ("warn", "Signal unstable", self.step_name(step),
                           f"The signal varied too much ({reason}). Rest your arm, keep still, then retake.{kept}",
                           "Retake", lambda: self.start_step(step))
        else:
            step.value = value
            step.stability = stability
        self.refresh()

    def weigh_another(self):
        # Clear the tare and object so the next object is weighed from a fresh tare
        for step in self.weigh_steps:
            step.value = None
        self.notice = None
        self.refresh()

    def clear_recordings(self):
        for step in self.cal_steps + self.weigh_steps:
            step.value = None
            step.stability = None
        self.notice = None
        self.last_hold_end = None

    def reset_calibration(self):
        answer = QMessageBox.question(self, "Reset calibration", "Clear all calibration and weighing recordings?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.clear_recordings()
            self.refresh()

    def on_channel_changed(self, index):
        self.channel = index
        self.reset_signal_state()
        self.clear_recordings()
        self.notice = ("info", "Settings changed", f"Channel {index + 1} selected", "Filters and calibration were reset.", "OK", self.dismiss_notice)
        self.refresh()

    def on_notch_changed(self, index):
        # Calibration values depend on the exact filter chain, so they are invalidated as well
        self.notch_freq = 50 if index == 0 else 60
        self.update_emg_title()
        self.reset_signal_state()    # Rebuilds the filters with the new notch
        self.clear_recordings()
        self.notice = ("info", "Settings changed", f"Notch set to {self.notch_freq} Hz", "Filters and calibration were reset.", "OK", self.dismiss_notice)
        self.refresh()

    def dismiss_notice(self):
        self.notice = None
        self.refresh()

def main():
    app = pg.mkQApp()    # Enables high-DPI scaling, so plots scale correctly when moved to another monitor
    window = EMGWeighingScale()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
