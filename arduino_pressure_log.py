# Full GUI with Buffered Logging + Plotting + Controls + Text Display (Updated)

import sys, os, csv
from datetime import datetime
from queue import Queue
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import serial

class SerialReader(QtCore.QThread):
    data_row = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, port: str, baud: int):
        super().__init__()
        self.port = port
        self.baud = baud
        self._stopping = False
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            QtCore.QThread.msleep(200)
            self.ser.reset_input_buffer()
        except Exception as e:
            self.error.emit(str(e))
            return

        while not self._stopping:
            try:
                raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue
                parts = [float(x) for x in raw.split(",")]
                self.data_row.emit(parts)
            except:
                continue

        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass

    def stop(self):
        self._stopping = True


class SerialLoggerPlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Buffered Serial Logger + Live Plotter")
        self.resize(1300, 700)

        self.data_queue = Queue()
        self.y_data = []
        self.curves = []
        self.max_samples = 1000
        self.series_checkboxes = []
        self.text_labels = []
        self.value_labels = []
        self.x_source_index = -1  # default: sample index

        self.csv_file = None
        self.csv_writer = None
        self.header_written = False
        self.worker = None
        self.num_series = 0

        self._build_ui()

        self.flush_timer = QtCore.QTimer()
        self.flush_timer.setInterval(50)
        self.flush_timer.timeout.connect(self._flush_data_queue)

        self.label_update_timer = QtCore.QTimer()
        self.label_update_timer.setInterval(500)  # update every 500 ms
        self.label_update_timer.timeout.connect(self._update_value_labels)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.controls = QtWidgets.QVBoxLayout()
        layout.addLayout(self.controls, 1)

        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        layout.addWidget(self.plot, 4)

        self.port_edit = QtWidgets.QLineEdit("/dev/ttyACM0")

        self.baud_combo = QtWidgets.QComboBox()
        common_bauds = [
            "300", "1200", "2400", "4800", "9600",
            "14400", "19200", "31250", "38400",
            "57600", "115200", "230400",
            "250000", "500000", "1000000", "2000000"
        ]
        self.baud_combo.addItems(common_bauds)
        self.baud_combo.setEditable(True)
        self.baud_combo.setCurrentText("1000000")

        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.setRange(10, 1000000)
        self.sample_spin.setValue(self.max_samples)
        self.sample_spin.valueChanged.connect(self._update_sample_size)

        self.csv_edit = QtWidgets.QLineEdit(f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_csv)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self.start_logging)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_logging)
        self.btn_stop.setEnabled(False)

        self.xsource_combo = QtWidgets.QComboBox()
        self.xsource_combo.addItem("Sample Index", -1)
        self.xsource_combo.currentIndexChanged.connect(self._change_x_source)

        self.controls.addWidget(QtWidgets.QLabel("Port:"))
        self.controls.addWidget(self.port_edit)
        self.controls.addWidget(QtWidgets.QLabel("Baud:"))
        self.controls.addWidget(self.baud_combo)
        self.controls.addWidget(QtWidgets.QLabel("Max Samples:"))
        self.controls.addWidget(self.sample_spin)
        self.controls.addWidget(QtWidgets.QLabel("CSV File:"))
        self.controls.addWidget(self.csv_edit)
        self.controls.addWidget(btn_browse)
        self.controls.addWidget(QtWidgets.QLabel("X-Axis Source:"))
        self.controls.addWidget(self.xsource_combo)
        self.controls.addWidget(self.btn_start)
        self.controls.addWidget(self.btn_stop)
        self.controls.addStretch()

        self.checkbox_container = QtWidgets.QVBoxLayout()
        self.controls.addLayout(self.checkbox_container)

    def _update_sample_size(self, value):
        self.max_samples = value

    def _browse_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", self.csv_edit.text())
        if path:
            self.csv_edit.setText(path)

    def start_logging(self):
        try:
            self.csv_file = open(self.csv_edit.text(), "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "CSV Error", str(e))
            return

        port = self.port_edit.text().strip()
        try:
            baud = int(self.baud_combo.currentText())
        except:
            QtWidgets.QMessageBox.critical(self, "Baud Error", "Invalid baud rate")
            return

        self.worker = SerialReader(port, baud)
        self.worker.data_row.connect(self._on_data_row)
        self.worker.error.connect(lambda msg: self._on_serial_error(msg))
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.flush_timer.start()
        self.label_update_timer.start()

    def _on_serial_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Serial Error", msg)
        self.stop_logging()

    def stop_logging(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.flush_timer.stop()
        self.label_update_timer.stop()

    def _on_data_row(self, values):
        self.data_queue.put(values)

    def _flush_data_queue(self):
        flushed = 0
        while not self.data_queue.empty():
            row = self.data_queue.get()
            flushed += 1

            if not self.header_written:
                self.num_series = len(row)
                headers = [f"S{i+1}" for i in range(self.num_series)]
                self.csv_writer.writerow(headers)
                self.y_data = [[] for _ in range(self.num_series)]
                self.curves = [self.plot.plot(pen=pg.intColor(i), name=f"S{i+1}") for i in range(self.num_series)]
                self.series_checkboxes = []
                self.value_labels = []

                for i in range(self.num_series):
                    row_widget = QtWidgets.QWidget()
                    row_layout = QtWidgets.QHBoxLayout(row_widget)
                    cb = QtWidgets.QCheckBox(f"S{i+1}")
                    cb.setChecked(True)
                    cb.stateChanged.connect(self._update_curve_visibility)
                    self.series_checkboxes.append(cb)

                    lbl = QtWidgets.QLabel("0.000")
                    self.value_labels.append(lbl)

                    row_layout.addWidget(cb)
                    row_layout.addWidget(lbl)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    self.checkbox_container.addWidget(row_widget)

                    self.xsource_combo.addItem(f"S{i+1}", i)

                self.header_written = True

            self.csv_writer.writerow(row)

            for i, v in enumerate(row):
                self.y_data[i].append(v)
                if len(self.y_data[i]) > self.max_samples:
                    self.y_data[i] = self.y_data[i][-self.max_samples:]

        if flushed:
            self._redraw_all_curves()
            self.csv_file.flush()

    def _redraw_all_curves(self):
        if self.x_source_index == -1:
            x = list(range(len(self.y_data[0]))) if self.y_data else []
        else:
            if self.x_source_index >= len(self.y_data):
                return
            x = self.y_data[self.x_source_index]

        for i, curve in enumerate(self.curves):
            if i == self.x_source_index:
                curve.clear()
                continue
            if self.series_checkboxes[i].isChecked():
                y = self.y_data[i]
                if len(x) == len(y):
                    curve.setData(x, y)
            else:
                curve.clear()

    def _change_x_source(self):
        self.x_source_index = self.xsource_combo.currentData()
        self._redraw_all_curves()

    def _update_curve_visibility(self):
        self._redraw_all_curves()

    def _update_value_labels(self):
        for i, lbl in enumerate(self.value_labels):
            if i < len(self.y_data) and self.y_data[i]:
                lbl.setText(f"{self.y_data[i][-1]:.3f}")

    def closeEvent(self, event):
        self.stop_logging()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SerialLoggerPlotter()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()