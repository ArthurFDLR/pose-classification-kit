import numpy as np
from .qt import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib import figure, patches, path

class BarGraphWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.fig = figure.Figure(figsize=(5, 5))
        self.fig.subplots_adjust(bottom=.0, top=0.975, left=.0, right=1.)
        self.canvas = FigureCanvas(self.fig)
        

        layout.addWidget(self.canvas)
        self.nbrCategories = 0
        self.offset_nullValue = 0.01
        self.ax = self.canvas.figure.subplots()
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.axis("off")
        self.changeCategories([])
        self.updateValues(np.random.rand(self.nbrCategories))

    def changeCategories(self, categories: int):
        self.clear()
        self.nbrCategories = len(categories)
        if self.nbrCategories == 0:
            bottom = 0
            top = 0
            left = 0
            right = self.offset_nullValue
            nrects = 0

        else:
            bins = np.array(
                [float(i) / self.nbrCategories for i in range(self.nbrCategories + 1)]
            )

            bottom = bins[:-1] + (0.1 / self.nbrCategories)
            top = bins[1:] - (0.1 / self.nbrCategories)
            left = np.zeros(len(top))
            right = left + self.offset_nullValue
            nrects = len(top)

        nverts = nrects * (1 + 3 + 1)
        self.verts = np.zeros((nverts, 2))
        codes = np.full(nverts, path.Path.LINETO)
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        self.verts[0::5, 0] = left
        self.verts[0::5, 1] = bottom
        self.verts[1::5, 0] = left
        self.verts[1::5, 1] = top
        self.verts[2::5, 0] = right
        self.verts[2::5, 1] = top
        self.verts[3::5, 0] = right
        self.verts[3::5, 1] = bottom

        patch = None

        barpath = path.Path(self.verts, codes)
        patch = patches.PathPatch(barpath, facecolor="#9500ff", alpha=0.5)
        self.ax.add_patch(patch)

        # Add category names
        font = {
            "family": "serif",
            "color": "#454545",
            "weight": "normal",
            "fontsize": "large",
            "fontname": "DejaVu Sans",
        }

        for i, cat in enumerate(categories):
            posy = (bottom[i] * 2 + top[i]) / 3.0
            self.ax.text(0.05, posy, cat.replace('_', ' '), fontdict=font)

        self.ax.axis("off")
        self.canvas.draw()

    def updateValues(self, values: np.ndarray):
        self.verts[2::5, 0] = values + self.offset_nullValue
        self.verts[3::5, 0] = values + self.offset_nullValue
        self.canvas.draw()

    def clear(self):
        self.ax.clear()