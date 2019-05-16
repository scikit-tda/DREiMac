import sys
import os
import pyqtgraph as pg
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

class MplQTCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        self.fig = fig
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class PersistenceSelector(MplQTCanvas):
    def __init__(self, parent, emobj, width=5, height=4, dpi=100, dgm_size = 20):
        MplQTCanvas.__init__(self, parent, width, height, dpi)
        self.emobj = emobj
        dgm = emobj.dgms_[1]
        ax_min, ax_max = np.min(dgm), np.max(dgm)
        x_r = ax_max - ax_min
        buffer = x_r / 5
        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer
        y_down, y_up = x_down, x_up
        yr = y_up - y_down
        self.axes.plot([x_down, x_up], [x_down, x_up], "--", c=np.array([0.0, 0.0, 0.0]))
        self.dgmplot, = self.axes.plot(dgm[:, 0], dgm[:, 1], 'o', picker=5, c='C0')
        self.selected_plot = self.axes.scatter([], [], 100, c='C1')
        self.axes.set_xlim([x_down, x_up])
        self.axes.set_ylim([y_down, y_up])
        self.axes.set_aspect('equal', 'box')
        self.axes.set_title("Persistent H1")
        self.axes.set_xlabel("Birth")
        self.axes.set_ylabel("Death")
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.selected = set([])
    
    def onpick(self, evt):
        if evt.artist == self.dgmplot:
            ## Step 1: Highlight point on persistence diagram
            clicked = set(evt.ind.tolist())
            self.selected = self.selected.symmetric_difference(clicked)
            idxs = np.array(list(self.selected))
            if idxs.size > 0:
                self.selected_plot.set_offsets(self.emobj.dgms_[1][idxs, :])
                ## Step 2: Update projective coordinates
                ## TODO: Finish this
            else:
                self.selected_plot.set_offsets(np.zeros((0, 2)))
        self.axes.figure.canvas.draw()
        return True

    

class ApplicationWindow(pg.QtGui.QMainWindow):
    def __init__(self, emobj):
        pg.QtGui.QMainWindow.__init__(self)
        self.setWindowTitle('matplotlib and pyqtpgraph integration example')
        
        self.main_widget = pg.QtGui.QWidget(self)
        self.layout = pg.QtGui.QHBoxLayout(self.main_widget)
        self.persistence_canvas = PersistenceSelector(self.main_widget, emobj, width=5, height=4, dpi=100)

        """
        self.glw = pg.LayoutWidget()
        self.b1_b = pg.QtGui.QPushButton('-')
        self.b1_b.clicked.connect(self.b1_clicked)     
        self.b2_b = pg.QtGui.QPushButton('+')
        self.b2_b.clicked.connect(self.b2_clicked)
        self.glw.addWidget(self.b1_b, row=0, col=0)        
        self.glw.addWidget(self.b2_b, row=0, col=1)
        """

        self.plw = pg.PlotWidget() 
        s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        pos = np.random.randn(10000, 2)
        spots = [{'pos': pos[i, :], 'data': 1} for i in range(pos.shape[0])] + [{'pos': [0,0], 'data': 1}]
        s1.addPoints(spots)
        self.plw.addItem(s1)

        self.layout.addWidget(self.persistence_canvas)      
        #self.layout.addWidget(self.glw)
        self.layout.addWidget(self.plw)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

from CircularCoordinates import CircularCoords

prime = 41
np.random.seed(2)
N = 500
X = np.zeros((N*2, 2))
t = np.linspace(0, 1, N+1)[0:N]**1.2
t = 2*np.pi*t
X[0:N, 0] = np.cos(t)
X[0:N, 1] = np.sin(t)
X[N::, 0] = 2*np.cos(t) + 4
X[N::, 1] = 2*np.sin(t) + 4
X = X[np.random.permutation(X.shape[0]), :]
X = X + 0.2*np.random.randn(X.shape[0], 2)

c = CircularCoords(X, 100, prime = prime)

qApp = pg.QtGui.QApplication([])
aw = ApplicationWindow(c)
aw.show()
sys.exit(qApp.exec_())
