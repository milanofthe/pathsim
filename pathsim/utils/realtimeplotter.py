#########################################################################################
##
##                              REALTIME PLOTTER CLASS 
##                            (utils/realtimeplotter.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use("fast")

import numpy as np

import time
from collections import deque


# PLOTTER CLASS =========================================================================


class RealtimePlotter:
    """
    Class that manages a realtime plotting window that 
    can stream in x-y-data and update accordingly
        
    INPUTS:
        max_samples     : (int) maximum number of samples to plot
        update_interval : (float) time in seconds between refreshs
        labels          : (list of strings) labels for plot traces
        x_label         : (str) label for x-axis
        y_label         : (str) label for y-axis
    """

    def __init__(self, max_samples=None, update_interval=1, labels=[], x_label="", y_label=""):
        
        #plotter settings
        self.max_samples = max_samples
        self.update_interval = update_interval
        self.labels = labels
        self.x_label = x_label
        self.y_label = y_label
        
        #figure initialization
        self.fig, self.ax = plt.subplots(nrows=1, 
                                         ncols=1, 
                                         figsize=(8,4), 
                                         tight_layout=True, 
                                         dpi=120)
        
        #custom colors
        self.ax.set_prop_cycle(color=["#e41a1c", 
                                      "#377eb8", 
                                      "#4daf4a", 
                                      "#984ea3", 
                                      "#ff7f00"])
        
        #plot settings
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.grid(True)
        
        #data and lines (traces) for plotting
        self.lines = []
        self.data = []
        
        #tracking update time
        self.last_update = time.time()
        
        #flag for running mode
        self.is_running = True
        
        # Connect the close event to the on_close method
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        
        # Initialize legend
        self.legend = None
        self.lined = {}
        
        #show the plotting window
        self.show()


    def update_all(self, x, y):

        #not running? -> quit early
        if not self.is_running:
            return False
    
        #no data yet? -> initialize lines
        if not self.data:

            #data initialization
            for i, val in enumerate(y):
                self.data.append({"x": [], "y": []})

                #label selection and line (trace) initialization
                label = self.labels[i] if i < len(self.labels) else f"port {i}"
                line, = self.ax.plot([], [], lw=1.5, label=label)
                self.lines.append(line)

            # Create legend
            self.legend = self.ax.legend(fancybox=False, ncols=int(np.ceil(len(y)/4)), loc="lower left")
            self._setup_legend_picking()

        #check if new update of plot is required
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:        
            
            #replace the data
            for i, val in enumerate(y):
                self.data[i]["x"] = x
                self.data[i]["y"] = val

            self._update_plot()
            self.last_update = current_time

        return True


    def update(self, x, y):
        
        #not running? -> quit early
        if not self.is_running:
            return False

        #no data yet? -> initialize lines
        if not self.data:

            #vectorial data -> multiple traces
            if np.isscalar(y):

                #size of data
                n = 1

                #check if rolling window plot
                if self.max_samples is None:
                    self.data.append({"x": [], "y": []})
                else:
                    self.data.append({"x": deque(maxlen=self.max_samples), 
                                      "y": deque(maxlen=self.max_samples)})

                #label selection and line (trace) initialization
                label = self.labels[0] if self.labels else "port 0"
                line, = self.ax.plot([], [], lw=1.5, label=label)
                self.lines.append(line)

            else:
                
                #size of data
                n = len(y)

                for i in range(n):
                    
                    #check if rolling window plot
                    if self.max_samples is None:
                        self.data.append({"x": [], "y": []})
                    else:
                        self.data.append({"x": deque(maxlen=self.max_samples), 
                                          "y": deque(maxlen=self.max_samples)})

                    #label selection and line (trace) initialization
                    label = self.labels[i] if i < len(self.labels) else f"port {i}"
                    line, = self.ax.plot([], [], lw=1.5, label=label)
                    self.lines.append(line)

            # Create legend
            self.legend = self.ax.legend(fancybox=False, ncols=int(np.ceil(n/4)), loc="lower left")
            self._setup_legend_picking()

        #add the data
        if np.isscalar(y):
            self.data[0]["x"].append(x)
            self.data[0]["y"].append(y)
        else:
            for i, val in enumerate(y):
                self.data[i]["x"].append(x)
                self.data[i]["y"].append(val)

        #check if new update of plot is required
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self._update_plot()
            self.last_update = current_time

        return True


    def _update_plot(self):

        #set the data to the lines (traces) of the plot
        for i, line in enumerate(self.lines):
            line.set_data(self.data[i]["x"], self.data[i]["y"])

        #rescale the window
        self.ax.relim()
        self.ax.autoscale_view()

        #redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def show(self):
        plt.show(block=False)


    def on_close(self, event):
        self.is_running = False


    def _setup_legend_picking(self):

        #setup the picking for the legend lines
        for legline, origline in zip(self.legend.get_lines(), self.lines):
            legline.set_picker(5)  # 5 points tolerance
            self.lined[legline] = origline

        def on_pick(event):
            legline = event.artist
            origline = self.lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            legline.set_alpha(1.0 if visible else 0.2)
            self.fig.canvas.draw()

        self.fig.canvas.mpl_connect("pick_event", on_pick)