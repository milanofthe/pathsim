#########################################################################################
##
##                              SCOPE BLOCK (blocks/scope.py)
##
##             This module defines a block for recording time domain data
##
##                                  Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import csv

import numpy as np
import matplotlib.pyplot as plt

from ._block import Block
from ..utils.funcs import dict_to_array
from ..utils.realtimeplotter import RealtimePlotter



# BLOCKS FOR DATA RECORDING =============================================================

class Scope(Block):
    """
    Block for recording time domain data with variable sampling sampling rate.
    
    A time threshold can be set by 'wait' to start recording data after the simulation 
    time is larger then the specified waiting time, i.e. 't - t_wait > 0'. 
    This is useful for recording data only after all the transients have settled.
    
    INPUTS : 
        sampling_rate : (int or None) number of samples per second, default is every timestep
        t_wait        : (float) wait time before starting recording
        labels        : (list of strings) labels for the scope traces, and for the csv
    """

    def __init__(self, sampling_rate=None, t_wait=0.0, labels=[]):
        super().__init__()
        
        #time delay until start recording
        self.t_wait = t_wait

        #params for sampling
        self.sampling_rate = sampling_rate

        #labels for plotting and saving data
        self.labels = labels

        #set recording data and time
        self.recording = {}


    def __len__(self):
        return 0


    def reset(self):
        #reset inputs
        self.inputs = {k:0.0 for k in sorted(self.inputs.keys())}  

        #reset recording data and time
        self.recording = {}


    def read(self):
        """
        return the recorded time domain data and the 
        corresponding time for all input ports
        """

        #just return 'None' if no recording available
        if not self.recording: return None, None

        #reformat the data from the recording dict
        time = np.array(list(self.recording.keys()))
        data = np.array(list(self.recording.values())).T
        return time, data


    def sample(self, t):
        """
        Sample the data from all inputs, and overwrites existing timepoints, 
        since we use a dict for storing the recorded data.
        """
        if t >= self.t_wait: 
            if (self.sampling_rate is None or 
                t * self.sampling_rate > len(self.recording)):
                self.recording[t] = dict_to_array(self.inputs)


    def plot(self, *args, **kwargs):
        """
        Directly create a plot of the recorded data for quick visualization and debugging.
        The 'fig' and 'ax' objects are accessible as attributes of the 'Scope' instance 
        from the outside for saving, or modification, etc.
        """ 

        #just return 'None' if no recording available
        if not self.recording:
            return None

        #get data
        time, data = self.read() 

        #initialize figure
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4), tight_layout=True, dpi=120)
        
        #custom colors
        self.ax.set_prop_cycle(color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])
        
        #plot the recorded data
        for p, d in enumerate(data):
            lb = self.labels[p] if p < len(self.labels) else f"port {p}"
            self.ax.plot(time, d, *args, **kwargs, label=lb)

        #legend labels from ports
        self.ax.legend(fancybox=False)

        #other plot settings
        self.ax.set_xlabel("time [s]")
        self.ax.grid()

        # Legend picking functionality
        lines = self.ax.get_lines()  # Get the lines from the plot
        leg = self.ax.get_legend()   # Get the legend

        # Map legend lines to original plot lines
        lined = dict()  
        for legline, origline in zip(leg.get_lines(), lines):
            # Enable picking within 5 points tolerance
            legline.set_picker(5)  
            lined[legline] = origline

        def on_pick(event):
            legline = event.artist
            origline = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            legline.set_alpha(1.0 if visible else 0.2)
            # Redraw the figure
            self.fig.canvas.draw()  

        #enable picking
        self.fig.canvas.mpl_connect("pick_event", on_pick)

        #show the plot without blocking following code
        plt.show(block=False)


    def save(self, path="scope.csv"):
        """
        Save the recording of the scope to a csv file.
        """

        #check path ending
        if not path.lower().endswith(".csv"):
            path += ".csv"

        #get data
        time, data = self.read() 

        #number of ports and labels
        P, L = len(data), len(self.labels)

        #make csv header
        header = ["time [s]", *[self.labels[p] if p < L else f"port {p}" for p in range(P)]]

        #write to csv file
        with open(path, "w", newline="") as file:
            wrt = csv.writer(file)

            #write the header to csv file
            wrt.writerow(header)

            #write each sample to the csv file
            for sample in zip(time, *data):
                wrt.writerow(sample)




class RealtimeScope(Scope):
    """
    An extension of the 'Scope' block that also initializes a realtime plotter 
    that creates an interactive plotting window while the simulation is running.

    Otherwise implements the same functionality as the regular 'Scope' block.

    NOTE : 
        Due to the plotting being relatively expensive, including this block 
        slows down the simulation significantly but may still be valuable for 
        debugging and testing.

    INPUTS : 
        sampling_rate : (int or None) number of samples per second, default is every timestep
        t_wait        : (float) wait time before starting recording
        labels        : (list of strings) labels for the scope traces, and for the csv
        max_samples   : (int or None) number of samples for realtime display, all per default
    """

    def __init__(self, sampling_rate=None, t_wait=0.0, labels=[], max_samples=None):
        super().__init__(sampling_rate, t_wait, labels)

        #initialize realtime plotter
        self.plotter = RealtimePlotter(max_samples=max_samples, 
                                       update_interval=0.1, 
                                       labels=labels, 
                                       x_label="time [s]", 
                                       y_label="")


    def sample(self, t):
        """
        Sample the data from all inputs, and overwrites existing timepoints, 
        since we use a dict for storing the recorded data.
        """
        if (self.sampling_rate is None or t * self.sampling_rate > len(self.recording)):
            values = dict_to_array(self.inputs)
            self.plotter.update(t, values)
            if t >= self.t_wait: 
                self.recording[t] = values
