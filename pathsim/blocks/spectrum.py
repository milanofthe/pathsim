#########################################################################################
##
##                   SPECTRUM ANALYZER BLOCK (blocks/spectrum.py)
##
##                                Milan Rother 2024
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

class Spectrum(Block):
    """
    Block for fourier spectrum analysis (basically a spectrum analyzer), computes 
    continuous time running fourier transform (RFT) of the incoming signal.
    
    A time threshold can be set by 't_wait' to start recording data only after the 
    simulation time is larger then the specified waiting time, i.e. 't - t_wait > dt'. 
    This is useful for recording the steady state after all the transients have settled.

    An exponential forgetting factor 'alpha' can be specified for realtime spectral 
    analysis. It biases the spectral components exponentially to the most recent signal 
    values by applying a single sided exponential window like this:

        int_0^t x(tau) * exp(alpha*(t-tau)) * exp(-j*omega*tau) dtau

    It is also known as the 'exponentially forgetting transform' (EFT) and a form of 
    short time fourier transform (STFT). It is implemented as a 1st order statespace model 
        
        dx/dt = - alpha * x +  exp(-j*omega*t) * u

    , where 'u' is the input signal and 'x' is the state variable that represents the 
    complex fourier coefficient to the frequency 'omega'. The ODE is integrated using the 
    numerical integration engine of the block.

    NOTE : 
        This block is very slow! But it is valuable for long running simulations 
        with few evaluation frequencies, where just FFT'ing the time series data 
        wouldnt be efficient OR if only the evaluation at weirdly spaced frequencies 
        is required. Otherwise its more efficient to just do an FFT on the time 
        series recording.
    
    INPUTS : 
        freq   : (list or array) list of evaluation frequencies for RFT
        t_wait : (float) t_wait time before starting RFT
        alpha  : (float) exponential forgetting factor for realtime spectrum
        labels : (list of strings) labels for the inputs
    """

    def __init__(self, freq=[], t_wait=0.0, alpha=0.0, labels=[]):
        super().__init__()
        
        #time delay until start recording
        self.t_wait = t_wait

        #local integration time
        self.time = 0.0

        #forgetting factor
        self.alpha = alpha

        #labels for plotting and saving data
        self.labels = labels

        #frequency
        self.freq = np.array(freq)
        self.omega = 2.0 * np.pi * self.freq


    def __len__(self):
        return 0


    def set_solver(self, Solver, **solver_args):
        
        if self.engine is None:
            
            #initialize the numerical integration engine with kernel
            def _f(x, u, t):
                return np.kron(u, np.exp(-1j * self.omega * t))

            def _f_decay(x, u, t):
                return np.kron(u, np.exp(-1j * self.omega * t)) - self.alpha * x

            #initialize depending on forgetting factor
            if self.alpha == 0.0: self.engine = Solver(0.0, _f, None, **solver_args)
            else: self.engine = Solver(0.0, _f_decay, None, **solver_args)

        else:

            #change solver if already initialized
            self.engine = self.engine.change(Solver, **solver_args)

        
    def reset(self):
        #reset inputs
        self.inputs = {k:0.0 for k in sorted(self.inputs.keys())}  

        #local integration time
        self.time = 0.0

        #reset numeric integration engine -> resets the spectrum
        self.engine.reset()


    def read(self):

        #just return 'None' if no engine initialized
        if self.engine is None:
            return self.freq, np.zeros_like(self.freq)

        #get state from engine
        state = self.engine.get()

        #catch case where state has not been updated
        if np.all(state == self.engine.initial_value):
            return self.freq, np.zeros_like(self.freq)

        #reshape state into spectra
        spec = np.reshape(state, (-1, len(self.freq)))

        #rescale spectrum and return it
        if self.alpha != 0.0:
            return self.freq, spec * self.alpha / (1.0 - np.exp(-self.alpha*self.time))

        #return spectrum from RFT
        return self.freq, spec/self.time


    def solve(self, t, dt):
        #effective time for integration
        _t = t - self.t_wait
        if _t > dt:

            #update local integtration time
            self.time = _t
            
            #advance solution of implicit update equation
            return self.engine.solve(dict_to_array(self.inputs), _t, dt)

        #no error 
        return 0.0


    def step(self, t, dt):
        #effective time for integration
        _t = t - self.t_wait
        if _t > dt:

            #update local integtration time
            self.time = _t
            
            #compute update step with integration engine
            return self.engine.step(dict_to_array(self.inputs), _t, dt)

        #no error estimate
        return True, 0.0, 0.0, 1.0


    def plot(self, *args, **kwargs):
        """
        Directly create a plot of the recorded data for visualization.
        The 'fig' and 'ax' objects are accessible as attributes of the 'Spectrum' instance 
        from the outside for saving, or modification, etc.
        """

        #just return 'None' if no engine initialized
        if self.engine is None:
            return None

        #get data
        freq, data = self.read()        

        #initialize figure
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4), tight_layout=True, dpi=120)

        #custom colors
        self.ax.set_prop_cycle(color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])

        #plot magnitude in dB and add label
        for p, d in enumerate(data):
            lb = self.labels[p] if p < len(self.labels) else f"port {p}"
            self.ax.plot(freq, abs(d), *args, **kwargs, label=lb)

        #legend labels from ports
        self.ax.legend(fancybox=False)

        #other plot settings
        self.ax.set_xlabel("freq [Hz]")
        self.ax.set_ylabel("magnitude")
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


    def save(self, path="spectrum.csv"):
        """
        save the recording of the spectrum to a csv file        
        """

        #check path ending
        if not path.lower().endswith(".csv"):
            path += ".csv"

        #get data
        freq, data = self.read() 

        #number of ports and labels
        P, L = len(data), len(self.labels)

        #construct port labels
        port_labels = [self.labels[p] if p < L else f"port {p}" for p in range(P)]

        #make csv header
        header = ["freq [Hz]"]
        for l in port_labels:
            header.extend([f"Re({l})", f"Im({l})"])

        #write to csv file
        with open(path, "w", newline="") as file:
            wrt = csv.writer(file)

            #write the header to csv file
            wrt.writerow(header)

            #write each sample to the csv file
            for f, *dta in zip(freq, *data):
                sample = [f]
                for d in dta:
                    sample.extend([np.real(d), np.imag(d)])
                wrt.writerow(sample)


class RealtimeSpectrum(Spectrum):

    """
    An extension of the 'Spectrum' block that also initializes a realtime plotter that 
    creates an interactive plotting window while the simulation is running. 
    
    Otherwise implements the same functionality as the regular 'Spectrum' block.
    
    NOTE :
        Due to the plotting being relatively expensive, including this block slows down 
        the simulation significantly but may still be valuable for debugging and testing.

    INPUTS : 
        freq      : (list or array) list of evaluation frequencies for RFT
        t_wait    : (float) t_wait time before starting RFT
        alpha     : (float) exponential forgetting factor for realtime spectrum
        labels    : (list of strings) labels for the inputs
    """

    def __init__(self, freq=[], t_wait=0.0, alpha=0.0, labels=[]):
        super().__init__(freq, t_wait, alpha, labels)

        #initialize realtime plotter
        self.plotter = RealtimePlotter(update_interval=0.1, 
                                       labels=labels, 
                                       x_label="freq [Hz]", 
                                       y_label="magnitude")


    def step(self, t, dt):
        #effective time for integration
        _t = t - self.t_wait
        if _t > dt:

            #update local integtration time
            self.time = _t

            if self.time > 2*dt:
                #update realtime plotter
                _, data = self.read()
                self.plotter.update_all(self.freq, abs(data))
            
            #compute update step with integration engine
            return self.engine.step(dict_to_array(self.inputs), _t, dt)

        #no error estimate
        return True, 0.0, 0.0, 1.0