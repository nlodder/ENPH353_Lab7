#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np

rewards_key = 'episode_rewards'

class LivePlot(object):
    def __init__(self, outdir, data_key=rewards_key, line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """
        self.outdir = outdir
        self.data_key = data_key
        self.line_color = line_color

        # styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')

        # create persistant figure and axis objects
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Simulation Graph')
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel(data_key)
        self.fig.canvas.set_window_title('Simulation Graph')

        # initialize two lines: one for raw data one for trend
        self.raw_line, = self.ax.plot([], [], color=self.line_color, alpha=0.3, label='Raw')
        self.trend_line, = self.ax.plot([], [], color='red', linewidth=2, label='Trend (Moving Avg)')
        self.ax.legend()

    # def plot(self, env):
    #     if self.data_key is rewards_key:
    #         data = gym.wrappers.Monitor.get_episode_rewards(env)
    #     else:
    #         data = gym.wrappers.Monitor.get_episode_lengths(env)

    #     plt.plot(data, color=self.line_color)

    #     # pause so matplotlib will display
    #     # may want to figure out matplotlib animation or use a different library in the future
    #     plt.pause(0.000001)

    def plot(self, env):
        # Fetch data (Handling different gym versions)
        try:
            if self.data_key == rewards_key:
                data = env.get_episode_rewards()
            else:
                data = env.get_episode_lengths()
        except AttributeError:
            # Fallback for some wrappers
            return

        if len(data) == 0:
            return

        episodes = np.arange(len(data))
        
        # --- DYNAMIC MOVING AVERAGE LOGIC ---
        # Window size is 5% of total episodes, minimum 1, maximum 100
        window_size = int(max(1, min(len(data) * 0.05, 100)))
        
        # Calculate moving average using a convolution
        # This creates a "sliding window" effect
        if len(data) >= window_size:
            weights = np.ones(window_size) / window_size
            moving_avg = np.convolve(data, weights, mode='valid')
            # Adjust x-axis for 'valid' convolution offset
            trend_x = episodes[window_size-1:]
        else:
            moving_avg = data
            trend_x = episodes

        # Update the lines with new data
        self.raw_line.set_data(episodes, data)
        self.trend_line.set_data(trend_x, moving_avg)

        # Rescale the axes to fit new data
        self.ax.relim()
        self.ax.autoscale_view()

        # Pause to refresh the GUI
        plt.pause(0.000001)
