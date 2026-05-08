import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets


class TrainingLiveLossPlotForJupyterNotebooks():
    def __init__(self, figsize=(15, 2)):
        """Live loss plot for Jupyter notebooks. Call update() with new train and val loss values after each epoch."""
        self.train_losses = []
        self.val_losses = []
        self.fig, self.ax = plt.subplots(figsize=figsize)

        self.plot_out = widgets.Output()
        display(self.plot_out)

        plt.close(self.fig)  # Prevent inline auto-display

    def update(self, train_loss, val_loss):
        """Update the plot with new loss values."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self.ax.clear()
        self.ax.plot(self.train_losses, label="Train loss", color="steelblue")
        self.ax.plot(self.val_losses,   label="Val loss",   color="tomato")
        self.ax.legend()
        self.fig.tight_layout()
        self.ax.grid(True, alpha=0.3)

        with self.plot_out:
            self.plot_out.clear_output(wait=True)  # Only clears inside the widget, not tqdm
            display(self.fig)
