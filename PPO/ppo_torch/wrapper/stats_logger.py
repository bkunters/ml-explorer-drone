import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class StatsLogger:
    """ Log all statistical values over the runtime
    """
    def __init__(self) -> None:
        pass

    def write_csv(self):
        pass

    def plot_csv(self):
        pass

    # TODO: Q&A - clarify Stats Wrapper


class StatsPlotter:
    """Plotting collected network statistics as graphs."""
    def __init__(self, csv_path, img_name, results_path) -> None:
        self.csv_path = csv_path
        self.img_name = img_name
        self.result_path = results_path
    
    def get_files(self, path):
        """Get all files from a path"""
        all_files = glob.glob(os.path.join(path , "/*.csv"))
        files = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            files.append(df)
        return files

    def read_csv(self, date:str):
        """ Use file path to collect all csv files.
            Concats all files and returns a pandas dataframe.
        """
        all_files = glob.glob(os.path.join(self.csv_path, f"*{date}*.csv"))
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        return df

    def plot(self, dataframe, kind='line', hue='', x='x-label', y='y-label', title='title', upper_bound=None, lower_bound=None):
        """ Create a lineplot with seaborn.
            Doc: https://seaborn.pydata.org/tutorial/introduction
        """
        line_plot = sns.relplot(
            data=dataframe, kind=kind,
            x=x, y=y, hue=hue, height=7, aspect=.7)
        line_plot.set(title=title)
        # draw a horizontal line
        if upper_bound:
            line_plot.axhline(upper_bound)
        if lower_bound:
            line_plot.axhline(lower_bound)
        # plot the file to given destination
        file_path = self.result_path + self.img_name
        line_plot.figure.savefig(file_path)

class CSVWriter:
    """Log the network outputs via pandas to a CSV file.
    """
    def __init__(self, filename: str):
        self.count = 1
        self.filename = filename

    def __call__(self, data: dict):
        df = pd.DataFrame(data)
        if not os.path.isfile(self.filename):
            df.to_csv(self.filename, header='column_names')
        else: # else it exists so append without writing the header
            df.to_csv(self.filename, mode='a', header=False)