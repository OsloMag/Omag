import ipywidgets as widgets
from ipywidgets import Output
from IPython.display import display, clear_output, HTML
import numpy as np
import pandas as pd
import pickle
import processing as pro
import plotting as plo
from qgridnext import show_grid


######### Manager class ############ 

class StatsManager:
    def __init__(self, infile):
        """
        initialize a Manager class with dataframe containing multiple specimens
        """
        self.infile = infile
        self.data_dict = {}
        self.df = None
        self.specimen_names = None
        self.interface = None
        self.output_area = Output()
        
        self.get_data()
        self.df = pro.process_pkl(self.data_dict)
        simplified_df = self.df.drop(['coefficients', 'treatment', 'gcs', 'gcg', 'gct'], axis=1)
        
        with self.output_area:
            clear_output(wait=True)

            display(HTML(f"""<div style="max-height: 400px; overflow-y: auto; border: 1px solid #ccc;">
                        {self.df.to_html()}</div>"""))

        display(self.output_area)

        self.interface = Interface(manager=self)

    def get_data(self):
        
        if self.infile.endswith('.pkl'):
            with open(self.infile, 'rb') as f:
                self.data_dict = pickle.load(f)
                self.specimen_names = list(self.data_dict.keys())

        else: print("Unsupported file type.")


######### Coordinate System class ############ 

class CoordinateSystem:
    """ Maintains the current coordinate system and handles changes to it """
    def __init__(self, coordinates='geographic', manager=None):
        self.coordinates = coordinates  # default coordinates
        self.manager = manager

    def change_coordinates(self, new_coordinates, specimen):
        """ Change the current coordinate system and transform the specimen data accordingly """
        self.coordinates = new_coordinates
        
    def get_coordinates(self):
        """ Returns the current coordinates """
        return self.coordinates


######### Interface class ############ 

class Interface:
    def __init__(self, specimen=None, manager=None):
        self.manager = manager
        self.df = self.manager.df
        self.coordinate_system = CoordinateSystem(coordinates='geographic', manager=self.manager)
        self.message_output_area1 = Output()
        self.message_output_area2 = Output()
        self.plot_output_area1 = Output()
        #self.plot_output_area2 = Output()
        
        self.components_list = self.df['component'].unique()

        with self.message_output_area1:
            clear_output(wait=True)
            display(HTML(f'components available in dataframe: {", ".join(map(str, self.components_list))}'))

        self.component_dfs = []
        self.component_colors = []
        self.fisher_means = []
        self.pests = []
        self.colors = []

        # widgets
        self.comps_textbox = widgets.Text(description='components to plot:', placeholder='e.g. A, B', layout=widgets.Layout(width='300px'),
                                            style={'description_width': '150px'} ) 
        self.flip_checkbox = widgets.Checkbox(value=False, description='Flip to common polarity', layout=widgets.Layout(width="auto"), margin='20px')
        self.w_gcs_checkbox = widgets.Checkbox(value=False, description='Include planes', layout=widgets.Layout(width="auto"), margin='20px')
        self.colors_textbox = widgets.Text(description='component colors:', placeholder='e.g. red, blue', layout=widgets.Layout(width='350px'),
                                            style={'description_width': '150px'} ) 
        space = widgets.Label("", layout=widgets.Layout(width="60px"))
        
        self.update_plot_button = widgets.Button(description="update plots", layout=widgets.Layout(width="120px"))  # toggle the data filter panel
        self.update_plot_button.on_click(self.update_plots)
        
        # widget layout
        display(self.message_output_area1,
                widgets.HBox([self.comps_textbox, self.flip_checkbox, self.w_gcs_checkbox, self.colors_textbox, space, self.update_plot_button]),
                self.message_output_area2,
                self.plot_output_area1)
                #self.plot_output_area2)
        
    def update_plots(self, b=None):
        """ ... """
        self.get_components()
        self.get_colors()
        self.compute_fisher_means()
        self.get_mean_decay()
        self.overview_plot()

    def get_components(self):
        """ ... """
        if self.comps_textbox: 
            raw_string = self.comps_textbox.value
            split_str = [s.strip() for s in raw_string.split(",")]

        self.component_dfs = []
        for comp in split_str:
            if "+" in comp:
                dfs = [self.df[self.df['component'] == str(x.strip())] for x in comp.split("+")]
                merged_dfs = pd.concat(dfs, ignore_index=True)
                self.component_dfs.append(merged_dfs)
            else:
                df = self.df[self.df['component'] == str(comp)]
                self.component_dfs.append(df)
        
        for i in range(len(self.component_dfs)):
            self.colors.append('green')

    def get_colors(self):
        """ ... """
        if self.colors_textbox: 
            raw_string = self.colors_textbox.value
            self.colors = [str(s.strip()) for s in raw_string.split(",")]

    def compute_fisher_means(self):
        """ ... """
        with self.message_output_area2:
            clear_output(wait=False)

        flip = self.flip_checkbox.value
        w_gcs = self.w_gcs_checkbox.value
        
        self.fisher_means = []
        for i, comp in enumerate(self.component_dfs): 
            fmean, pests = pro.get_fisher_mean(comp, self.coordinate_system.coordinates, flip=flip, w_gcs=w_gcs)
            self.fisher_means.append(fmean)
            self.pests.append(pests)

            print (self.pests)

            with self.message_output_area2:
                print (f"Comp. {i} mean (n={fmean['n']}): Dec: {fmean['dec']:.2f}, Inc: {fmean['inc']:.2f}, a95: {fmean['alpha95']:.2f}")

    def get_mean_decay(self):
        """ ... """
        self.mean_treatments, self.mean_coefficients = pro.mean_decay(self.component_dfs)

    def overview_plot(self):
        """ ... """
        with self.plot_output_area1:
            clear_output(wait=True)
            plo.overview_plt(self.component_dfs, self.fisher_means, self.pests, self.mean_treatments, self.mean_coefficients, self.coordinate_system.coordinates, self.colors)
