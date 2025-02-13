import ipywidgets as widgets
from ipywidgets import Output
from IPython.display import display, clear_output, HTML
import numpy as np
import pandas as pd
import pickle
import processing as pro
import plotting as plo


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
        
        self.get_data()
        self.df = pro.process_pkl(self.data_dict)
        
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

    def change_coordinates(self, new_coordinates):
        """ Change the current coordinate system and transform the specimen data accordingly """
        self.coordinates = new_coordinates
        self.manager.interface.update_plot()
        
    def get_coordinates(self):
        """ Returns the current coordinates """
        return self.coordinates


######### Interface class ############ 

class Interface:
    def __init__(self, specimen=None, manager=None):
        self.manager = manager
        self.df = self.manager.df
        self.working_df = self.df.copy()
        self.coordinate_system = CoordinateSystem(coordinates='geographic', manager=self.manager)        
        self.components_list = sorted(self.df['component'].unique(), key=lambda x: (not x.isalpha(), not x.isdigit(), x))
        self.mean_data = []

        # setup widgets
        self.show_data_widgets()
        self.modify_data_widgets()
        self.compute_mean_widgets()
        self.decay_widgets()

        # setup display and layout
        self.message_output_area1 = Output()
        self.plot_output_area1 = Output(layout={'width': '80%', 'height': '600px', 'overflow_y': 'hidden'})
        self.plot_output_area2 = Output(layout={'width': '98%', 'height': '600px', 'overflow_y': 'auto', 'overflow_x': 'hidden'})
        self.plot_output_area3 = Output()

        self.plot_container1 = widgets.VBox([self.plot_output_area1], layout={'width': '100%', 'padding': '0px', 'margin': '0px'})
        self.plot_container2 = widgets.VBox([self.plot_output_area2], layout={'width': '100%', 'padding': '0px', 'margin': '0px'})
        
        # widget layout
        hb_layout = widgets.Layout(align_items='flex-start',  width='90%', padding='5px', margin='5px')
        
        display(widgets.HBox([self.restore_button, self.update_data_button, self.drop_point_textbox, self.reassign_textbox], layout=hb_layout),
                widgets.HBox([self.hide_selection_button, self.update_plot_button, 
                              self.lines_checkbox, self.planes_checkbox, self.normals_checkbox, self.mads_checkbox, self.coordinates_dropdown,], layout=hb_layout),
                self.data_selection_container,
                widgets.HBox([self.hide_means_button, self.calculate_button, self.add_mean_button], layout=hb_layout), 
                self.mean_widgets_container,
                self.message_output_area1,
                widgets.HBox([self.plot_container1, self.plot_container2], layout={'width': '100%', 'padding': '0px', 'margin': '0px'}),
                widgets.HBox([self.hide_spectra_button, self.decay_spectra_button, self.dMdD_checkbox, self.AF_log_checkbox]),
                self.spectra_selection_container,
                self.plot_output_area3)

        self.display_data(self.working_df)
        self.update_plot()
        
        
    #### Widgets setup ####

    def modify_data_widgets(self):

        self.restore_button = widgets.Button(description="restore", layout=widgets.Layout(width="80px"))
        self.restore_button.on_click(self.restore_data)

        self.update_data_button = widgets.Button(description="update data", layout=widgets.Layout(width="120px"))
        self.update_data_button.on_click(self.update_data)
        
        self.drop_point_textbox = widgets.Text(description='drop result:', placeholder='e.g. 19', layout=widgets.Layout(width='200px'),
                                            style={'description_width': '100px'} )
        self.reassign_textbox = widgets.Text(description='reassign comp:', placeholder='e.g. 19: B', layout=widgets.Layout(width='200px'),
                                            style={'description_width': '100px'} )

    def show_data_widgets(self):
        self.hide_selection_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
        self.hide_selection_button.on_click(self.toggle_data_selection)

        self.update_plot_button = widgets.Button(description="update plot", layout=widgets.Layout(width="120px"))
        self.update_plot_button.on_click(self.update_plot)
        
        self.lines_checkbox = widgets.Checkbox(value=False, description='lines', layout=widgets.Layout(width="auto"))
        self.planes_checkbox = widgets.Checkbox(value=False, description='planes', layout=widgets.Layout(width="auto"))
        self.normals_checkbox = widgets.Checkbox(value=False, description='plane normals', layout=widgets.Layout(width="auto"))
        self.mads_checkbox = widgets.Checkbox(value=False, description='MAD angles', layout=widgets.Layout(width="auto"))

        self.coordinates_dropdown = widgets.Dropdown(options=['specimen', 'geographic', 'tectonic'], value=self.coordinate_system.coordinates, description='Coordinates:', 
                                                    layout=widgets.Layout(width='200px'))
        self.coordinates_dropdown.observe(self.update_coordinates, names='value')

        self.data_selection_checkbox_states = {}
        self.data_selection_container = widgets.HBox([])

    def compute_mean_widgets(self):
        self.hide_means_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
        self.hide_means_button.on_click(self.toggle_means_widgets)

        self.add_mean_button = widgets.Button(description="+ mean", layout=widgets.Layout(width="80px"))
        self.add_mean_button.on_click(self.add_mean_row)

        self.calculate_button = widgets.Button(description="calculate", layout=widgets.Layout(width="120px"))
        self.calculate_button.on_click(self.calculate_mean)

        self.mean_widgets = []
        self.mean_widgets_container = widgets.VBox([])
        self.mean_widgets_container.layout.display = 'none'

    def decay_widgets(self):
        self.hide_spectra_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
        self.hide_spectra_button.on_click(self.toggle_spectra_selection)
        
        self.decay_spectra_button = widgets.Button(description="decay spectra", layout=widgets.Layout(width="150px"))
        self.decay_spectra_button.on_click(self.show_decay_spectra)

        self.dMdD_checkbox = widgets.Checkbox(value=False, description='show dMdD', layout=widgets.Layout(width="auto"))
        self.AF_log_checkbox = widgets.Checkbox(value=False, description='AF log scale', layout=widgets.Layout(width="auto"))

        self.spectra_selection_checkbox_states = {}
        self.spectra_selection_container = widgets.HBox([])


    #### methods ####

    def display_data(self, df):
        """ ... """
        simplified_df = df.drop(['coefficients', 'treatment', 'gcs', 'gcg', 'gct'], axis=1)
        
        if self.coordinate_system.coordinates == 'specimen':
            simplified_df = simplified_df.drop(['Dg', 'Ig', 'Dt', 'It'], axis=1)
        elif self.coordinate_system.coordinates == 'geographic':
            simplified_df = simplified_df.drop(['Ds', 'Is', 'Dt', 'It'], axis=1)
        elif self.coordinate_system.coordinates == 'tectonic':
            simplified_df = simplified_df.drop(['Ds', 'Is', 'Dg', 'Ig'], axis=1)

        with self.plot_output_area2:
            clear_output(wait=True)
            display(simplified_df)
            #display(HTML(f"""<div style="max-height: 400px; overflow-y: auto; border: 1px solid #ccc;">
            #            {simplified_df.to_html()}</div>"""))

    def toggle_data_selection(self, b=None):
        """ Show/hide checkboxes while preserving selections """
        if self.data_selection_container.children:  # if panel is open...
            self.data_selection_checkbox_states = {cb.description: cb.value for cb in self.data_selection_container.children} # save current checkbox states...
            self.data_selection_container.children = []  # and hide panel
        else:
            self.data_selection_container.children = [widgets.Checkbox(value=False, description=label, indent=False, layout=widgets.Layout(width='auto', 
                                                                        margin='0px 30px 0px 30px', padding='0px')) for label in self.components_list]

    def update_coordinates(self, change):
        """ Update the coordinate system based on user selection """
        new_coordinates = change['new']
        self.coordinates_dropdown.value = new_coordinates
        self.coordinate_system.change_coordinates(new_coordinates)

    def update_plot(self, b=None):
        """ ... """
        coordinates = self.coordinate_system.get_coordinates()
        show_lines = self.lines_checkbox.value
        show_planes = self.planes_checkbox.value
        show_normals = self.normals_checkbox.value
        show_mads = self.mads_checkbox.value

        if self.data_selection_container.children:
            selected_comps = [cb.description for cb in self.data_selection_container.children if cb.value] # if checkboxes are visible, get selected indices from them
        else:
            selected_comps = [i for i, v in self.data_selection_checkbox_states.items() if v]

        selected_df = self.working_df[self.working_df['component'].isin(selected_comps)]
        
        with self.plot_output_area1:
            clear_output(wait=True)
            plo.stat_stereo(coordinates, selected_df, self.mean_data, show_lines, show_planes, show_normals, show_mads)

    def update_data(self, b=None):
        """ ... """
        drop_list = [int(x.strip()) for x in self.drop_point_textbox.value.split(',') if x.strip()]
        self.working_df = self.working_df.drop(index=drop_list)

        reassignments = {int(pair.split(':')[0].strip()): pair.split(':')[1].strip() 
                         for pair in self.reassign_textbox.value.split(',') if ':' in pair}
        self.working_df.loc[reassignments.keys(), 'component'] = [str(v) for v in reassignments.values()]

        self.display_data(self.working_df)
    
    def restore_data(self, b=None):
        """ ... """
        self.working_df = self.df.copy()
        self.display_data(self.working_df)

    def toggle_means_widgets(self, b=None):
        """ Show/hide the means widgets """
        if self.mean_widgets_container.layout.display == 'none': self.mean_widgets_container.layout.display = ''
        else: self.mean_widgets_container.layout.display = 'none'

    def add_mean_row(self, b=None):
        """ Add a new row for mean calculation """
        comps_textbox = widgets.Text(description='component(s):', placeholder='e.g. A, B+C', layout=widgets.Layout(width='200px'),
                                        style={'description_width': '100px'} )
        sample_checkbox = widgets.Checkbox(value=False, description='show samples', layout=widgets.Layout(width="auto"), margin='20px')
        flip_checkbox = widgets.Checkbox(value=False, description='force same polarity', layout=widgets.Layout(width="auto"), margin='20px')
        lines_checkbox = widgets.Checkbox(value=False, description='fisher', layout=widgets.Layout(width="auto"), margin='20px')
        planes_checkbox = widgets.Checkbox(value=False, description='GC intersection', layout=widgets.Layout(width="auto"), margin='20px')
        mixed_checkbox = widgets.Checkbox(value=False, description='mixed', layout=widgets.Layout(width="auto"), margin='20px')
        colors_textbox = widgets.Text(description='color:', placeholder='e.g. red, blue', layout=widgets.Layout(width='200px'),
                                        style={'description_width': '100px'} )
        remove_mean_button = widgets.Button(description="X", layout=widgets.Layout(width="30px"))
        remove_mean_button.on_click(lambda x: self.remove_mean_row(widget_row))

        widget_row = widgets.HBox([comps_textbox, sample_checkbox, flip_checkbox, lines_checkbox, planes_checkbox, mixed_checkbox, colors_textbox, remove_mean_button],
                                 layout=widgets.Layout(width='100%', justify_content='flex-start'))
        
        self.mean_widgets.append((comps_textbox, sample_checkbox, flip_checkbox, lines_checkbox, planes_checkbox, mixed_checkbox, colors_textbox, widget_row))
        self.mean_widgets_container.children += (widget_row,)


    def calculate_mean(self, b=None):
            
        self.mean_data = []
        with self.message_output_area1:
            self.message_output_area1.clear_output(wait=True)
            
            for row in self.mean_widgets:
                comps_tb, sample_cb, flip_cb, lines_cb, planes_cb, mixed_cb, colors_tb, widget_row = row
                    
                if comps_tb.value: 
                    
                    # access values from each widget in the row
                    label = comps_tb.value
                    
                    comp_df = self.get_components(comps_tb.value)

                    show_samples = sample_cb.value
                    flip = flip_cb.value
                    
                    if lines_cb.value:
                        lin_mean, _ = pro.get_fisher_mean(comp_df, self.coordinate_system.coordinates, flip=flip, w_gcs=False)
                        if lin_mean:
                            print (f"component {comps_tb.value} fisher mean (n={lin_mean['n']}), dec: {lin_mean['dec']:.2f}, inc: {lin_mean['inc']:.2f}," 
                                   f" a95: {lin_mean['alpha95']:.2f},  k: {lin_mean['k']:.2f}")
                    else: lin_mean = None
                        
                    if planes_cb.value:
                        gc_int = pro.gc_intersection(comp_df, self.coordinate_system.coordinates)
                        if gc_int:
                            print (f"component {comps_tb.value} GC intersection (m={gc_int['n']}), dec: {gc_int['dec']:.2f}, inc: {gc_int['inc']:.2f}," 
                                   f" MAD: {gc_int['mad']:.2f}")
                    else: gc_int = None
    
                    if mixed_cb.value:
                        mix_mean, gc_endpts = pro.get_fisher_mean(comp_df, self.coordinate_system.coordinates, flip=flip, w_gcs=True)
                        if mix_mean:
                            print (f"component {comps_tb.value} mixed fisher mean (n={mix_mean['n']}/m={mix_mean['m']}), dec: {mix_mean['dec']:.2f}," 
                                   f" inc: {mix_mean['inc']:.2f}, a95: {mix_mean['alpha95']:.2f},  k: {mix_mean['k']:.2f}")
                    else:
                        mix_mean = None
                        gc_endpts = None
                        
                    if colors_tb.value:
                        color = colors_tb.value
                    else: color = 'k'

                    self.mean_data.append([label, comp_df, lin_mean, gc_int, mix_mean, gc_endpts, show_samples, color])
                
                self.update_plot()

    
    def get_components(self, comp_str):
        """ ... """
        
        if "+" in comp_str:
            dfs = [self.working_df[self.working_df['component'] == str(x.strip())] for x in comp_str.split("+")]
            comp_df = pd.concat(dfs, ignore_index=True)
        else:
            comp_df = self.working_df[self.working_df['component'] == str(comp_str)]

        return comp_df
        
        
    def remove_mean_row(self, widget_row):
        """ Remove a row of fit-line widgets """
        self.mean_widgets = [w for w in self.mean_widgets if w[-1] != widget_row]
        self.mean_widgets_container.children = [w[-1] for w in self.mean_widgets]
        

    def toggle_spectra_selection(self, b=None):
        """ Show/hide checkboxes while preserving selections """
        if self.spectra_selection_container.children:  # if panel is open...
            self.spectra_selection_checkbox_states = {cb.description: cb.value for cb in self.spectra_selection_container.children} # save current checkbox states...
            self.spectra_selection_container.children = []  # and hide panel
        else:
            self.spectra_selection_container.children = [widgets.Checkbox(value=False, description=label, indent=False, layout=widgets.Layout(width='auto', 
                                                                            margin='0px 30px 0px 30px', padding='0px')) for label in self.components_list]

    
    def show_decay_spectra(self, b=None):

        if self.spectra_selection_container.children:
            selected_comps = [cb.description for cb in self.spectra_selection_container.children if cb.value] # if checkboxes are visible, get selected indices from them
        else:
            selected_comps = [i for i, v in self.spectra_selection_checkbox_states.items() if v]
        
        mean_treatments, mean_coefficients, mean_dMdD = pro.mean_decay(selected_comps, self.working_df)

        show_dMdD = self.dMdD_checkbox.value
        AF_log = self.AF_log_checkbox.value

        with self.plot_output_area3:
            self.plot_output_area3.clear_output(wait=True)
            plo.decay_spectra(selected_comps, self.working_df, mean_treatments, mean_coefficients, mean_dMdD, show_dMdD, AF_log)


