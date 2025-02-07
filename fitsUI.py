import ipywidgets as widgets
from ipywidgets import Output
from IPython.display import display, clear_output, HTML
import numpy as np
import pickle
import processing as pro
import plotting as plo


######### Manager class ############ 

class FitsManager:
    def __init__(self, infile):
        """ Class to initialize everything and manage inputs and outputs """
        self.infile = infile          # input file
        self.data_dict = {}           # empty dictionary that will hold all the data
        self.df = None                # some imports put the data in a dataframe before passing to the dictionary
        self.output_area = Output()   # setup an initial display area
        self.interface_area = Output()
        self.specimen_names = None    # list over the specimen / sample names in the file
        self.selected_specimen_name = None

        # import the data
        self.get_data()               # calls method to import the data
        if len(self.specimen_names) == 0:  # make sure there is data in the file
            print ('No specimen names available.')
        else:
            self.selected_specimen_name = self.specimen_names[0]  # defaults to first specimen in the infile
            self.setup_widgets()           # setup initial widgets
            self.specimen = Specimen(manager=self)   # initialize Specimen class

            with self.interface_area:
                clear_output(wait=True)
                self.interface = Interface(self.specimen, manager=self)  # initialize Interface

    def get_data(self):
        """ Method to get data from infile; at the moment takes pickled files and jr6 files """
        # pickled files are used for saving the data out and can be reimported with all the fits preserved
        if self.infile.endswith('.pkl'):    
            with open(self.infile, 'rb') as f:
                self.data_dict = pickle.load(f)
                self.specimen_names = list(self.data_dict.keys())

        # jr6 files with the .jr6 suffix
        elif self.infile.endswith('.jr6'):
            self.df = pro.import_jr6(self.infile)  # use a dataframe temporarily to hold the data
            self.init_dict() # set up the dictionary to pass the data into

        # other file-types to be added later
        else: print("Unsupported file type.")
  
    def init_dict(self):
        """ Setup the initial dictionary (if importing new data) """
        self.specimen_names = list(self.df['specimen'].unique())  # get all the specimen names in the dataframe
        for specimen in self.specimen_names:
            specimen_raw_df = self.df[self.df['specimen'] == specimen].reset_index(drop=True)  # make dict entries for each specimen
            self.data_dict[specimen] = {
                'raw': specimen_raw_df,   # the raw data taken directly from the infile
                'filtered': [],           # the filtered dataset (if any measurements are removed during processing)
                'lines': [],              # the details of any fitted lines
                'planes': [],             # the details of any fitted planes
                'coefficients_norm': []}  # the normalized coefficients of linear model fits of the interpreted components

    def setup_widgets(self):
        """ Initializes the initial widgets in the upper display panel """
        self.specimen_selector = widgets.Dropdown(options=self.specimen_names, value=self.selected_specimen_name, description='specimen:')
        self.specimen_selector.observe(self.on_specimen_change, names='value')
        self.save_text_box = widgets.Text(description='save to:', placeholder='file prefix') 
        
        # widget layout
        self.controls = widgets.HBox([self.specimen_selector, self.save_text_box])
        display(self.controls, self.output_area, self.interface_area)
        
    def on_specimen_change(self, change):
        """ Method to update the selected specimen when a new one is selected from the dropdown menu or cycled through the arrows """
        if change['name'] == 'value':
            self.selected_specimen_name = self.specimen_selector.value
            self.specimen = Specimen(manager=self)

            with self.interface_area:
                clear_output(wait=True)
                self.interface = Interface(self.specimen, manager=self)

    def save_data(self, b=None):
        """ Save data out to pickled file """
        if self.save_text_box: 
            filename = self.save_text_box.value + '.pkl'
        else:
            filename = 'data_dict.pkl'
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.data_dict, f)

                with self.interface.message_output_area1:
                    print(f'Data saved to {filename}')

        except Exception as e:
            print (f'Error saving data: {e}')

    def get_selected_specimen_name(self):
        """ Returns the selected specimen name """
        return self.selected_specimen_name


######### Specimen class ############ 
        
class Specimen:
    """ Class that maintains the basic specimen data (raw and filtered) """
    def __init__(self, manager=None):
        self.manager = manager
        self.raw = self.manager.data_dict[self.manager.selected_specimen_name]['raw']
        if not isinstance(self.manager.data_dict[self.manager.selected_specimen_name]['filtered'], list): # if not an empty list...
            self.filtered = self.manager.data_dict[self.manager.selected_specimen_name]['filtered']
            self.indices_removed = set(self.raw.index) - set(self.filtered.index)  # get the indices that were filtered previously
        else:
            self.filtered = self.raw.copy()  # if filtered data is an empty list, then instead just copy the raw data
            self.indices_removed =  [] # empty list if no indices yet removed

    def filter_data(self, indices_to_remove):
        """ Filters out points (from user-specified indices) from the raw specimen data and returns a filtered version """
        self.indices_removed = indices_to_remove
        self.filtered = self.raw.drop(indices_to_remove)    # Note: we don't reset the indices because we want to keep track of the removed records 
        self.manager.data_dict[self.manager.selected_specimen_name]['filtered'] = self.filtered  # make sure we update the data_dict in case they aren't linked

    def update_raw_specimen(self, updated_raw_specimen):
        """ Updates raw specimen with new dataframe """
        self.raw = updated_raw_specimen

    def get_raw_specimen(self):
        """ Returns the raw specimen """
        return self.raw
    
    def get_filtered_specimen(self):
        """ Returns the filtered specimen """
        return self.filtered


######### Coordinate System class ############ 

class CoordinateSystem:
    """ Maintains the current coordinate system and handles changes to it """
    def __init__(self, coordinates='specimen', manager=None):
        self.coordinates = coordinates  # default coordinates
        self.manager = manager

    def change_coordinates(self, new_coordinates, specimen):
        """ Change the current coordinate system and transform the specimen data accordingly """
        self.coordinates = new_coordinates
        transformed_specimen = pro.set_coordinates(new_coordinates, specimen.get_raw_specimen())  # rotate the raw specimen data
        specimen.update_raw_specimen(transformed_specimen)  # replace the raw specimen data with the transformed data
        specimen.filter_data(specimen.indices_removed)      # reapply any filtering to get the transformed filtered data

        with self.manager.output_area:       # update the displayed data
            clear_output(wait=True)
            #display(self.manager.data_dict[self.manager.selected_specimen_name]['raw']) 

            display(HTML(f"""<div style="max-height: 400px; overflow-y: auto; border: 1px solid #ccc;">
                        {self.manager.data_dict[self.manager.selected_specimen_name]['raw'].to_html()}</div>"""))
        
    def get_coordinates(self):
        """ Returns the current coordinates """
        return self.coordinates


######### AxesProjection class ############ 

class AxesProjection:
    """ Maintains the current axes projection and handles changes to it """
    def __init__(self, projection=1):
        self.projection = projection  # default projection

    def change_projection(self, new_projection):
        """ Change the current projection """
        self.projection = new_projection

    def get_projection(self):
        """ Returns the current projection """
        return self.projection
    
    
######### Plotter class ############ 
        
class Plotter:
    """ Class to handle plotting methods """
    def update_zij_plot(self, coordinates, projection, specimen, lines=[], planes=[], plot_output_area=None):
        """ Update the Zijderveld plot """            
        with plot_output_area:
            clear_output(wait=True)
            plo.zij_plt(coordinates, projection, specimen.get_raw_specimen(), specimen.get_filtered_specimen(), lines, planes)

    def update_linmod_plot(self, coordinates, projection, specimen, lines, fitted_points, coefficients, coefficients_norm, plot_output_area=None):
        """ Update the Linear Model plot """
        with plot_output_area:
            clear_output(wait=True)
            plo.linzij_plt(coordinates, projection, specimen.get_raw_specimen(), specimen.get_filtered_specimen(), lines, fitted_points, coefficients, coefficients_norm)


######### Interface class ############ 
            
class Interface:
    def __init__(self, specimen=None, manager=None):
        self.manager = manager
        self.specimen = specimen
        self.lines = self.manager.data_dict[self.manager.selected_specimen_name]['lines']
        self.planes = self.manager.data_dict[self.manager.selected_specimen_name]['planes']
        self.fitted_points = None
        self.coefficients = []
        self.coefficients_norm = []

        if self.lines != [] or self.planes != []:  # check if there already exists fitted data
            self.fits_applied = True
        else: self.fits_applied = False
        self.linmod_applied = False

        # initialize the coordinate system, projection and plotter classes
        self.axes_projection = AxesProjection(projection=1)
        self.coordinate_system = CoordinateSystem(coordinates='specimen', manager=self.manager)
        self.plotter = Plotter()

        # ensure that specimen is in the right coordinates
        if self.specimen.raw.get('coordinates') is not None:
            initial_coordinates = self.specimen.raw['coordinates'].iloc[0]
        else: initial_coordinates = 'specimen'
        self.coordinate_system.change_coordinates(initial_coordinates, self.specimen)

        # setup widgets
        self.sample_selection_widgets()
        self.save_data_widgets()
        self.coordinate_projection_widgets()
        self.filtering_widgets()
        self.fitting_widgets()
        self.pca_and_linmod_widgets()

        # setup display and layout
        self.message_output_area1 = Output()
        self.plot_output_area1 = Output()
        self.message_output_area2 = Output()
        self.plot_output_area2 = Output()

        display(widgets.HBox([self.hide_data_button, self.apply_filter_button, self.projection_dropdown1, self.coordinates_dropdown1,    # upper panel, first row
                self.left_arrow_button, self.right_arrow_button, self.save_data_button], layout=widgets.Layout(justify_content='flex-start', align_items='center')),         
                self.checkboxes_container, 
                widgets.HBox([self.hide_lines_button, self.add_line_button, self.apply_fits_button, self.clear_fits_button]), self.fit_lines_container, # second row
                widgets.HBox([self.hide_planes_button, self.add_plane_button]), self.fit_planes_container,   # third row
                self.message_output_area1, self.plot_output_area1, 
                widgets.HBox([self.toggle_PCA_button, self.autoPCA_button, self.PCA_weight_text_box, self.PCA_penalty_text_box]),  # lower panel, first row
                self.message_output_area2,
                widgets.HBox([self.toggle_linmod_button, self.linmod_button, self.projection_dropdown2, self.coordinates_dropdown2]), # second row
                self.plot_output_area2)


        # refresh the Zijderveld
        self.plotter.update_zij_plot(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, plot_output_area=self.plot_output_area1)
    
    
    #### Widgets setup ####
    
    def sample_selection_widgets(self):
        """ Widgets controlling sample selection / cycling """
        self.left_arrow_button = widgets.Button(description='◀', layout=widgets.Layout(width="50px"))
        self.left_arrow_button.on_click(self.on_left_arrow_click)    
        
        self.right_arrow_button = widgets.Button(description='▶', layout=widgets.Layout(width="50px"))
        self.right_arrow_button.on_click(self.on_right_arrow_click)

    def save_data_widgets(self):
        """ Widgets enabling data output """
        self.save_data_button = widgets.Button(description="Save Data", layout=widgets.Layout(width="150px"))
        self.save_data_button.on_click(self.manager.save_data)

    def coordinate_projection_widgets(self):
        """ Widgets controlling coordinate and projection settings """
        self.projection_dropdown1 = widgets.Dropdown(options=[1, 2], value=self.axes_projection.projection, description='Projection:',
                                                layout=widgets.Layout(width="250px", margin="0px 50px 0px 50px"))
        self.projection_dropdown1.observe(self.update_projections, names='value')
        
        self.coordinates_dropdown1 = widgets.Dropdown(options=['specimen', 'geographic', 'tectonic'], value=self.coordinate_system.coordinates, description='Coordinates:',
                                                    layout=widgets.Layout(width="250px", margin="0px 50px 0px 50px"))
        self.coordinates_dropdown1.observe(self.update_coordinates, names='value')

        # this second set of dropdown menus corresponds to the lower plot (but their functionality is bound to the first set)
        self.projection_dropdown2 = widgets.Dropdown(options=[1, 2], value=self.axes_projection.projection, description='Projection:',
                                                layout=widgets.Layout(width="250px", margin="0px 50px 0px 50px"))
        self.projection_dropdown2.observe(self.update_projections, names='value')

        self.coordinates_dropdown2 = widgets.Dropdown(options=['specimen', 'geographic', 'tectonic'], value=self.coordinate_system.coordinates, description='Coordinates:',
                                                    layout=widgets.Layout(width="250px", margin="0px 50px 0px 50px"))
        self.coordinates_dropdown2.observe(self.update_coordinates, names='value')
        
    def filtering_widgets(self):
        """ Widgets enabling data filtering """    
        self.hide_data_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))  # toggle the data filter panel
        self.hide_data_button.on_click(self.toggle_data_checkboxes)

        self.apply_filter_button = widgets.Button(description="filter selected", layout=widgets.Layout(width="150px", margin="0px, 80px, 0px, 0px"))
        self.apply_filter_button.on_click(self.filter_data)

        # Checkbox container (these are the tickboxes enabling points to be selected for removal)
        self.checkbox_states = {}
        self.checkboxes_container = widgets.HBox([])

    def fitting_widgets(self):
        """ Widgets enabling line and plane fitting to the data """  
        
        ### line fitting widgets ###
        self.hide_lines_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))  # toggle the fits panel
        self.hide_lines_button.on_click(self.toggle_hide_lines_widgets)

        self.add_line_button = widgets.Button(description="+ line", layout=widgets.Layout(width="80px"))
        self.add_line_button.on_click(self.add_fit_line_row)

        self.apply_fits_button = widgets.Button(description="apply fits", layout=widgets.Layout(width="150px"))
        self.apply_fits_button.on_click(self.apply_fits)

        self.clear_fits_button = widgets.Button(description="clear fits", layout=widgets.Layout(width="150px"))
        self.clear_fits_button.on_click(self.clear_fits)

        self.fit_line_widgets = []
        self.fit_lines_container = widgets.VBox([])
        self.fit_lines_container.layout.display = 'none'

        for line in self.lines:                      # add any lines that have already been fit
            self.add_fit_line_row(prefill=line)

        ### plane fitting widgets ###
        self.hide_planes_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
        self.hide_planes_button.on_click(self.toggle_hide_planes_widgets)

        self.add_plane_button = widgets.Button(description="+ plane", layout=widgets.Layout(width="80px"))
        self.add_plane_button.on_click(self.add_fit_plane_row)

        self.fit_plane_widgets = []
        self.fit_planes_container = widgets.VBox([])
        self.fit_planes_container.layout.display = 'none'

        for plane in self.planes:                     # add any planes that have already been fit
            self.add_fit_plane_row(prefill=plane)

    def pca_and_linmod_widgets(self):
        """ Widgets controlling auto-PCA and linear modelling of data """
        self.toggle_PCA_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))  # toggle the modelling panel
        self.toggle_PCA_button.on_click(self.toggle_PCA_display)
        
        self.autoPCA_button = widgets.Button(description="auto-PCA", layout=widgets.Layout(width="150px"))
        self.autoPCA_button.on_click(self.do_autoPCA)
    
        self.PCA_weight_text_box = widgets.Text(description='weight:', placeholder='0.5', layout=widgets.Layout(width="150px"))
        self.PCA_penalty_text_box = widgets.Text(description='penalty:', placeholder='0.5', layout=widgets.Layout(width="150px"))

        self.toggle_linmod_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
        self.toggle_linmod_button.on_click(self.toggle_linmod_display)

        self.linmod_button = widgets.Button(description="linear model", layout=widgets.Layout(width="150px"))
        self.linmod_button.on_click(self.run_linear_model)


    
    #### Widgets methods (approx. in order of widget appearance on page) ####

    def toggle_data_checkboxes(self, b=None):
        """ Show/hide checkboxes while preserving selections """
        if self.checkboxes_container.children:  # if panel is open...
            self.checkbox_states = {int(cb.description): cb.value for cb in self.checkboxes_container.children} # save current checkbox states...
            self.checkboxes_container.children = []  # and hide panel
        else:
            self.checkboxes_container.children = [widgets.Checkbox(value=(i in self.specimen.indices_removed), description=str(i), indent=False) # restore checkboxes
                                                  for i in range(len(self.specimen.raw))]
    
    def filter_data(self, b=None):
        """ Filter the data based on user selection (e.g. checkboxes) """
        if self.checkboxes_container.children:
            selected_indices = [int(cb.description) for cb in self.checkboxes_container.children if cb.value] # if checkboxes are visible, get selected indices from them
        else:
            selected_indices = [i for i, v in self.checkbox_states.items() if v]  # if checkboxes are hidden, use the last saved states
        
        self.specimen.filter_data(selected_indices)  # call Specimen's method
        updated_range = self.specimen.get_filtered_specimen().index.tolist()
        
        for widgets_set in self.fit_line_widgets + self.fit_plane_widgets:  # update the fitting dropdown selection (so removed indices don't appear there)
            widgets_set[0].options = updated_range
            widgets_set[1].options = updated_range

        # update Zijderveld
        self.plotter.update_zij_plot(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, plot_output_area=self.plot_output_area1)
    
    def update_projections(self, change):
        """ Update the axes projection based on user selection """
        new_projection = change['new']
        self.projection_dropdown1.value = new_projection
        self.projection_dropdown2.value = new_projection       
        self.axes_projection.change_projection(new_projection)

        if self.fits_applied: self.apply_fits()   # if fits have been applied, re-compute them; otherwise update Zijderveld with just the data
        else:
            self.plotter.update_zij_plot(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, plot_output_area=self.plot_output_area1)
            
        if self.linmod_applied:                   # update the lower panels too if need be...
            self.plotter.update_linmod_plot(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, 
                                            self.lines, self.fitted_points, self.coefficients, self.coefficients_norm, plot_output_area=self.plot_output_area2)
    
    def update_coordinates(self, change):
        """ Update the coordinate system based on user selection """
        new_coordinates = change['new']
        self.coordinates_dropdown1.value = new_coordinates
        self.coordinates_dropdown2.value = new_coordinates  
        self.coordinate_system.change_coordinates(new_coordinates, self.specimen)
        
        if self.fits_applied: self.apply_fits()   # if fits have been applied, re-compute them; otherwise update Zijderveld with just the data
        else:
            self.plotter.update_zij_plot(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, plot_output_area=self.plot_output_area1)

        if self.linmod_applied: self.run_linear_model()  # update the lower panels too if need be...

    def on_left_arrow_click(self, b=None):
        """  Move backwards through specimen list """
        current_index = self.manager.specimen_names.index(self.manager.selected_specimen_name)
        new_index = (current_index - 1) % len(self.manager.specimen_names)  # wrap around to last if we reach the first
        self.manager.specimen_selector.value = self.manager.specimen_names[new_index]  # update dropdown value
    
    def on_right_arrow_click(self, b=None):
        """  Move forwards through specimen list """
        current_index = self.manager.specimen_names.index(self.manager.selected_specimen_name)
        new_index = (current_index + 1) % len(self.manager.specimen_names)  # wrap around to first if we reach the last
        self.manager.specimen_selector.value = self.manager.specimen_names[new_index] # update dropdown value

    def toggle_hide_lines_widgets(self, b=None):
        """ Show/hide the fit lines widgets """
        if self.fit_lines_container.layout.display == 'none': self.fit_lines_container.layout.display = ''
        else: self.fit_lines_container.layout.display = 'none'

    def toggle_hide_planes_widgets(self, b=None):
        """ Show/hide the fit planes widgets."""
        if self.fit_planes_container.layout.display == 'none': self.fit_planes_container.layout.display = ''
        else: self.fit_planes_container.layout.display = 'none'

    def apply_fits(self, b=None):
        """ Apply line / plane fits to data """

        self.lines, self.planes = [], []   # clear the line and plane fits
        filtered = self.specimen.get_filtered_specimen()         # get the current filtered specimen
        coordinates = self.coordinate_system.get_coordinates()   # get the current coordinates

        with self.message_output_area1:
            self.message_output_area1.clear_output(wait=True)
            
            for line_widgets in self.fit_line_widgets:
                from_idx, to_idx, incl_origin, anchor, line_name, line_color, widget_row = line_widgets
                
                # access values from each widget in the row
                lfrom = from_idx.value
                lto = to_idx.value
                lorigin = incl_origin.value
                lanchor = anchor.value
                lname = line_name.value
                lcolor = line_color.value
    
                if lfrom is None or lto is None or lfrom >= lto:
                    print(f"Invalid index selection for {lname}, skipping...")
                    continue
    
                lfit = filtered.loc[lfrom:lto]                                                                             # get the subset of data points that are being fitted
                v1, mad, v1_segment = pro.linefit(np.column_stack((lfit['x1'], lfit['x2'], lfit['x3'])), lorigin, lanchor) # do PCA
                di = pro.to_sph([v1])                                                                                      # convert to spherical coordinates
                print(f'Comp. {lname} (n={len(lfit)}), Dec: {di[0][0]:.2f}, Inc: {di[0][1]:.2f}, MAD: {mad:.2f}')          # print summary of results
                self.lines.append(['line', lname, lfit, lorigin, lanchor, v1, mad, v1_segment, coordinates, lcolor])       # append the key results to lines

            for plane_widgets in self.fit_plane_widgets:
                from_idx, to_idx, normalize, constraints, plane_name, plane_color, widget_row = plane_widgets
                
                # Access values from each widget in the row
                pfrom = from_idx.value
                pto = to_idx.value
                pnormalize = normalize.value
                pconstraints = constraints.value
                pname = plane_name.value
                pcolor = plane_color.value
    
                if pfrom is None or pto is None or pfrom >= pto:
                    print(f"Invalid index selection for {pname}, skipping...")
                    continue
    
                pfit = filtered.loc[pfrom:pto]                                                                               # get the subset of data points that are being fitted
                v3, mad, gc_segment = pro.gcfit(np.column_stack((pfit['x1'], pfit['x2'], pfit['x3'])), pnormalize, pconstraints)  # do PCA
                di = pro.to_sph([v3])                                                                                        # convert to spherical coordinates
                print (f'GC. {pname} (n={len(pfit)}), Dec: {di[0][0]:.2f}, Inc: {di[0][1]:.2f}, MAD: {mad:.2f}')             # print summary
                self.planes.append(['plane', pname, pfit, pnormalize, pconstraints, v3, mad, gc_segment, coordinates, pcolor]) # append key results

        self.manager.data_dict[self.manager.selected_specimen_name]['lines'] = self.lines     # update the data_dict
        self.manager.data_dict[self.manager.selected_specimen_name]['planes'] = self.planes

        self.fits_applied = True
        self.plotter.update_zij_plot(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, self.plot_output_area1)

    def clear_fits(self, b=None):
        """ Clear existing fits """
        with self.message_output_area1:
            self.message_output_area1.clear_output(wait=True)
            print ('')
        
        self.lines, self.planes, self.coefficients, self.coefficients_norm = [], [], [], []
        self.manager.data_dict[self.manager.selected_specimen_name]['lines'] = self.lines
        self.manager.data_dict[self.manager.selected_specimen_name]['planes'] = self.planes
        self.manager.data_dict[self.manager.selected_specimen_name]['coefficients_norm'] = self.coefficients_norm

        if self.fits_applied: self.plotter.update_zij_plot(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, 
                                                           self.lines, self.planes, plot_output_area=self.plot_output_area1)
        with self.plot_output_area2: self.plot_output_area2.clear_output()
    
    def add_fit_line_row(self, b=None, prefill=None):
        """ Add a new row for a new line fit """
        from_dropdown = widgets.Dropdown(options=list(self.specimen.get_filtered_specimen().index), description='From:', layout=widgets.Layout(width="150px"))
        to_dropdown = widgets.Dropdown(options=list(self.specimen.get_filtered_specimen().index), description='To:', layout=widgets.Layout(width="150px"))
        include_origin_checkbox = widgets.Checkbox(value=False, description='Include Origin', indent=False, layout=widgets.Layout(width="auto"), margin='20px')
        anchor_checkbox = widgets.Checkbox(value=False, description='Anchor', indent=False, layout=widgets.Layout(width="auto"), margin='20px')
        line_name_text = widgets.Text(description='Component:', layout=widgets.Layout(width="200px"))
        line_color_dropdown = widgets.ColorPicker(value='green', description='Color:', layout=widgets.Layout(width="200px"))
        remove_line_button = widgets.Button(description="X", layout=widgets.Layout(width="30px"))
        remove_line_button.on_click(lambda x: self.remove_fit_line_row(widget_row))

        if prefill:    # reintroduce the settings of any prior fit
            from_dropdown.value = prefill[2].index[0]
            to_dropdown.value = prefill[2].index[-1]
            include_origin_checkbox.value = prefill[3]
            anchor_checkbox.value = prefill[4]
            line_name_text.value = prefill[1]
            line_color_dropdown.value = prefill[9]

        space = widgets.Label("", layout=widgets.Layout(width="60px"))

        widget_row = widgets.HBox([from_dropdown, to_dropdown, space, include_origin_checkbox, space, anchor_checkbox, space, line_name_text, line_color_dropdown, remove_line_button],
                                 layout=widgets.Layout(width='100%', justify_content='flex-start'))
        
        self.fit_line_widgets.append((from_dropdown, to_dropdown, include_origin_checkbox, anchor_checkbox, line_name_text, line_color_dropdown, widget_row))
        self.fit_lines_container.children += (widget_row,)

    def remove_fit_line_row(self, widget_row):
        """ Remove a row of fit-line widgets """
        self.fit_line_widgets = [w for w in self.fit_line_widgets if w[-1] != widget_row]
        self.fit_lines_container.children = [w[-1] for w in self.fit_line_widgets]

    def add_fit_plane_row(self, b=None, prefill=None):
        """ Add a new row for a new plane fit """
        from_dropdown = widgets.Dropdown(options=list(self.specimen.get_filtered_specimen().index), description='From:', layout=widgets.Layout(width="150px"))
        to_dropdown = widgets.Dropdown(options=list(self.specimen.get_filtered_specimen().index), description='To:', layout=widgets.Layout(width="150px"))       
        normalize_checkbox = widgets.Checkbox(value=False, description='Normalize', indent=False, layout=widgets.Layout(width="auto"), margin='20px')
        constraints_checkbox = widgets.Checkbox(value=False, description='Constraints', indent=False, layout=widgets.Layout(width="auto"), margin='20px')
        plane_name_text = widgets.Text(description='Component:', layout=widgets.Layout(width="200px"))
        plane_color_dropdown = widgets.ColorPicker(value='green', description='Color:', layout=widgets.Layout(width="200px"))
        remove_plane_button = widgets.Button(description="X", layout=widgets.Layout(width="30px"))
        remove_plane_button.on_click(lambda x: self.remove_fit_plane_row(widget_row))

        if prefill:  # reintroduce the settings of any prior fit
            from_dropdown.value = prefill[2].index[0]
            to_dropdown.value = prefill[2].index[-1]
            normalize_checkbox.value = prefill[3]
            constraints_checkbox.value = prefill[4]
            plane_name_text.value = prefill[1]
            plane_color_dropdown.value = prefill[9]

        space = widgets.Label("", layout=widgets.Layout(width="60px"))

        widget_row = widgets.HBox([from_dropdown, to_dropdown, space, normalize_checkbox, space, constraints_checkbox, space, plane_name_text, plane_color_dropdown, remove_plane_button],
                                 layout=widgets.Layout(width='100%', justify_content='flex-start'))
        
        self.fit_plane_widgets.append((from_dropdown, to_dropdown, normalize_checkbox, constraints_checkbox, plane_name_text, plane_color_dropdown, widget_row))
        self.fit_planes_container.children += (widget_row,)

    def remove_fit_plane_row(self, widget_row):
        """ Remove a row of fit-line widgets """
        self.fit_plane_widgets = [w for w in self.fit_plane_widgets if w[-1] != widget_row]
        self.fit_planes_container.children = [w[-1] for w in self.fit_plane_widgets]

    def toggle_PCA_display(self, b=None):
        """ Toggle the visibility of PCA display """
        self.message_output_area2.layout.display = '' if self.message_output_area2.layout.display == 'none' else 'none'

    def do_autoPCA(self, weight=0.5, penalty=0.5):
        """ Execute autoPCA routine """
        weight = float(self.PCA_weight_text_box.value) if self.PCA_weight_text_box.value.strip() else 0.5    # take in user-specified weight if present
        penalty = float(self.PCA_penalty_text_box.value) if self.PCA_penalty_text_box.value.strip() else 0.5 # take in user-specified penalty if present

        with self.message_output_area2:
            self.message_output_area2.clear_output(wait=True)
            pro.cycle_autoPCA(self.specimen.get_filtered_specimen(), weight, penalty)

    def toggle_linmod_display(self, b=None):
        """ Toggle the visibility of linear model display """
        self.plot_output_area2.layout.display = '' if self.plot_output_area2.layout.display == 'none' else 'none'

    def run_linear_model(self, b=None):
        """ Execute linear modelling routine """
        fspec = self.specimen.get_filtered_specimen()
        lines = self.get_lines()
        self.fitted_points, self.coefficients, self.coefficients_norm = pro.dirmod(np.column_stack((fspec['x1'], fspec['x2'], fspec['x3'])), [x[5] for x in lines])
        
        self.plotter.update_linmod_plot(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, 
                                        self.fitted_points, self.coefficients, self.coefficients_norm, plot_output_area=self.plot_output_area2)

        self.manager.data_dict[self.manager.selected_specimen_name]['coefficients_norm'] = self.coefficients_norm
        self.linmod_applied = True

    def get_lines(self):
        """ Return current fitted lines """
        return self.lines

    def get_planes(self):
        """ Return current fitted planes """
        return self.planes

