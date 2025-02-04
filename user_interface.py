import ipywidgets as widgets
from ipywidgets import Output
from IPython.display import display, clear_output
import numpy as np
import pickle
import processing as pro
import plotting as plo

######### Manager class ############ 
class Manager:
    def __init__(self, infile):
        """
        initialize a Manager class with dataframe containing multiple specimens
        """
        self.infile = infile
        self.data_dict = {}
        self.df = None
        self.specimen_names = None
        self.interface = None
        self.output_area = None

        self.get_data()
        self.selected_specimen_name = self.specimen_names[0]
        if len(self.specimen_names) > 0:
            self.specimen_selector = widgets.Dropdown(options=self.specimen_names, value=self.selected_specimen_name, description='specimen:')
        else:
            print ('No specimen names available.')
        self.specimen_selector.observe(self.on_specimen_change, names='value')

        self.save_text_box = widgets.Text(description='save to:', placeholder='file prefix') 

        self.controls = widgets.HBox([self.specimen_selector, self.save_text_box])
        self.output_area = Output()
        display(self.controls, self.output_area)

    def get_data(self):
        
        if self.infile.endswith('.pkl'):
            with open(self.infile, 'rb') as f:
                self.data_dict = pickle.load(f)
                self.specimen_names = list(self.data_dict.keys())
        
        elif self.infile.endswith('.jr6'):
            self.df = pro.import_jr6(self.infile)
            self.init_dict()

        else: print("Unsupported file type.")
  
    def init_dict(self):
        self.specimen_names = self.df['specimen'].unique()
        for specimen in self.specimen_names:
            specimen_raw_df = self.df[self.df['specimen'] == specimen].reset_index(drop=True)
            self.data_dict[specimen] = {
                'raw': specimen_raw_df,
                'filtered': [],
                'lines': [],
                'planes': [],
                'coefficients': []}
        
    def on_specimen_change(self, change):
        """
        callback function that updates the selected specimen when a value is selected in the dropdown
        """
        if change['name'] == 'value':
            self.selected_specimen_name = self.specimen_selector.value

            # retrieve existing data where available
            filtered_data = self.data_dict[self.selected_specimen_name]['filtered']
            lines = self.data_dict[self.selected_specimen_name]['lines']
            planes = self.data_dict[self.selected_specimen_name]['planes']
            
            with self.output_area:
                clear_output(wait=True)
                display(self.data_dict[self.selected_specimen_name]['raw'])

                # initialize the Specimen and Interface classes with the subset of the dataframe
                specimen = Specimen(self.data_dict[self.selected_specimen_name]['raw'], manager=self)
                specimen.filtered = filtered_data

                self.interface = Interface(specimen, lines, planes, manager=self)

    def get_selected_specimen_name(self):
        """
        returns the selected specimen name after user interacts with menu
        """
        return self.selected_specimen_name

    def save_data(self, b=None):

        if self.save_text_box:
            filename = self.save_text_box.value + '.pkl'
        else:
            filename = 'data_dict.pkl'
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.data_dict, f)
                print (f'Data saved to {filename}')
        except Exception as e:
            print (f'Error saving data: {e}')


######### Specimen class ############ 
class Specimen:
    def __init__(self, df, manager=None):
        self.raw = df  # original data
        self.manager = manager
        self.filtered = df.copy()  # filtered data
        self.indices_removed = []

    def filter_data(self, indices_to_remove):
        """Filters out points from the original specimen based on the provided indices."""
        self.indices_removed = indices_to_remove
        self.filtered = self.raw.drop(indices_to_remove)    #.reset_index(drop=True)
        self.manager.data_dict[self.manager.selected_specimen_name]['filtered'] = self.filtered

    def update_raw_specimen(self, updated_raw_specimen):
        """updates raw specimen with new dataframe"""
        self.raw = updated_raw_specimen

    def get_raw_specimen(self):
        """Returns the raw specimen."""
        return self.raw
    
    def get_filtered_specimen(self):
        """Returns the filtered specimen."""
        return self.filtered


######### Coordinate System class ############ 
class CoordinateSystem:
    def __init__(self, coordinates='specimen'):
        self.coordinates = coordinates  # default coordinates

    def change_coordinates(self, new_coordinates, specimen):
        """change the current coordinate system"""
        self.coordinates = new_coordinates
        transformed_specimen = pro.set_coordinates(new_coordinates, specimen.get_raw_specimen())
        specimen.update_raw_specimen(transformed_specimen)
        specimen.filter_data(specimen.indices_removed)  # reapply the previously removed indices

    def get_coordinates(self):
        return self.coordinates


######### AxesProjection class ############ 
class AxesProjection:
    def __init__(self, projection=1):
        self.projection = projection  # default projection

    def change_projection(self, new_projection):
        """change the current projection system"""
        self.projection = new_projection

    def get_projection(self):
        return self.projection
    
    
######### Plotter class ############ 
class Plotter:
    def update_plot1(self, coordinates, projection, specimen, lines=None, planes=None, plot_output_area=None):
        """Update the plot with the current data, fitted models, and coordinate system."""
        if lines is None: lines = []
        if planes is None: planes = []
        if plot_output_area is None:
            raise ValueError("plot output area must be provided")  # Ensure it's not missing
            
        with plot_output_area:
            clear_output(wait=True)
            plo.zij_plt(coordinates, projection, specimen.get_raw_specimen(), specimen.get_filtered_specimen(), lines, planes)

    def update_plot2(self, coordinates, projection, raw_specimen, filtered_specimen, lines, fitted_points, coefficients, plot_output_area=None):
        """Update the plot with the current data, fitted models, and coordinate system."""
        if plot_output_area is None:
            raise ValueError("plot output area must be provided")  # Ensure it's not missing
            
        with plot_output_area:
            clear_output(wait=True)
            plo.linzij_plt(coordinates, projection, raw_specimen, filtered_specimen, lines, fitted_points, coefficients)

######### Interface class ############ 
class Interface:
    def __init__(self, specimen, lines=[], planes=[], coefficients=[], manager=None):
        self.specimen = specimen
        self.lines = lines
        self.planes = planes
        self.coefficients = coefficients
        self.fitted_points = None
        self.manager = manager
        self.plot_output_area1 = Output()
        self.message_output_area1 = Output()
        self.plot_output_area2 = Output()
        self.message_output_area2 = Output()
        self.fits_applied = False
        self.linmod_applied = False
        
        if self.lines != [] or self.planes != []:
            self.fits_applied = True

        #auto-initialize the coordinate system and plotter classes
        self.axes_projection = AxesProjection(projection=1)
        self.coordinate_system = CoordinateSystem(coordinates='specimen')
        self.plotter = Plotter()

        # Widgets for user interaction       
        self.filter_data_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
        self.filter_data_button.on_click(self.toggle_data_checkboxes)

        self.apply_filter_button = widgets.Button(description="filter selected", layout=widgets.Layout(width="150px", margin="0px, 80px, 0px, 0px"))
        self.apply_filter_button.on_click(self.filter_data)

        # Checkbox container
        self.checkbox_states = {}
        self.checkboxes_container = widgets.HBox([])

        self.projection_dropdown1 = widgets.Dropdown(options=[1, 2], value=self.axes_projection.projection, description='Projection:',
                                                    layout=widgets.Layout(width="250px", margin="0px 50px 0px 50px"))
        self.projection_dropdown1.observe(self.update_projections, names='value')
        
        self.coordinates_dropdown1 = widgets.Dropdown(options=['specimen', 'geographic', 'tectonic'], value=self.coordinate_system.coordinates, description='Coordinates:',
                                                    layout=widgets.Layout(width="250px", margin="0px 50px 0px 50px"))
        self.coordinates_dropdown1.observe(self.update_coordinates, names='value')

        self.left_arrow_button = widgets.Button(description='◀', layout=widgets.Layout(width="50px"))
        self.left_arrow_button.on_click(self.on_left_arrow_click)    
        self.right_arrow_button = widgets.Button(description='▶', layout=widgets.Layout(width="50px"))
        self.right_arrow_button.on_click(self.on_right_arrow_click)

        self.save_data_button = widgets.Button(description="Save Data", layout=widgets.Layout(width="150px"))
        self.save_data_button.on_click(self.save_data)


        self.hide_lines_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
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


        self.hide_planes_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
        self.hide_planes_button.on_click(self.toggle_hide_planes_widgets)

        self.add_plane_button = widgets.Button(description="+ plane", layout=widgets.Layout(width="80px"))
        self.add_plane_button.on_click(self.add_fit_plane_row)

        self.fit_plane_widgets = []

        self.fit_planes_container = widgets.VBox([])
        self.fit_planes_container.layout.display = 'none'

        self.toggle_PCA_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
        self.toggle_PCA_button.on_click(self.toggle_PCA_display)
        
        self.autoPCA_button = widgets.Button(description="auto-PCA", layout=widgets.Layout(width="150px"))
        self.autoPCA_button.on_click(self.do_autoPCA)
    
        self.PCA_weight_text_box = widgets.Text(description='weight:', placeholder='0.5', layout=widgets.Layout(width="150px"))
        self.PCA_penalty_text_box = widgets.Text(description='penalty:', placeholder='0.5', layout=widgets.Layout(width="150px"))

        self.toggle_linmod_button = widgets.Button(description="toggle", layout=widgets.Layout(width="80px"))
        self.toggle_linmod_button.on_click(self.toggle_linmod_display)

        self.linmod_button = widgets.Button(description="linear model", layout=widgets.Layout(width="150px"))
        self.linmod_button.on_click(self.run_linear_model)

        self.projection_dropdown2 = widgets.Dropdown(options=[1, 2], value=self.axes_projection.projection, description='Projection:',
                                                    layout=widgets.Layout(width="250px", margin="0px 50px 0px 50px"))
        self.projection_dropdown2.observe(self.update_projections, names='value')

        self.coordinates_dropdown2 = widgets.Dropdown(options=['specimen', 'geographic', 'tectonic'], value=self.coordinate_system.coordinates, description='Coordinates:',
                                                    layout=widgets.Layout(width="250px", margin="0px 50px 0px 50px"))
        self.coordinates_dropdown2.observe(self.update_coordinates, names='value')
    
        
        # Layout
        self.controls = widgets.HBox([self.filter_data_button, self.apply_filter_button, self.projection_dropdown1, self.coordinates_dropdown1, 
                                      self.left_arrow_button, self.right_arrow_button, self.save_data_button], layout=widgets.Layout(justify_content='flex-start', align_items='center'))
        display(self.controls, 
                self.checkboxes_container, 
                widgets.HBox([self.hide_lines_button, self.add_line_button, self.apply_fits_button, self.clear_fits_button]), self.fit_lines_container,
                widgets.HBox([self.hide_planes_button, self.add_plane_button]), self.fit_planes_container,
                self.message_output_area1, self.plot_output_area1,
                widgets.HBox([self.toggle_PCA_button, self.autoPCA_button, self.PCA_weight_text_box, self.PCA_penalty_text_box]),
                self.message_output_area2,
                widgets.HBox([self.toggle_linmod_button, self.linmod_button, self.projection_dropdown2, self.coordinates_dropdown2]),
                self.plot_output_area2)
        
        for line in self.lines:
            self.add_fit_line_row(prefill=line)

        for plane in self.planes:
            self.add_fit_plane_row(prefill=plane)

        self.coordinate_system.change_coordinates(self.coordinate_system.coordinates, self.specimen)
        self.plotter.update_plot1(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, plot_output_area=self.plot_output_area1)

    def toggle_data_checkboxes(self, b):
        """Show/hide checkboxes while preserving selections."""
        if self.checkboxes_container.children:
            # save current checkbox states and hide checkboxes
            self.checkbox_states = {int(cb.description): cb.value for cb in self.checkboxes_container.children}
            self.checkboxes_container.children = []
        else:
            # Restore checkboxes with previous states
            self.checkboxes_container.children = [widgets.Checkbox(value=self.checkbox_states.get(i, False), description=str(i), indent=False,) 
                                                  for i in range(len(self.specimen.raw))]
    
    def filter_data(self, b):
        """filter the data based on user selection (e.g., using checkboxes)"""
        # Add logic to allow user to filter data (e.g., via checkboxes or range selection)
        selected_indices = [int(cb.description) for cb in self.checkboxes_container.children if cb.value]
        self.specimen.filter_data(selected_indices)  # Call Specimen's method

        updated_range = self.specimen.get_filtered_specimen().index.tolist()

        for widgets_set in self.fit_line_widgets + self.fit_plane_widgets:
            widgets_set[0].options = updated_range
            widgets_set[1].options = updated_range
        
        self.plotter.update_plot1(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, plot_output_area=self.plot_output_area1)  # Refresh plot
    
    def update_projections(self, change):
        """update the axes projection based on user selection."""
        new_projection = change['new']
        self.projection_dropdown1.value = new_projection
        self.projection_dropdown2.value = new_projection       
        self.axes_projection.change_projection(new_projection)

        if self.fits_applied:
            self.apply_fits(None)
        else:
            self.plotter.update_plot1(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, plot_output_area=self.plot_output_area1)
            
        if self.linmod_applied:
            self.plotter.update_plot2(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen.get_raw_specimen(), 
                                  self.specimen.get_filtered_specimen(), self.lines, self.fitted_points, self.coefficients, plot_output_area=self.plot_output_area2)
    
    def update_coordinates(self, change):
        """update the coordinate system based on user selection."""
        new_coordinates = change['new']
        self.coordinates_dropdown1.value = new_coordinates
        self.coordinates_dropdown2.value = new_coordinates  
        self.coordinate_system.change_coordinates(new_coordinates, self.specimen)
        
        if self.fits_applied:
            self.apply_fits(None)
        else:
            self.plotter.update_plot1(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, plot_output_area=self.plot_output_area1)

        if self.linmod_applied: 
            self.run_linear_model()


    def on_left_arrow_click(self, b):
        """Callback function when left arrow is clicked"""
        current_index = self.manager.specimen_names.index(self.manager.selected_specimen_name)
        new_index = (current_index - 1) % len(self.manager.specimen_names)  # wrap around to last if we reach the first
        self.manager.specimen_selector.value = self.manager.specimen_names[new_index]  # update dropdown value
    
    def on_right_arrow_click(self, b):
        """Callback function when right arrow is clicked"""
        current_index = self.manager.specimen_names.index(self.manager.selected_specimen_name)
        new_index = (current_index + 1) % len(self.manager.specimen_names)  # Wrap around to first if we reach the last
        self.manager.specimen_selector.value = self.manager.specimen_names[new_index]

    def save_data(self, filename):
        filename = 'data_dict.json'
        self.manager.save_data(filename)
        with self.message_output_area1:
            print(f'Data saved to {filename}')

    def toggle_hide_lines_widgets(self, b):
        """Show/hide the Fit Lines widgets."""
        if self.fit_lines_container.layout.display == 'none':
            self.fit_lines_container.layout.display = ''
        else:
            self.fit_lines_container.layout.display = 'none'

    def toggle_hide_planes_widgets(self, b):
        """Show/hide the Fit Planes widgets."""
        if self.fit_planes_container.layout.display == 'none':
            self.fit_planes_container.layout.display = ''
        else:
            self.fit_planes_container.layout.display = 'none'

    def apply_fits(self, b):

        self.lines, self.planes = [], []
        filtered = self.specimen.get_filtered_specimen()

        with self.message_output_area1:
            self.message_output_area1.clear_output(wait=True)
            
            for line_widgets in self.fit_line_widgets:
                line_from_dropdown, line_to_dropdown, include_origin, anchor, line_name, line_color, widget_row = line_widgets
                
                # Access values from each widget in the row
                lfrom = line_from_dropdown.value
                lto = line_to_dropdown.value
                lorigin = include_origin.value
                lanchor = anchor.value
                lname = line_name.value
                lcolor = line_color.value
    
                if lfrom is None or lto is None or lfrom >= lto:
                    print(f"Invalid index selection for {lname}, skipping...")
                    continue
    
                lfit = filtered.loc[lfrom:lto] 
                v1, mad, v1_segment = pro.linefit(np.column_stack((lfit['x1'], lfit['x2'], lfit['x3'])), lorigin, lanchor)
                di = pro.to_sph([v1])
                print(f'Comp. {lname} (n={len(lfit)}), Dec: {di[0][0]:.2f}, Inc: {di[0][1]:.2f}, MAD: {mad:.2f}')
                
                self.lines.append(['line', lname, lfit, lorigin, lanchor, v1, mad, v1_segment, lcolor])

            for plane_widgets in self.fit_plane_widgets:
                plane_from_dropdown, plane_to_dropdown, normalize, apply_constraints, plane_name, plane_color, widget_row = plane_widgets
                
                # Access values from each widget in the row
                pfrom = plane_from_dropdown.value
                pto = plane_to_dropdown.value
                pnormalize = normalize.value
                pconstraints = apply_constraints.value
                pname = plane_name.value
                pcolor = plane_color.value
    
                if pfrom is None or pto is None or pfrom >= pto:
                    print(f"Invalid index selection for {pname}, skipping...")
                    continue
    
                pfit = filtered.loc[pfrom:pto] 
                v3, mad, gc_segment = pro.gcfit(np.column_stack((pfit['x1'], pfit['x2'], pfit['x3'])), pnormalize, pconstraints)
                di = pro.to_sph([v3])
                print (f'GC. {pname} (n={len(pfit)}), Dec: {di[0][0]:.2f}, Inc: {di[0][1]:.2f}, MAD: {mad:.2f}')
                
                self.planes.append(['plane', pname, pfit, pnormalize, pconstraints, v3, mad, gc_segment, pcolor])

        self.manager.data_dict[self.manager.selected_specimen_name]['lines'] = self.lines
        self.manager.data_dict[self.manager.selected_specimen_name]['planes'] = self.planes

        self.fits_applied = True
        self.plotter.update_plot1(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, self.plot_output_area1)

    
    def clear_fits(self, b=None):

        with self.message_output_area1:
            self.message_output_area1.clear_output(wait=True)
            print ('')
        
        self.lines, self.planes, self.coefficients = [], [], []
        self.manager.data_dict[self.manager.selected_specimen_name]['lines'] = self.lines
        self.manager.data_dict[self.manager.selected_specimen_name]['planes'] = self.planes
        self.manager.data_dict[self.manager.selected_specimen_name]['coefficients'] = self.coefficients

        if self.fits_applied: self.plotter.update_plot1(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen, self.lines, self.planes, plot_output_area=self.plot_output_area1)
        with self.plot_output_area2:
            self.plot_output_area2.clear_output()
    
    
    def add_fit_line_row(self, b=None, prefill=None):
        
        line_from_dropdown = widgets.Dropdown(options=list(self.specimen.get_filtered_specimen().index), description='From:', layout=widgets.Layout(width="150px"))
        line_to_dropdown = widgets.Dropdown(options=list(self.specimen.get_filtered_specimen().index), description='To:', layout=widgets.Layout(width="150px"))
        include_origin_checkbox = widgets.Checkbox(value=False, description='Include Origin', layout=widgets.Layout(width="200px"), margin='0px')
        anchor_checkbox = widgets.Checkbox(value=False, description='Anchor', layout=widgets.Layout(width="200px"), margin='0px')
        line_name_text = widgets.Text(description='Component:', layout=widgets.Layout(width="200px"))
        line_color_dropdown = widgets.ColorPicker(value='green', description='Color:', layout=widgets.Layout(width="200px"))
        remove_line_button = widgets.Button(description="X", layout=widgets.Layout(width="30px"))
        remove_line_button.on_click(lambda x: self.remove_fit_line_row(widget_row))

        if prefill:
            line_from_dropdown.value = prefill[2].index[0]
            line_to_dropdown.value = prefill[2].index[-1]
            include_origin_checkbox.value = prefill[3]
            anchor_checkbox.value = prefill[4]
            line_name_text.value = prefill[1]
            line_color_dropdown.value = prefill[8]

        widget_row = widgets.HBox([line_from_dropdown, line_to_dropdown, include_origin_checkbox, anchor_checkbox, line_name_text, line_color_dropdown, remove_line_button],
                                 layout=widgets.Layout(width='100%', justify_content='flex-start', spacing='0px'))
        self.fit_line_widgets.append((line_from_dropdown, line_to_dropdown, include_origin_checkbox, anchor_checkbox, line_name_text, line_color_dropdown, widget_row))
        self.fit_lines_container.children += (widget_row,)

    def remove_fit_line_row(self, widget_row):
        """Remove a row of fit-line widgets"""
        self.fit_line_widgets = [w for w in self.fit_line_widgets if w[-1] != widget_row]
        self.fit_lines_container.children = [w[-1] for w in self.fit_line_widgets]

    def add_fit_plane_row(self, b=None, prefill=None):
        
        plane_from_dropdown = widgets.Dropdown(options=list(self.specimen.get_filtered_specimen().index), description='From:', layout=widgets.Layout(width="150px"))
        plane_to_dropdown = widgets.Dropdown(options=list(self.specimen.get_filtered_specimen().index), description='To:', layout=widgets.Layout(width="150px"))       
        normalize_checkbox = widgets.Checkbox(value=False, description='Normalize', layout=widgets.Layout(width="200px"), margin='0px')
        apply_constraints_checkbox = widgets.Checkbox(value=False, description='Apply Constraints', layout=widgets.Layout(width="200px"), margin='0px')
        plane_name_text = widgets.Text(description='Component:', layout=widgets.Layout(width="200px"))
        plane_color_dropdown = widgets.ColorPicker(value='green', description='Color:', layout=widgets.Layout(width="200px"))
        remove_plane_button = widgets.Button(description="X", layout=widgets.Layout(width="30px"))
        remove_plane_button.on_click(lambda x: self.remove_fit_plane_row(widget_row))

        if prefill:
            plane_from_dropdown.value = prefill[2].index[0]
            plane_to_dropdown.value = prefill[2].index[-1]
            normalize_checkbox.value = prefill[3]
            apply_constraints_checkbox.value = prefill[4]
            plane_name_text.value = prefill[1]
            plane_color_dropdown.value = prefill[8]

        widget_row = widgets.HBox([plane_from_dropdown, plane_to_dropdown, normalize_checkbox, apply_constraints_checkbox, plane_name_text, plane_color_dropdown, remove_plane_button],
                                 layout=widgets.Layout(width='100%', justify_content='flex-start', spacing='0px'))
        self.fit_plane_widgets.append((plane_from_dropdown, plane_to_dropdown, normalize_checkbox, apply_constraints_checkbox, plane_name_text, plane_color_dropdown, widget_row))
        self.fit_planes_container.children += (widget_row,)

    def remove_fit_plane_row(self, widget_row):
        """Remove a row of fit-line widgets"""
        self.fit_plane_widgets = [w for w in self.fit_plane_widgets if w[-1] != widget_row]
        self.fit_planes_container.children = [w[-1] for w in self.fit_plane_widgets]

    def toggle_PCA_display(self, b):
        """toggle the visibility of PCA display"""
        self.message_output_area2.layout.display = '' if self.message_output_area2.layout.display == 'none' else 'none'

    def do_autoPCA(self, weight=0.5, penalty=0.5):
        
        weight = float(self.PCA_weight_text_box.value) if self.PCA_weight_text_box.value.strip() else 0.5
        penalty = float(self.PCA_penalty_text_box.value) if self.PCA_penalty_text_box.value.strip() else 0.5

        with self.message_output_area2:
            self.message_output_area2.clear_output(wait=True)
            pro.cycle_autoPCA(self.specimen.get_filtered_specimen(), weight, penalty)

    def toggle_linmod_display(self, b):
        """toggle the visibility of PCA display"""
        self.plot_output_area2.layout.display = '' if self.plot_output_area2.layout.display == 'none' else 'none'

    def run_linear_model(self, b=None):

        fspec = self.specimen.get_filtered_specimen()
        lines = self.get_lines()
        self.fitted_points, self.coefficients = pro.dirmod(np.column_stack((fspec['x'], fspec['y'], fspec['z'])), [x[5] for x in lines])
        
        self.plotter.update_plot2(self.coordinate_system.coordinates, self.axes_projection.projection, self.specimen.get_raw_specimen(), 
                                  self.specimen.get_filtered_specimen(), self.lines, self.fitted_points, self.coefficients, plot_output_area=self.plot_output_area2)

        self.manager.data_dict[self.manager.selected_specimen_name]['coefficients'] = self.coefficients
        self.linmod_applied = True

    def get_lines(self):
        return self.lines

    def get_planes(self):
        return self.planes
        
