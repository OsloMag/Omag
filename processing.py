import re, copy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import pickle, base64


############## File formatting ##############

def import_jr6(filename):
    """ Imports standard JR6 data file into formatted Pandas DataFrame """
    
    cols = ['specimen', 'treatment', 'x', 'y','z','exp','az','dip','fol_az','fol_dip','lin_trend','lin_plunge','P1','P2','P3','P4','precision','end']
    widths = [10,8,6,6,6,4,4,4,4,4,4,4,3,3,3,3,4,2]  # standard jr6 column widths
    
    dtype_map = {'x': float, 'y': float, 'z': float, 'exp': float}  # specify some datatypes 
    df = pd.read_fwf(filename, widths=widths, names=cols, dtype=dtype_map)

    # set correct intensity for x,y,z components
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].mul(10 ** df['exp'], axis=0)
    df['res'] = np.linalg.norm(df[['x', 'y', 'z']], axis=1)  # compute resultant

    # determine the orientation conventions
    if df[['P1', 'P2', 'P3', 'P4']].nunique().nunique() != 1:  # check that the values are consistent per column
        print('Orientation conventions are not the same throughout the file!')
        return None
        
    P1, P2, P3, P4 = df.iloc[0][['P1', 'P2', 'P3', 'P4']] # extract first row values for orientation check
    
    if (P1, P2, P3, P4) == (12, 0, 12, 0):
        df['hade'], df['bed_dip'], df['bed_dip_dir'] = df['dip'], df['fol_dip'], df['fol_dip']
    elif (P1, P2, P3, P4) == (12, 0, 12, 90):
        df['hade'], df['bed_dip'], df['bed_dip_dir'] = df['dip'], df['fol_dip'], df['fol_dip'] + 90
    else:
        print('This orientation convention is not yet handled')
        return None 
    
    # split out the treatment suffix and the demag step value
    df['demag'] = df['treatment'].str[0].map({'N': 'None', 'A': 'AF', 'T': 'TH'}).fillna('Unknown')
    # remove the letters from the treatment and convert to int (if NRM, assign to '0')
    df['treatment'] = df['treatment'].str.extract(r'(\d+)').fillna(0).astype(int)

    # compute dec and inc in specimen coordinates
    dec_inc = to_sph(df[['x', 'y', 'z']].values)
    df[['sdec', 'sinc']] = dec_inc

    df.drop(columns=['exp','dip','fol_az','fol_dip','lin_trend','lin_plunge','P1','P2','P3','P4','precision','end'], inplace=True)
    # reorder columns
    df = df[['specimen', 'demag', 'treatment', 'x', 'y', 'z', 'res', 'sdec', 'sinc', 'az', 'hade', 'bed_dip', 'bed_dip_dir']]
    
    return df

def import_2G_binary(filename):
    """ Imports 2G binary .dat file into Pandas DataFrame """

    def find_next(line, idx):
        """ find non-zero entries in binary header """
        for next_idx in range(idx, len(line)):
            if line[next_idx] not in {b'\x00', b'\x01', b''}:
                next_item = line[next_idx].decode(errors='ignore')
                return next_item, next_idx

    with open(filename, 'rb') as datafile:
        raw = datafile.read()
        split_raw = raw.split(b'\xcd')  # split data file into lines
    
        # collect header data
        line1 = split_raw[0]
        parts = [x for x in line1.split(b'\x00')]
        idx = parts.index(b'\x88')
        specimen, idx = find_next(parts, idx+1)   # get the specimen name
        vol, idx = find_next(parts, idx+1)        # get specimen volume
        date, idx = find_next(parts, idx+1)       # get date 
        idx+=1                                    # grab comment if present (or empty space if not)
        comment = parts[idx].decode(errors='ignore')

        azimuth, idx = find_next(parts, idx+1)
        plunge, idx = find_next(parts, idx+1) 
        dip_dir, idx = find_next(parts, idx+1) 
        dip, idx = find_next(parts, idx+1) 

        vol = float(vol)
        azimuth = float(azimuth)
        plunge = float(plunge)
        dip_dir = float(dip_dir)
        dip = float(dip)
    
        # collect measurement data
        demag = []
        Ds, Is, Dg, Ig, Dt, It = [], [], [], [], [], []
        M, J = [], []
        X, SX, NX, EX = [], [], [], []
        Y, SY, NY, EY = [], [], [], []
        Z, SZ, NZ, EZ = [], [], [], []
        SN, SD, SH = [], [], []
        axis, mdate = [], []
        
        for line in split_raw[1::]:
            data = [x for x in line.split(b'\x00') if x]
            
            demag.append(data[0].decode(errors='ignore'))
            Ds.append(float(data[1].decode(errors='ignore')))
            Is.append(float(data[2].decode(errors='ignore')))
            Dg.append(float(data[3].decode(errors='ignore')))
            Ig.append(float(data[4].decode(errors='ignore')))
            Dt.append(float(data[5].decode(errors='ignore')))
            It.append(float(data[6].decode(errors='ignore')))
            M.append(float(data[7].decode(errors='ignore')))
            J.append(float(data[8].decode(errors='ignore')))
            X.append(float(data[9].decode(errors='ignore')))
            SX.append(float(data[10].decode(errors='ignore')))
            NX.append(int(data[11].decode(errors='ignore')))
            EX.append(float(data[12].decode(errors='ignore')))
            Y.append(float(data[13].decode(errors='ignore')))
            SY.append(float(data[14].decode(errors='ignore')))
            NY.append(int(data[15].decode(errors='ignore')))
            EY.append(float(data[16].decode(errors='ignore')))
            Z.append(float(data[17].decode(errors='ignore')))
            SZ.append(float(data[18].decode(errors='ignore')))
            NZ.append(int(data[19].decode(errors='ignore')))
            EZ.append(float(data[20].decode(errors='ignore')))
            SN.append(float(data[21].decode(errors='ignore')))
            SD.append(float(data[22].decode(errors='ignore')))
            SH.append(float(data[23].decode(errors='ignore')))
            axis.append(data[24].decode(errors='ignore'))
            mdate.append(data[25].decode(errors='ignore'))

    # pass all the data into a dataframe
    dict_2G = {'DEMAG': demag, 'Ds': Ds, 'Is': Is, 'Dg': Dg, 'Ig': Ig, 'Dt': Dt, 'It': It, 'M': M, 'J': J, 
                 'X': X, 'SX': SX, 'NX': NX, 'EX': EX, 'Y': Y, 'SY': SY, 'NY': NY, 'EY': EY, 'Z': Z, 'SZ': SZ, 'NZ': NZ, 'EZ': EZ,
                 'SN': SN, 'SD': SD, 'SH': SH, 'axis': axis, 'mdate': mdate}
    
    df_2G = pd.DataFrame(dict_2G)
    df_2G['specimen'] = specimen
    df_2G['vol'] = vol
    df_2G['az'] = azimuth
    df_2G['plunge'] = plunge
    df_2G['bed_dip_dir'] = dip_dir
    df_2G['bed_dip'] = dip
    df_2G['comment'] = comment
    
    return df_2G

def import_2G(filename):
    """ Imports 2G data file and returns the reformatted data in a Pandas dataframe """

    df = import_2G_binary(filename)  # import the 2G data and get into a dataframe

    # parse the demag steps
    df['demag'] = None
    df['treatment'] = 'unknown'
    for idx, val in df['DEMAG'].items():
        if 'mT' in val:
            df.at[idx, 'demag'] = 'AF'
            df.at[idx, 'treatment'] = int(val.replace('mT', ''))
        elif 'C' in val:
            df.at[idx, 'demag'] = 'TH'
            df.at[idx, 'treatment'] = int(val.replace('C', ''))
        elif 'NRM' in val:
            df.at[idx, 'demag'] = 'None'
            df.at[idx, 'treatment'] = 0
        else:
            df.at[idx, 'demag'] = 'unknown'
            df.at[idx, 'treatment'] = val

    # convert the measurements from emu to A/m
    df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].div(df['vol'], axis=0) # divide by volume to get magnetization    
    df[['x', 'y', 'z']] = df[['X', 'Y', 'Z']].mul(1000)  # convert from emu/cc to A/m
    df['res'] = np.linalg.norm(df[['x', 'y', 'z']], axis=1)  # compute resultant

    # compute dec and inc in specimen coordinates
    dec_inc = to_sph(df[['x', 'y', 'z']].values)
    df[['sdec', 'sinc']] = dec_inc
    
    df['hade'] = 90-df['plunge']
    
    selected_columns = ['specimen', 'demag', 'treatment', 'x', 'y', 'z', 'res', 'sdec', 'sinc', 'az', 'hade', 'bed_dip', 'bed_dip_dir']
    df_reformatted = df[selected_columns]

    return df_reformatted

def process_pkl(data_dict):
    """ Imports pickled data saved by mag_fits notebook for processing in mag_stats notebook """

    Ds, Is, Dg, Ig, Dt, It, gcs, gcg, gct = [[] for _ in range(9)]
    spec, comp, fit_type, n, from_treat, to_treat, mad, demag, coefficients, treatment, segments = [[] for _ in range(11)]
    
    for key, sub_dict in data_dict.items():
        #for subkey, elem in sub_dict.items():

        merged_coeffs = sub_dict['coefficients_norm'] if 'coefficients_norm' in sub_dict and isinstance(sub_dict['coefficients_norm'], list) and sub_dict['coefficients_norm'] else None
        if merged_coeffs is not None: split_coeffs = list(zip(*merged_coeffs))
        else: split_coeffs = None

        if isinstance(sub_dict['filtered'], pd.DataFrame):
            treatment_steps = sub_dict['filtered']['treatment'].values
    
        if 'lines' in sub_dict:               
            for i, line in enumerate(sub_dict['lines']):
                lfit = line[2]
                dirs = line[5]
                coordinates = line[8]
                DI_s, DI_g, DI_t = collect_alt_dirs(dirs, coordinates, lfit['az'].iloc[0], lfit['hade'].iloc[0], lfit['bed_dip_dir'].iloc[0], lfit['bed_dip'].iloc[0])
                Ds.append(DI_s[0][0])
                Is.append(DI_s[0][1])
                Dg.append(DI_g[0][0])
                Ig.append(DI_g[0][1])
                Dt.append(DI_t[0][0])
                It.append(DI_t[0][1])
                gcs.append(None)
                gcg.append(None)
                gct.append(None)

                spec.append(key)
                comp.append(line[1])
                fit_type.append(line[0])
                n.append(len(line))
                mad.append(line[6])
                
                from_treat.append(lfit['treatment'].iloc[0])
                to_treat.append(lfit['treatment'].iloc[-1])
                treatment.append(treatment_steps)
                
                demag_treat = lfit['demag'].astype(str)
                if demag_treat.str.contains('TH').any():
                    demag_type  = 'TH'
                else: demag_type = 'AF'
                demag.append(demag_type)

                if split_coeffs is None:
                    #print (f'The number of line fits and coefficient arrays does not match for {key}; not passing any coefficients')
                    coefficients.append(None)
                elif len(split_coeffs) != len(sub_dict['lines']):
                    #print (f'The number of line fits and coefficient arrays does not match for {key}; not passing any coefficients')
                    coefficients.append(None)
                else:
                    coefficients.append(np.array(split_coeffs[i]))

        if 'planes' in sub_dict:
            for i, plane in enumerate(sub_dict['planes']):
                
                gcfit = plane[2]
                dirs = plane[5]
                coordinates = plane[8]
                DI_s, DI_g, DI_t = collect_alt_dirs(dirs, coordinates, gcfit['az'].iloc[0], gcfit['hade'].iloc[0], gcfit['bed_dip_dir'].iloc[0], gcfit['bed_dip'].iloc[0])
                Ds.append(DI_s[0][0])
                Is.append(DI_s[0][1])
                Dg.append(DI_g[0][0])
                Ig.append(DI_g[0][1])
                Dt.append(DI_t[0][0])
                It.append(DI_t[0][1])
                gcpts_s, gcpts_g, gcpts_t = collect_alt_dirs(plane[7], coordinates, gcfit['az'].iloc[0], gcfit['hade'].iloc[0], gcfit['bed_dip_dir'].iloc[0], gcfit['bed_dip'].iloc[0])
                gcs.append(gcpts_s)
                gcg.append(gcpts_g)
                gct.append(gcpts_t)    

                spec.append(key)
                comp.append(plane[1])
                fit_type.append(plane[0])
                n.append(len(plane))
                mad.append(plane[6])
                
                from_treat.append(gcfit['treatment'].iloc[0])
                to_treat.append(gcfit['treatment'].iloc[-1])
                treatment.append(treatment_steps)

                demag_treat = gcfit['demag'].astype(str)
                if demag_treat.str.contains('TH').any():
                    demag_type  = 'TH'
                else: demag_type = 'AF'
                demag.append(demag_type)

                coefficients.append(None)

    df = pd.DataFrame({'specimen': spec, 'component': comp, 'fit_type': fit_type, 'n': n, 'from': from_treat, 'to': to_treat, 
                       'Ds': Ds, 'Is': Is, 'Dg': Dg, 'Ig': Ig, 'Dt': Dt, 'It': It, 'mad': mad, 'demag': demag, 'coefficients': coefficients, 
                       'treatment': treatment, 'gcs': gcs, 'gcg': gcg, 'gct': gct})
    return df


############## Coordinate transformations ##############

def set_coordinates(coordinates, df):
    """ Switches x-y-z and dec/inc coordinates (specimen / geographic / tectonic) in a dataframe """
    
    if coordinates == 'specimen': 
        df[['x1', 'x2', 'x3']] = df[['x', 'y', 'z']]
        df[['dec', 'inc']] = df[['sdec', 'sinc']]
        df['coordinates'] = 'specimen'

    elif coordinates in ['geographic', 'tectonic']:
        rotated_pts = spe2geo(np.column_stack((df['x'], df['y'], df['z'])), df['az'], df['hade'])
        df[['x1', 'x2', 'x3']] = rotated_pts[:, :3]

        dec_inc = to_sph(rotated_pts)
        df[['dec', 'inc']] = dec_inc[:, :2]
        
        if coordinates =='geographic': df['coordinates'] = 'geographic'
        
        else:
            untilted_pts = untilt(rotated_pts, df['bed_dip_dir'], df['bed_dip'])
            df[['x1', 'x2', 'x3']] = untilted_pts[:, :3]
            dec_inc = to_sph(untilted_pts)
            df[['dec', 'inc']] = dec_inc[:, :2]
            df['coordinates'] = 'tectonic'
            
    return df

def to_sph(vecs):
    """ 
    Convert Cartesian vectors to spherical coordinates  
    Input: [[x, y, z], ..., [x, y, z]];  Output: [[dec, inc], ..., [dec, inc]]
    """
    
    vecs = np.asarray(vecs)
    norms = np.linalg.norm(vecs, axis=1)
    
    dec = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    inc = np.degrees(np.arcsin(vecs[:, 2] / norms))
    return  np.stack((dec, inc), axis=1)

def to_car(dirs):
    """ 
    Convert spherical directions to Cartesian coordinates  
    Input: [[dec, inc], ..., [dec, inc]];  Output: [[x, y, z], ..., [x, y, z]]
    """
    
    dirs = np.asarray(dirs)
    dec, inc = np.radians(dirs[:, 0]), np.radians(dirs[:, 1])

    cos_inc = np.cos(inc)
    x, y, z = cos_inc * np.cos(dec), cos_inc * np.sin(dec), np.sin(inc)
    return np.stack((x, y, z), axis=1)

def spe2geo(vecs, az, hade):
    """
    Rotates Cartesian vectors from specimen to geographic coordinates using given azimuth and hade
    Input: [[x, y, z], ..., [x, y, z]], azimuth (in degrees), hade (in degrees);  Output: [[x, y, z], ..., [x, y, z]]
    """
    az_rad = np.radians(az)
    hade_rad = np.radians(hade)
    R_az = R.from_euler('z', az_rad, degrees=False)
    R_hade = R.from_euler('y', hade_rad, degrees=False)
    rot = R_az * R_hade
    rot_data = rot.apply(vecs)
    return rot_data

def untilt(vecs, dipdir, dip):  # NEED TO CHECK THAT THIS WORKS PROPERLY!
    """
    Applies tilt correction to Cartesian vectors using dip direction and dip.
    Input: [[x, y, z], ..., [x, y, z]], dip-direction (in degrees), dip (in degrees);  Output: [[x, y, z], ..., [x, y, z]]
    """
    rot = R.from_euler('y', dip, degrees=True) * R.from_euler('z', -(dipdir - 90), degrees=True)
    return rot.apply(vecs)
    
def collect_alt_dirs(vecs, coordinates, az, hade, dipdir, dip):
    """
    Computes the mean direction in all three coordinate systems (specimen, geographic, tectonic).
    Input: [[x, y, z], ..., [x, y, z]], coordinates (string), azimuth, hade, dip-direction, dip (the latter 4 in degrees)
    Output: [[x, y, z], ..., [x, y, z]]
    """
    if vecs.ndim == 1: vecs = vecs.reshape(1, -1) 
    
    to_specimen = lambda v: to_sph(spe2geo(v, 360 - az, -hade))
    to_geographic = lambda v: to_sph(spe2geo(v, az, hade))
    to_tectonic = lambda v: to_sph(untilt(to_car(to_geographic(v)), dipdir, dip))

    if coordinates == "specimen":
        return to_sph(vecs), to_geographic(vecs), to_tectonic(vecs)

    if coordinates == "geographic":
        return to_specimen(vecs), to_sph(vecs), to_tectonic(vecs)

    if coordinates == "tectonic":
        DI_g = to_sph(untilt(vecs, dipdir, -dip))  # reverse tilt correction
        return to_specimen(to_car(DI_g)), DI_g, to_sph(vecs)

    raise ValueError("Invalid coordinate system specified.")


############## Component fitting routines ##############

def doPCA(vecs):
    """
    Applies PCA to a set of Cartesian vectors
    Input: [[x, y, z], ..., [x, y, z]];  Output: eigenvectors, eigenvalues, mean    
    """
    pca = PCA(n_components=3, svd_solver="full")  # make PCA object
    evecs = pca.fit(vecs).components_  # fit components and get eigenvectors (transforms/centers the data by removing mean)
    evals = np.sort(pca.explained_variance_)[::-1]  # get eigenvalues
    
    return evecs, evals, pca.mean_ 

def doPCA_anchored(vecs):
    """
    Does Single value decomposition (PCA with data anchored to the origin)
    Input: [[x, y, z], ..., [x, y, z]];  Output: eigenvectors, eigenvalues
    """
    pca = TruncatedSVD(n_components=3, algorithm='randomized')  # make PCA object
    evecs = pca.fit(vecs).components_      # fit components and get eigenvectors (does not remove the mean to re-center the data)
    svals = np.sort(pca.singular_values_)[::-1]  # get the single values 
    evals = (svals ** 2) / len(vecs)   # convert single values to equivalent eigenvalues
    
    return evecs, evals
               
def linefit(vecs, incl_origin=False, anchor=False):
    """
    Executes linear fitting via PCA on a set of cartesian vectors. Returns the first principal component, the associated MAD angle, 
    and a line segment for visualization.
    Input: [[x, y, z], ..., [x, y, z]], include_origin (True/False), anchor (True/False)
    Output: v1 (principal eigenvector), mad angle and line segment delineating the fitted component.
    """
    if incl_origin: 
        vecs = np.concatenate([vecs, [[0, 0, 0]]])  # append the origin itself to the end of the data before doing PCA
    
    result = doPCA_anchored(vecs) if anchor else doPCA(vecs)
    evecs, evals, centroid = (result + (None,))[:3]  # unpack based on PCA method used
    
    v1 = evecs[0]  # get first principal component
    mad = np.degrees(np.arctan(np.sqrt((evals[1] + evals[2]) / evals[0]))) # get MAD angle   
    
    #check polarity of principal component (the polarity of which is arbitrary)
    if np.dot(v1, vecs[0] - vecs[-1]) < 0:
        v1 *= -1

    #get end-points of a line segment that is parallel to the principal component, and centered on the data mean (for plotting)
    scale = np.sqrt(evals[0] * len(vecs))  # re-scale the component to the original data scale
    if anchor: 
        v1_segment = [[0,0,0], v1 * scale]
    else: 
        v1_segment = [centroid - 0.5 * scale * v1, centroid + 0.5 * scale * v1]
    
    return v1, mad, v1_segment

def gcfit(vecs, normalize=False, constraints=False):
    """
    Executes planar fitting via PCA on a set of cartesian vectors; returns the third principal component (normal to the plane), 
    the associated MAD angle, and a series of points tracing the circle (or a section of the circle if constraints are applied).
    Input: [[x, y, z], ..., [x, y, z]], normalize (True/False), constraints (True/False) 
    Output: v3 (normal to the great circle), mad angle, points delineating the plane.
    """
    if normalize:          # normalize vector lengths if desired
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    evecs, evals = doPCA_anchored(vecs)  # do PCA (send to the anchored method so that the plane is constrained to pass through the origin)
    v3 = evecs[-1]  # get the smallest principal component
    mad = np.degrees(np.arctan(np.sqrt(evals[2]/evals[1] + evals[2]/evals[0])))  # get mad angle

    if constraints:   # if constraints used, grab first and last points
        last_3_dirs = to_sph(vecs[-3:])  # take mean of last 3 points to avoid problems with outlying last point(s)
        fmean = fisher_mean(last_3_dirs)
        mean_last_vecs = to_car([[fmean['dec'], fmean['inc']]])
        constraint_pts = [vecs[0], mean_last_vecs[0]]
    else:
        constraint_pts = []
    gc_segment = sample_gc(v3, constraint_pts, ang_step=0.5)  # get points along GC (or partial segment)
    
    return v3, mad, gc_segment

def sample_gc(vnorm, constraints, ang_step=0.5):
    """
    Collect points along a great circle (defined by the normal to the plane: vnorm). If constraints are provided, only a section of the GC arc is sampled.
    Input: vnorm ([x, y, z]), constraints ([[x1, y1, z1], [x2, y2, z2]]), angle step size (degrees).
    Output: points delineating the great circle (or a small segment along it).
    """
    # choose start point based on the direction of vnorm
    start_pt = np.cross(vnorm, [1, 0, 0]) if not np.isclose(vnorm[0], 1) else np.array([0, 1, 0])
    
    angs = np.arange(0, 360, ang_step) # setup the array of angles to rotate by
    rot = R.from_rotvec(np.radians(angs)[:, np.newaxis] * vnorm)
    gc_pts = rot.apply(start_pt)

    if constraints: 
        gc_pts = subsample_gc(gc_pts, vnorm, constraints, ang_step)
        
    return gc_pts

def subsample_gc(gc_pts, vnorm, constraints, ang_step=0.5):
    """
    Collect points along the arc of a great circle between two points (constraints). Returns points along this arc.
    Input: points along great circle [[x, y, z], ..., [x, y, z]], vnorm ([x, y, z]), constraints ([[x1, y1, z1], [x2, y2, z2]]), angular step size (in degrees)
    Output: points delineating the great circle (or a small segment along it).
    """
    antipode = -constraints[0]  # antipode of starting data point
    init_idx = np.argmin(np.abs(np.array([angle(pt, antipode) for pt in gc_pts])))
    start_pt = gc_pts[init_idx] # the closest point to antipode becomes starting point

    # now identify polarity of vector normal to the GC
    testang = angle(start_pt, constraints[1])
    test_pt = R.from_rotvec(np.radians(10.0) * vnorm).apply(start_pt) # rotate starting point by 10 degree
    if angle(test_pt, constraints[1]) > testang: # if the point moves away from the end-point, invert the pole
        vnorm = vnorm * -1

    # now generate a new GC series from starting point until the final constraint is reached
    angs = np.arange(0, 180, ang_step)
    rot = R.from_rotvec(np.radians(angs)[:, np.newaxis] * vnorm)
    seg_pts = rot.apply(start_pt)  # rotate from starting point to a point 180 degrees away (to ensure we pass final constraint)

    # find the point at which we cross the final constraint, and segment is everything up to that point
    final_idx = np.argmin(np.abs(np.array([angle(pt, constraints[1]) for pt in seg_pts])))
    
    return seg_pts[:final_idx]

def autoPCA(fspec, comps=1, w=0.5, p=0.5):
    """
    Performs PCA for all possible consecutive series of data (n>2), and assuming 1, 2, or 3 components to automatically find best options.
    Input: Cartesian vectors [[x, y, z], ..., [x, y, z]], the max number of components to try to fit (1-3), 
    a weighting value (0-1, higher values favor lower MAD; lower values favor more points), 
    and a penalty value (higher penalty favors less components).
    Output: the results per number of components, sorted by the score set by the weighting and penalty values.
    """
    fit_results = []

    # convert relevant columns of fspec to a numpy array for faster slicing
    points = np.column_stack((fspec['x'].values, fspec['y'].values, fspec['z'].values))

    # iterate through all combinations of series in fspec
    for i1 in range(len(fspec)):
        for j1 in range(i1+3, len(fspec)):
            range1 = points[i1:j1+1]
            _, mad1, _ = linefit(range1)       
            num_pts1 = j1-i1+1
            
            if comps <= 1:  # if assuming just one component
                score = -w*mad1 + (1-w)*num_pts1 - 1*p
                fit_results.append((i1, j1, num_pts1, mad1, score))
        
            else:  # otherwise look for a 2nd component that lies at higher range than first component
                for i2 in range(j1, len(fspec)):
                    for j2 in range(i2+3, len(fspec)):
                        range2 = points[i2:j2+1]
                        _, mad2, _ = linefit(range2) 
                        num_pts2 = j2-i2+1

                        if comps <= 2:  # if assuming just 2 components
                            avg_mad = (mad1+mad2)/2
                            score = -w*avg_mad + (1-w)*(num_pts1+num_pts2) - 2*p
                            fit_results.append(((i1, i2), (j1, j2), (num_pts1, num_pts2), (mad1, mad2), score))
                       
                        else:  # otherwise look for a 3rd component that lies at higher range than second component
                            for i3 in range(j2, len(fspec)):
                                for j3 in range(i3+3, len(fspec)):
                                    range3 = points[i3:j3+1]
                                    _, mad3, _ = linefit(range3) 
                                    num_pts3 = j3-i3+1
                                    
                                    avg_mad = (mad1+mad2+mad3)/3
                                    score = -w*avg_mad + (1-w)*(num_pts1+num_pts2+num_pts3) - 3*p
                                    fit_results.append(((i1, i2, i3), (j1, j2, j3), (num_pts1, num_pts2, num_pts3), (mad1, mad2, mad3), score))
    
    results_sorted = sorted(fit_results, key=lambda x: x[-1], reverse=True)
    return results_sorted

#### ************* could optimize the codes below ******************** #####
    
def dirmod(observed, v1s):
    """
    finds the best fitting linear combination of the prescribed principal components (v1s) to explain the observed directional data
    """
    def lin_mod(params):
        return sum(a * pc for a, pc in zip(params, v1s))   

    def cost_func(params, obs):
        predicted = lin_mod(params)
        return np.sum((predicted - obs)**2)

    fitted_pts = []
    coefficients = []
    coefficients_norm = []
    prev_params = None
    init_norm = None
    for i, obs in enumerate(observed):
        init_guess = [1] * len(v1s)
        constraints = []
        if prev_params is not None:
            # apply constraints
            for idx in range(len(v1s)):
                # magnitude constraint: current coefficient magnitude <= previous coefficient magnitude
                constraints.append({'type': 'ineq', 'fun': lambda params, idx=idx: abs(prev_params[idx]) - abs(params[idx])})
                # sign constraint: the sign of the current coefficient should be the same as the previous one
                constraints.append({'type': 'ineq', 'fun': lambda params, idx=idx: params[idx] * prev_params[idx]})
            
        result = minimize(cost_func, init_guess, args=(obs,), method='SLSQP', constraints=constraints)
        params = result.x

        # normalize the coefficients to have a unit vector sum
        if init_norm is None:
            init_norm = np.linalg.norm(params)
        if init_norm != 0:
            params_norm = params / init_norm
        
        coefficients.append(tuple(params))
        coefficients_norm.append(tuple(params_norm))
        prev_params = params
        fitted_pt = lin_mod(params)
        fitted_pts.append(fitted_pt)
    
    return np.array(fitted_pts), coefficients, coefficients_norm
    

############## Mathematical / statistical operations ##############

def angle(v1, v2):
    """
    Compute the angle between two cartesian vectors.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dprod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = np.clip(dprod / (norm_v1 * norm_v2), -1.0, 1.0)
    
    return np.arccos(cos_theta)

def get_fisher_mean(df, coordinates, flip=False, w_gcs=False):
    """
    Set up Fisher mean calculation from dec/inc data in a dataframe.
    Input: Dataframe with prescribed column headers for dec, inc in various coordinates, flip (True/False) determines whether or
    not to flip any data that is 90 degrees from the mean, and w_gcs (True/False) determined whether to include any plane data or not.
    Output: the Fisher mean dictionary (following pmagpy convention).
    """
    
    ldf = df[df['fit_type'] == 'line']
    gcdf = df[df['fit_type'] == 'plane']
    
    if len(ldf) <= 2 and not w_gcs:
        print ('too few lines to compute Fisher mean')
        return None, None

    if coordinates == 'specimen': 
        decs, incs = ldf['Ds'], ldf['Is']
        gcs = gcdf['gcs']
    elif coordinates == 'geographic': 
        decs, incs = ldf['Dg'], ldf['Ig']
        gcs = gcdf['gcg']
    elif coordinates == 'tectonic': 
        decs, incs = df['Dt'], df['It']
        gcs = gcdf['gct']

    DI = np.column_stack((decs, incs))

    if flip:
        normal, flipped = do_flip(DI)
        if flipped: 
            print (f'{len(flipped)} directions flipped')
            DI = normal + flipped
    
    fmean = fisher_mean(DI)

    if not w_gcs or len(gcdf) == 0:
        return fmean, None
    else:
        return iterative_fisher_mean(DI, gcs)


def iterative_fisher_mean(DI, gcs):
    """
    Estimates the Fisher mean using a mixture of lines and planes, following the methodology of McFadden and McElhinny (1988).
    Input: dec/inc data [[dec, inc], ..., [dec, inc]] and great circle points [[dec, inc], ..., [dec, inc]].
    Output: Fisher mean estimate (following pmagpy convention, but with n = number of lines, m = number of planes) updated with k and alpha95
    recalculated following McFadden and McElhinny, and also outputs the contributing point on each great circle.
    """

    # first get an initial fisher mean 
    fmean = fisher_mean(DI)
    fcar = to_car([[fmean['dec'], fmean['inc']]]).flatten()

    # convert the GC data to cartesian coordinates
    car_gcs = []
    for gc in gcs:
        car_gcs.append(to_car(gc))
        
    # then make an initial pass through all the GCs and find the closest point and update the fmean as we go
    new_fcar = fcar
    new_DI = DI.copy() 
    nearest = np.empty((0, 2))
    for gc in car_gcs:
        angs = [np.abs(angle(pt, new_fcar)) for pt in gc] # find closest point
        min_idx = np.argmin(angs)
        ndir = to_sph([gc[min_idx]])            # convert to direction
        nearest = np.vstack([nearest, ndir])    # add to array
        new_DI = np.vstack([new_DI, ndir])      # compile new ID list
        new_fmean = fisher_mean(new_DI)         # compute new fdir
        new_fcar = to_car([[new_fmean['dec'], new_fmean['inc']]]).flatten()

    # now loop over the list of GCs and conduct a leave-one-out analysis until the closest directional estimates converge
    max_shift = 45
    while max_shift > 1.0:  # keep looping until the max change is less than one degree
        pt_shifts = []
        for i, gc in enumerate(car_gcs):
            former_pt = nearest[i]
            temp_nearest = np.concatenate((nearest[:i], nearest[i+1:]), axis=0)   # remove the corresponding point
            new_DI = np.vstack([DI, temp_nearest])   # recalculate the fisher mean without this point
            new_fmean = fisher_mean(new_DI)
            new_fcar = to_car([[new_fmean['dec'], new_fmean['inc']]]).flatten()
            angs = [np.abs(angle(pt, new_fcar)) for pt in gc]  # now find the closest point again
            min_idx = np.argmin(angs)
            new_pt = gc[min_idx]
            former_pt = to_car([former_pt]).flatten()
            shift = np.degrees(angle(new_pt, former_pt))
            pt_shifts.append(shift)
            nearest[i] = to_sph([new_pt])             # insert this new point in the nearest list
        max_shift = max(pt_shifts)          # after completing the loop, check the size of max value|

    # calculate final fisher mean and return
    fin_DI = np.vstack([DI, nearest])
    fin_fmean = fisher_mean(fin_DI)
    nearest_list = [[arr.tolist() for arr in sublist] for sublist in nearest]

    # calculate the modified k and alpha95 following the formulas in McFadden and McElhinny (1988)
    N = len(DI)
    M = len(gcs)
    R = fin_fmean['r']
    Ni = M+N/2
    
    k = (2*M+N-2) / (2*(M+N-R))
    a95 = np.degrees(np.arccos(1 - (Ni-1)/(k*R) * ((1/0.05)**(1/(Ni-1)) - 1)))

    fin_fmean['n'] = N
    fin_fmean['m'] = M
    fin_fmean['k'] = k
    fin_fmean['alpha95'] = a95
    del fin_fmean['csd']
    
    return fin_fmean, nearest_list


def gc_intersection(df, coordinates):
    """ ... """

    df = df[df['fit_type'] == 'plane']
    if len(df) < 4:
        print ('too few planes to conduct intersection analysis')
        return None
    
    if coordinates == 'specimen': 
        decs, incs = df['Ds'], df['Is']
    elif coordinates == 'geographic': 
        decs, incs = df['Dg'], df['Ig']
    elif coordinates == 'tectonic': 
        decs, incs = df['Dt'], df['It']

    DI = np.column_stack((decs, incs))
    vecs = to_car(DI)

    v3, mad, _ = gcfit(vecs, normalize=True, constraints=False)
    v3dir = to_sph([v3])
    idir = {'dec': v3dir[0][0], 'inc': v3dir[0][1], 'mad': mad, 'n': len(df)}

    return idir

############## Taken from Pmagpy ... need to fix ##############

def fisher_mean(dirs):
    """ ... """
    N, fpars = len(dirs), {}
    
    if N < 2: 
        return {'dec': dirs[0][0], 
                'inc': dirs[0][1]}

    x = np.array(to_car(dirs))
    xbar = x.sum(axis=0)
    res = np.linalg.norm(xbar)
    xbar_norm = xbar/res
    mean_dir = to_sph([xbar_norm])

    fpars["dec"] = mean_dir[0][0]
    fpars["inc"] = mean_dir[0][1]
    fpars["n"] = N
    fpars["r"] = res
    
    if N != res:
        k = (N - 1.) / (N - res)
        fpars["k"] = k
        csd = 81./np.sqrt(k)
    else:
        fpars['k'] = 'inf'
        csd = 0.
    b = 20.**(1./(N - 1.)) - 1
    a = 1 - b * (N - res) / res
    if a < -1:
        a = -1
    a95 = np.degrees(np.arccos(a))
    fpars["alpha95"] = a95
    fpars["csd"] = csd
    if a < 0:
        fpars["alpha95"] = 180.0
    return fpars


def do_flip(dirs):
    """ checks if there appear to be 2 polarity sets in a population of directions. Returns the larger of the two groups, and the 'inverted' one. """
    vecs = to_car(dirs)
    evecs, _ = doPCA_anchored(vecs)  # get principle direction
    ev1 = evecs[0]

    p1, p2 = [], []
    for i, v in enumerate(vecs):
        ang = np.degrees(angle(v, ev1))
        if ang < 90.: 
            p1.append(dirs[i])
        else:
            p2.append(dirs[i])

    if len(p1) == 0: 
        return p2, None
    elif len(p2) == 0: 
        return p1, None
    elif len(p1) >= len(p2):
        for i, rec in enumerate(p2):
            d, i = (rec[0] - 180.) % 360., -rec[1]
            p2[i] = np.array([d, i])
        return p1, p2
    else:
        for i, rec in enumerate(p1):
            d, i = (rec[0] - 180.) % 360., -rec[1]
            p1[i] = np.array([d, i])
        return p2, p1


############## Output ##############


"""
def update_saved_record(outfile, spec, fspec, lines, planes, coefficients, coordinates):

    save new data to csv file (this will overwrite any existing entry with the same specimen name, component name and fit type (line or plane)      

    # if there is an existing file, check if there is a row with the same specimen name, component name and fit type, and if so, overwrite
    cols2match = ['specimen', 'component', 'fit_type']
    new_df.set_index(cols2match, inplace=True)
    
    try:
        old_df = pd.read_csv(outfile)
        old_df.set_index(cols2match, inplace=True)
        updated_df = old_df.combine_first(new_df)
        updated_df.reset_index(inplace=True)
    
    except FileNotFoundError:
        updated_df = new_df.reset_index()
    
    updated_df.to_csv(outfile, index=False)  

"""

############## Other functions ##############

def cycle_autoPCA(fspec, weight, penalty):
    
    one_pc = autoPCA(fspec, comps=1, w=weight, p=penalty)
    print ("Best options for fitting a single component:")
    for res in one_pc[:3]: print (f"pc1: {res[0]}-{res[1]} (n={res[2]}), MAD={res[3]:.2f}; score: {res[4]:.2f}")
    
    two_pcs = autoPCA(fspec, comps=2, w=weight, p=penalty)
    print ("\nBest options for fitting two components:")
    for res in two_pcs[:3]: 
        print (f"pc1: {res[0][0]}-{res[1][0]} (n={res[2][0]}), MAD={res[3][0]:.2f} / pc2: {res[0][1]}-{res[1][1]} (n={res[2][1]}), MAD={res[3][1]:.2f}; score: {res[4]:.2f}")
    
    three_pcs = autoPCA(fspec, comps=3, w=weight, p=penalty)
    print ("\nBest options for fitting three components:")
    for res in three_pcs[:3]: 
        print (f"pc1: {res[0][0]}-{res[1][0]} (n={res[2][0]}), MAD={res[3][0]:.2f} / pc2: {res[0][1]}-{res[1][1]} (n={res[2][1]}), MAD={res[3][1]:.2f} / "
               f"pc3: {res[0][2]}-{res[1][2]} (n={res[2][2]}), MAD={res[3][2]:.2f}; score: {res[4]:.2f}")


def mean_decay(components, df):

    mean_treatments = []
    mean_coefficients = []
    mean_dMdD = []
    for c in components:
        ldf = df[(df['component'] == c) & (df['fit_type'] == 'line')]
        
        # do AF first
        AFdf = ldf[(ldf['demag'] == 'AF') &
           (ldf['treatment'].apply(lambda x: x is not None and len(x) > 0)) &
           (ldf['coefficients'].apply(lambda x: x is not None and len(x) > 0))]
        if len(AFdf) == 0:
            mean_treatments.append(None)
            mean_coefficients.append(None)
            mean_dMdD.append(None)
        else:
            treatments = AFdf['treatment'].reset_index(drop=True)
            coefficients = AFdf['coefficients'].reset_index(drop=True)
            reg_treatment_list = [np.arange(0, t.max() + 1, 1) for t in treatments]
            int_coefficients_list = []
            
            for i in range(len(treatments)):
                linear_interp = interp1d(treatments[i], abs(coefficients[i]), kind='linear', bounds_error=False, fill_value=np.nan)
                int_coefficients = linear_interp(reg_treatment_list[i])
                int_coefficients_list.append(int_coefficients)

            # Find the longest treatment sequence to align arrays
            max_length = max(len(arr) for arr in int_coefficients_list)
            
            # Pad arrays to match the longest one (using NaNs)
            padded_interpolated = [np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in int_coefficients_list]
            
            # Stack arrays and compute mean, ignoring NaNs
            interpolated_array = np.vstack(padded_interpolated)
            mean_interp_coeffs = np.nanmean(interpolated_array, axis=0)
            
            # Use the longest treatment sequence as the reference for final output
            ref_treat_steps = np.arange(0, max_length * 1, 1)  # Ensures consistent x-axis

            # approximate first derivative
            dMdD = np.gradient(mean_interp_coeffs, ref_treat_steps)
        
            # Store results
            mean_treatments.append(ref_treat_steps)
            mean_coefficients.append(mean_interp_coeffs)
            mean_dMdD.append(dMdD)

        # do TH
        THdf = ldf[(ldf['demag'] == 'TH') &
           (ldf['treatment'].apply(lambda x: x is not None and len(x) > 0)) &
           (ldf['coefficients'].apply(lambda x: x is not None and len(x) > 0))]
        if len(THdf) == 0:
            mean_treatments.append(None)
            mean_coefficients.append(None)
            mean_dMdD.append(None)
        else:
            treatments = THdf['treatment'].reset_index(drop=True)
            coefficients = THdf['coefficients'].reset_index(drop=True)
            reg_treatment_list = [np.arange(0, t.max() + 5, 5) for t in treatments]
            int_coefficients_list = []
            
            for i in range(len(treatments)):
                linear_interp = interp1d(treatments[i], abs(coefficients[i]), kind='linear', bounds_error=False, fill_value=np.nan)
                int_coefficients = linear_interp(reg_treatment_list[i])
                int_coefficients_list.append(int_coefficients)
                
            # Find the longest treatment sequence to align arrays
            max_length = max(len(arr) for arr in int_coefficients_list)
            
            # Pad arrays to match the longest one (using NaNs)
            padded_interpolated = [np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in int_coefficients_list]
            
            # Stack arrays and compute mean, ignoring NaNs
            interpolated_array = np.vstack(padded_interpolated)
            mean_interp_coeffs = np.nanmean(interpolated_array, axis=0)
            
            # Use the longest treatment sequence as the reference for final output
            ref_treat_steps = np.arange(0, max_length * 5, 5)  # Ensures consistent x-axis

            # approximate first derivative
            dMdD = np.gradient(mean_interp_coeffs, ref_treat_steps)
        
            # Store results
            mean_treatments.append(ref_treat_steps)
            mean_coefficients.append(mean_interp_coeffs)
            mean_dMdD.append(dMdD)

    return (mean_treatments, mean_coefficients, mean_dMdD)
        