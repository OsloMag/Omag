import re, copy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from pmagpy import ipmag, pmag
import pickle, base64


############## File formatting ##############

def import_jr6(filename):
    """
    Imports standard JR6 data file into a Pandas DataFrame.
    """
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

def deserialize(df):
    """
    recovers data saved in a serialized format
    """

    df['coefficients'] = df['coefficients'].apply(lambda x: np.array(pickle.loads(base64.b64decode(x))))
    df['treatment'] = df['treatment'].apply(lambda x: np.array(pickle.loads(base64.b64decode(x))))
    df['gcpts_s'] = df['gcpts_s'].apply(lambda x: pickle.loads(base64.b64decode(x)))
    df['gcpts_g'] = df['gcpts_g'].apply(lambda x: pickle.loads(base64.b64decode(x)))
    df['gcpts_t'] = df['gcpts_t'].apply(lambda x: pickle.loads(base64.b64decode(x)))

    return df
    

############## Coordinate transformations ##############

def set_coordinates(coordinates, df):
    """
    Switches x-y-z and dec/inc coordinates (specimen / geographic / tectonic) in a dataframe.
    """
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
    Convert Cartesian vectors (n,3) to declination (dec) and inclination (inc).
    """
    vecs = np.asarray(vecs)
    norms = np.linalg.norm(vecs, axis=1)
    
    dec = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    inc = np.degrees(np.arcsin(vecs[:, 2] / norms))
    return  np.stack((dec, inc), axis=1)

def to_car(dirs):
    """
    Convert declination (dec), inclination (inc) pairs (n,2) to unit-length Cartesian vectors.
    """
    dirs = np.asarray(dirs)
    dec, inc = np.radians(dirs[:, 0]), np.radians(dirs[:, 1])

    cos_inc = np.cos(inc)
    x, y, z = cos_inc * np.cos(dec), cos_inc * np.sin(dec), np.sin(inc)
    return np.stack((x, y, z), axis=1)

def spe2geo(vecs, az, hade):
    """
    rotates cartesian vectors (n,3) from specimen to geographic coordinates using given azimuth and hade
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
    Applies tilt correction to Cartesian vectors (n,3) using dip direction and dip.
    """
    rot = R.from_euler('y', dip, degrees=True) * R.from_euler('z', -(dipdir - 90), degrees=True)
    return rot.apply(vecs)
    
def collect_alt_dirs(vecs, coordinates, az, hade, dipdir, dip):
    """
    Computes the mean direction in all three coordinate systems (specimen, geographic, tectonic).
    Input: Cartesian vectors (n,3).
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
    Applies PCA to a set of Cartesian vectors (n,3), centering the data.
    """
    pca = PCA(n_components=3)  # make PCA object
    evecs = pca.fit(vecs).components_  # fit components and get eigenvectors (transforms/centers the data by removing mean)
    evals = np.sort(pca.explained_variance_)[::-1]  # get eigenvalues
    return evecs, evals, pca.mean_ 

def doPCA_anchored(vecs):
    """
    Applies PCA without centering the data.
    """
    pca = TruncatedSVD(n_components=3)  # make PCA object
    evecs = pca.fit(vecs).components_      # fit components and get eigenvectors (does not remove the mean to re-center the data)
    evals = np.sort(pca.explained_variance_)[::-1]  # get eigenvalues
    return evecs, evals 
               
def linefit(vecs, incl_origin=False, anchor=False):
    """
    Executes linear fitting via PCA on a set of cartesian vectors (n,3). Returns the first principal component, the associated MAD angle, 
    and a line segment for visualization.
    """
    if incl_origin: 
        vecs = np.concatenate([vecs, [[0, 0, 0]]])  # append the origin itself to the end of the data before doing PCA
    
    result = doPCA_anchored(vecs) if anchor else doPCA(vecs)
    evecs, evals, centroid = (result + (None,))[:3]  # unpack based on PCA method used
    
    v1 = evecs[0]  # get first principal component
    mad = np.degrees(np.arctan(np.sqrt((evals[1] + evals[2]) / evals[0]))) #get MAD angle   
    
    #check polarity of principal component (the polarity of which is arbitrary)
    if np.dot(v1, vecs[0] - vecs[-1]) < 0:
        v1 *= -1
        
    if anchor:
        v1_segment = [[0,0,0], v1]
    else:
        #get end-points of a line segment that is parallel to the principal component, and centered on the data mean (for plotting)
        scale = 0.5 * np.sqrt(evals[0] * len(vecs))    # re-scale the component to the original data scale
        v1_segment = [centroid - scale * v1, centroid + scale * v1]
    
    return v1, mad, v1_segment

def gcfit(vecs, normalize=False, constraints=False):
    """
    Executes planar fitting via PCA on a set of cartesian vectors (n, 3); returns the third principal component (normal to the plane), 
    the associated MAD angle, and a series of points tracing the circle (or a section of the circle if constraints are applied).
    """
    if normalize:          # normalize vector lengths if desired
        centered = vecs - np.mean(vecs, axis=0)
        vecs = centered / np.linalg.norm(centered, axis=1, keepdims=True)

    evecs, evals, centroid = doPCA(vecs)  # do PCA
    v3 = evecs[-1]  # get the smallest principal component
    mad = np.degrees(np.arctan(np.sqrt(evals[2]/evals[1] + evals[2]/evals[0])))  # get mad angle

    constraint_pts = [vecs[0], vecs[-1]] if constraints else []  # if constraints used, grab first and last points
    gc_segment = sample_gc(v3, constraint_pts, ang_step=0.5)  # get points along GC (or partial segment)
    
    return v3, mad, gc_segment

def sample_gc(vnorm, constraints, ang_step=0.5):
    """
    Samples points along a great circle normal to vnorm (n, 3); 
    if constraints are provided, only a section of the GC arc is sampled.
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
    Samples points along the arc of a great circle between two points (constraints).
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
    Performs PCA for all possible consecutive series of data (n>2),
    and assuming 1, 2, or 3 components to automatically find the best options.
    """
    fit_results = []

    # convert relevant columns of fspec to a numpy array for faster slicing
    points = np.column_stack((fspec['x'].values, fspec['y'].values, fspec['z'].values))

    # iterate through all combinations of series in fspec
    for i1 in range(len(fspec)):
        for j1 in range(i1+3, len(fspec)):
            range1 = points[i1:j1]
            _, mad1, _ = linefit(range1)       
            num_pts1 = j1-i1
            
            if comps <= 1:  # if assuming just one component
                score = -w*mad1 + (1-w)*num_pts1 - 1*p
                fit_results.append((i1, j1, num_pts1, mad1, score))
        
            else:  # otherwise look for a 2nd component that lies at higher range than first component
                for i2 in range(j1, len(fspec)):
                    for j2 in range(i2+3, len(fspec)):
                        range2 = points[i2:j2]
                        _, mad2, _ = linefit(range2) 
                        num_pts2 = j2-i2

                        if comps <= 2:  # if assuming just 2 components
                            avg_mad = (mad1+mad2)/2
                            score = -w*avg_mad + (1-w)*(num_pts1+num_pts2) - 2*p
                            fit_results.append(((i1, i2), (j1, j2), (num_pts1, num_pts2), (mad1, mad2), score))
                       
                        else:  # otherwise look for a 3rd component that lies at higher range than second component
                            for i3 in range(j2, len(fspec)):
                                for j3 in range(i3+3, len(fspec)):
                                    range3 = points[i3:j3]
                                    _, mad3, _ = linefit(range3) 
                                    num_pts3 = j3 - i3
                                    
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
    prev_params = None
    for i, obs in enumerate(observed):
        init_guess = [1] * len(v1s)
        constraints = []
        if prev_params is not None:
            constraints = [
                {'type': 'ineq', 'fun': lambda params, idx=idx: abs(prev_params[idx]) - abs(params[idx])}
                for idx in range(len(v1s))]

        result = minimize(cost_func, init_guess, args=(obs,), method='SLSQP', constraints=constraints)
        params = result.x
        coefficients.append(tuple(params))
        prev_params = params
        fitted_pt = lin_mod(params)
        fitted_pts.append(fitted_pt)
        
    return np.array(fitted_pts), coefficients
    

############## Mathematical / statistical operations ##############

def angle(v1, v2):
    """
    compute the angle between two cartesian vectors
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dprod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = np.clip(dprod / (norm_v1 * norm_v2), -1.0, 1.0)
    
    return np.arccos(cos_theta)

def fisher_mean(df, coordinates, flip='y', w_gcs='n'):
    """
    calculate fisher mean from dec/inc data in a dataframe
    """
    
    ldf = df[df['fit_type'] == 'line']
    gcdf = df[df['fit_type'] == 'plane']
    
    if len(ldf) <= 2:
        print ('not enough lines to compute Fisher mean')
        return

    if coordinates == 'specimen': 
        decs, incs = ldf['Ds'], ldf['Is']
        gcs = gcdf['gcpts_s']
    elif coordinates == 'geographic': 
        decs, incs = ldf['Dg'], ldf['Ig']
        gcs = gcdf['gcpts_g']
    elif coordinates == 'tectonic': 
        decs, incs = df['Dt'], df['It']
        gcs = gcdf['gcpts_t']

    DI = np.column_stack((decs, incs))

    if flip == 'y':
        normal, flipped = pmag.flip(DI)
        if len(flipped) != 0: 
            print (f'{len(flipped)} directions flipped')
            DI = normal + flipped
    
    fmean = ipmag.fisher_mean(di_block=DI)

    if w_gcs == 'n' or len(gcdf) == 0:
        return fmean, None
    else:
        return iterative_fisher_mean(DI, gcs)


def iterative_fisher_mean(DI, gcs):

    fmean = ipmag.fisher_mean(di_block=DI) # get initial mean
    fdir = [fmean['dec'], fmean['inc']]

    shift = 45
    while shift > 1.0:
        nearest = []
        for gc in gcs:
            angs = abs(np.array([pmag.angle(pt, fdir) for pt in gc]))
            minidx = np.argmin(angs)
            nearest.append(gc[minidx])

        new_DI = copy.deepcopy(DI).tolist()
        for pt in nearest:
            new_DI.append(pt)
        new_fmean = ipmag.fisher_mean(di_block=new_DI)
        new_fdir = [new_fmean['dec'], new_fmean['inc']]
        shift = pmag.angle(fdir, new_fdir)
        fdir = new_fdir
    
    return new_fmean, nearest


############## Output ##############

    
def update_saved_record(outfile, spec, fspec, lines, planes, coefficients, coordinates):
    """
    save new data to csv file (this will overwrite any existing entry with the same specimen name, component name and fit type (line or plane)
    """
    coefficients = np.array(coefficients) # make sure coefficients is a numpy array

    # determine if the specimen was thermal or AF demagnetized (the present formulation assigns anything with some TH steps as being 'thermally demagnetized')
    thermal = fspec['demag'].str.contains('TH').any()
    if thermal: demag = 'TH'
    else: demag = 'AF' # only 'AF' if there are no TH steps (more complex situations can be dealt with if needed)

    new_fits = []
    allfits = lines+planes # combine all lines and fits together
    for i in range(len(allfits)):
        fit_range = np.array(allfits[i][3])  # range of fitted data (lines or planes)
        treatment = np.array(fspec['treatment']).tolist()  # keep only the treatment steps from the filtered data
        
        # collect mean directional records in all three coordinate systems (spec, geo, tec)
        DI_s, DI_g, DI_t = collect_alt_dirs(allfits[i][4], coordinates, spec['az'][0], spec['hade'][0], spec['bed_dip_dir'][0], spec['bed_dip'][0])
        if allfits[i][0] == 'line':
            coeffs = coefficients[:, i].tolist()
            gcpts_s, gcpts_g, gcpts_t = None, None, None
        if allfits[i][0] == 'plane':
            coeffs = None
            gcpts_s, gcpts_g, gcpts_t = collect_alt_dirs(allfits[i][6], coordinates, spec['az'][0], spec['hade'][0], spec['bed_dip_dir'][0], spec['bed_dip'][0])
    
        # Serialize objects to strings
        srl_coeffs = base64.b64encode(pickle.dumps(coeffs)).decode('utf-8')
        srl_treatment = base64.b64encode(pickle.dumps(treatment)).decode('utf-8')
        srl_gcpts_s = base64.b64encode(pickle.dumps(gcpts_s)).decode('utf-8')
        srl_gcpts_g = base64.b64encode(pickle.dumps(gcpts_g)).decode('utf-8')
        srl_gcpts_t = base64.b64encode(pickle.dumps(gcpts_t)).decode('utf-8')
    
        new_fits.append({
            'specimen': spec['specimen'][i],
            'component': allfits[i][1],
            'fit_type': allfits[i][0],
            'n': len(allfits[i][2]),
            'from': fit_range[0],
            'to': fit_range[-1],
            'Ds': DI_s[0][0],
            'Is': DI_s[0][1],
            'Dg': DI_g[0][0],
            'Ig': DI_g[0][1],
            'Dt': DI_t[0][0],
            'It': DI_t[0][1],
            'mad': allfits[i][5],
            'demag': demag,
            'coefficients': srl_coeffs,
            'treatment': srl_treatment,
            'gcpts_s':  srl_gcpts_s,
            'gcpts_g':  srl_gcpts_g,
            'gcpts_t':  srl_gcpts_t
        })
    new_df = pd.DataFrame(new_fits)

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


def mean_decay(comp_dfs):

    interp_treatments = []
    interp_coefficients = []
    for df in comp_dfs:       
        ldf = df[df['fit_type'] == 'line']

        # do AF first
        AFdf = ldf[ldf['demag'] == 'AF']
        if len(AFdf) == 0:
            interp_treatments.append([])
            interp_coefficients.append([])
        else:
            treatments = AFdf['treatment'].reset_index(drop=True)
            coefficients = AFdf['coefficients'].reset_index(drop=True)
            treatments_flattened = np.concatenate(treatments.values)
            treat_steps = np.arange(0, treatments_flattened.max()+5, 5)
            for i in range(len(treatments)):
                linear_interp = interp1d(treatments[i], abs(coefficients[i]), kind='linear')
                int_coefficients = linear_interp(treat_steps)
            interp_treatments.append(treat_steps)
            interp_coefficients.append(int_coefficients)

        # do TH
        THdf = ldf[ldf['demag'] == 'TH']
        if len(THdf) == 0:
            interp_treatments.append([])
            interp_coefficients.append([])
        else:
            treatments = THdf['treatment'].reset_index(drop=True)
            coefficients = THdf['coefficients'].reset_index(drop=True)
            treatments_flattened = np.concatenate(treatments.values)
            treat_steps = np.arange(0, treatments_flattened.max()+5, 5)
            for i in range(len(treatments)):
                linear_interp = interp1d(treatments[i], abs(coefficients[i]), kind='linear')
                int_coefficients = linear_interp(treat_steps)
            interp_treatments.append(treat_steps)
            interp_coefficients.append(int_coefficients)

    return (interp_treatments, interp_coefficients)
        