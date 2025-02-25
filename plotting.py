import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
import processing as pro
import plotly.io as pio
pio.renderers.default = "iframe"

def get_scaled_axes(ax, xmin, xmax, ymin, ymax):
    """ ... """

    #ensure that both ranges include zero
    xmin, xmax = min(xmin, 0), max(xmax, 0)
    ymin, ymax = min(ymin, 0), max(ymax, 0)

    xmargin = (xmax-xmin)*0.05
    ymargin = (ymax-ymin)*0.05

    xmin2, xmax2 = xmin-xmargin, xmax+xmargin
    ymin2, ymax2 = ymin-ymargin, ymax+ymargin

    # find the larger range, get means and set new ranges
    max_range = max(abs(xmax2-xmin2), abs(ymax2-ymin2))
    xmean, ymean = (xmax2+xmin2) / 2, (ymax2+ymin2) / 2
    xrange = [xmean-max_range/2, xmean+max_range/2]
    yrange = [ymean-max_range/2, ymean+max_range/2]

    # get ticks
    tick_unit = max_range/5
    xticks_pos = np.arange(0, xrange[1], tick_unit)
    xticks_neg = np.arange(0, xrange[0], -tick_unit)
    xticks = np.concatenate((xticks_neg[::-1], xticks_pos[1:]))
    yticks_pos = np.arange(0, yrange[1], tick_unit)
    yticks_neg = np.arange(0, yrange[0], -tick_unit)
    yticks = np.concatenate((yticks_neg[::-1], yticks_pos[1:]))

    # apply to axes
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    return ax, tick_unit

def coord_vars(coordinates):
    """ ... """
    if coordinates == 'specimen':
        X, negX, Y, Z, negZ = 'X', '-X', 'Y', 'Z', '-Z'
    elif coordinates in ['geographic', 'tectonic']:
        X, negX, Y, Z, negZ = 'N', 'S', 'E', 'Down', 'Up'

    return X, negX, Y, Z, negZ
       
def proj_vars(projection, X, negX, Y, Z, negZ):
    """ ... """
    if projection=='N v. E/Dn':
        Hx, Hy = 'x2', 'x1'
        Vx, Vy = 'x3', 'x1'
        lHx, lHy = 1, 0
        lVx, lVy = 2, 0
        hxs, vxs, vys = 1, 1, 1
        xlab = f"{Y}, {Z}"
        ylab = f"{X}"

    elif projection=='N/Up v. E':
        Hx, Hy = 'x2', 'x1'
        Vx, Vy = 'x2', 'x3'  
        lHx, lHy = 1, 0
        lVx, lVy = 1, 2      
        hxs, vxs, vys = 1, 1, -1
        xlab = f"{Y}"
        ylab = f"{X}  {negZ}"

    elif projection=='E v. S/Dn':
        Hx, Hy = 'x1', 'x2'
        Vx, Vy = 'x3', 'x2'  
        lHx, lHy = 0, 1
        lVx, lVy = 2, 1      
        hxs, vxs, vys = -1, 1, 1
        xlab = f"{negX}, {Z}"
        ylab = f"{Y}"

    elif projection=='E/Up v. S':
        Hx, Hy = 'x1', 'x2'
        Vx, Vy = 'x1', 'x3'  
        lHx, lHy = 0, 1
        lVx, lVy = 0, 2   
        hxs, vxs, vys = -1, -1, -1
        xlab = f"{negX}"
        ylab = f"{Y}  {negZ}"

    else: print ('projection not found')

    return Hx, Hy, Vx, Vy, lHx, lHy, lVx, lVy, hxs, vxs, vys, xlab, ylab

    
############# specimen-level plots #############

def zij_plt(coordinates, projection, data, filtered, lines, planes):
    """
    Makes an orthogonal vector (e.g. Zijderveld) plot together with a stereoplot and a remanence decay plot.
    """

    X, negX, Y, Z, negZ = coord_vars(coordinates)
    Hx, Hy, Vx, Vy, lHx, lHy, lVx, lVy, hxs, vxs, vys, xlab, ylab = proj_vars(projection, X, negX, Y, Z, negZ)
        
    if lines:
        lnames, lpts, ldirs, lmads, lsegs, lcolors = zip(*[(line[1], line[2], line[5], line[6], line[7], line[9]) for line in lines])
    if planes:
        gcnames, gcpts, ndirs, gcmads, gcsegs, gccolors = zip(*[(plane[1], plane[2], plane[5], plane[6], plane[7], plane[9]) for plane in planes])

    fig = plt.figure(constrained_layout=True, figsize=(14.5,8))  
    annotations, annotation_positions = [], [] 
    
    gs = fig.add_gridspec(3, 2)
    
    ax1 = fig.add_subplot(gs[:3, 0]) # make the zijderveld diagram
      
    ax1.plot(data[Hx]*hxs, data[Hy], marker='o', color='k', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0) # plot all the data
    ax1.plot(data[Vx]*vxs, data[Vy]*vys, marker='o', color='k',  markerfacecolor='white', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
    ax1.plot(filtered[Hx]*hxs, filtered[Hy], marker='o', color='k', linewidth=0.5,  label='horizontal')  # plot the filtered subset of data
    ax1.plot(filtered[Vx]*vxs, filtered[Vy]*vys, marker='o', color='k', linewidth=0.5, markerfacecolor='white', label='vertical')

    xmin, xmax = min(min(data[Hx]*hxs), min(data[Vx]*vxs)), max(max(data[Hx]*hxs), max(data[Vx]*vxs))
    ymin, ymax = min(min(data[Hy]), min(data[Vy]*vys)), max(max(data[Hy]), max(data[Vy]*vys))
    ax1, tick_unit = get_scaled_axes(ax1, xmin, xmax, ymin, ymax)
    
    if lines:
        for i in range(len(lines)):
            ax1.plot(lpts[i][Hx]*hxs, lpts[i][Hy], marker='o', color=lcolors[i], linestyle='none', label=f'comp. {lnames[i]}') # plot the fitted points
            ax1.plot(lpts[i][Vx]*vxs, lpts[i][Vy]*vys, marker='o', markerfacecolor='none', markeredgecolor=lcolors[i], markeredgewidth=1.5, linestyle='none')
            lstart, lend = lsegs[i][0], lsegs[i][1]
            ax1.plot([lstart[lHx]*hxs, lend[lHx]*hxs], [lstart[lHy], lend[lHy]], color=lcolors[i], lw=3, alpha=0.5) # plot the principal component
            ax1.plot([lstart[lVx]*vxs, lend[lVx]*vxs], [lstart[lVy]*vys, lend[lVy]*vys], color=lcolors[i], lw=3, alpha=0.5)

    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    x_offset = 0.015 * (x_lim[1] - x_lim[0]) 
    y_offset = 0.01 * (y_lim[1] - y_lim[0])
    for i, (x, y) in enumerate(zip(data[Hx]*hxs, data[Hy])):
        ann = ax1.text(x + y_offset, y + x_offset, str(i), fontsize=7, color='k', ha='center', va='center')  # plot axes labels
        annotations.append(ann)
        annotation_positions.append((x, y))
    for i, (x, y) in enumerate(zip(data[Vx]*vxs, data[Vy]*vys)):
        ann = ax1.text(x + y_offset, y + x_offset, str(i), fontsize=7, color='grey', ha='center', va='center')
        annotations.append(ann)
        annotation_positions.append((x, y))
    ax1.set_xlabel(f"{xlab}", loc='right')
    ax1.set_ylabel(f"{ylab}", rotation='horizontal', ha='left', va='center', x=0.5, y=0.99)

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    xmax = ax1.get_xlim()[1]
    ax1.text(xmax, tick_unit*0.1, f"ticks: {tick_unit:.1e} A/m", ha='right', va='center', fontsize=9)
    
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()
    if labels:  # Only call legend if there are labels
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.53, 1))
    
    ax2 = fig.add_subplot(gs[:2, 1])  # plot stereonet
    plot_net()
    plot_di(dec=[x for x in data['dec']], inc=[x for x in data['inc']], marker=None, connect_points=True, line_color='grey', line_width=0.5, alpha=0.50) # plot lines connecting all data
    plot_di(dec=[x for x in data['dec']], inc=[x for x in data['inc']], markersize=50, alpha=0.20) # plot all the data
    plot_di(dec=[x for x in filtered['dec']], inc=[x for x in filtered['inc']], markersize=50)     # plot the filtered data
    plot_di(dec=filtered['dec'].iloc[0], inc=filtered['inc'].iloc[0], marker='+', markersize=200)

    if lines:
        for i in range(len(lines)):
            ldir = pro.to_sph([ldirs[i]])
            plot_di_mean(dec=ldir[0][0], inc=ldir[0][1], a95=lmads[i], marker='*', markersize=75, color=lcolors[i], alpha=0.75, label=f'comp. {lnames[i]}') # plot the linear fits

    if planes:
        for i in range(len(planes)):
            ndir = pro.to_sph([ndirs[i]])
            plot_di_mean(dec=ndir[0][0], inc=ndir[0][1], a95=gcmads[i], marker='D', markersize=25, color=gccolors[i], label=f'GC {gcnames[i]}') # plot the gc fits
            gcsegdirs = pro.to_sph(gcsegs[i])
            dn_pts = gcsegdirs[gcsegdirs[:, 1] > 0]
            up_pts = gcsegdirs[gcsegdirs[:, 1] < 0]
            plot_di(dec=[x for x in dn_pts[:,0]], inc=[x for x in dn_pts[:,1]], markersize=0.2, alpha=0.75, color=gccolors[i])
            plot_di(dec=[x for x in up_pts[:,0]], inc=[x for x in up_pts[:,1]], markersize=0.2, alpha=0.25, color=gccolors[i])

    handles, labels = ax2.get_legend_handles_labels()
    if labels:  # Only call legend if there are labels
        ax2.legend(loc='upper right', fontsize=12, markerscale=1.25)
    ax2.text(0.05, 0.20, f"{data['coordinates'][0]}\ncoordinates", transform=ax2.transAxes, fontsize=12, ha='left', va='top', color='black')
    
    ax3 = fig.add_subplot(gs[2:3, 1])   # plot the remanence decay plot
    mnorm = filtered['res'].max()
    ax3.plot(data['treatment'], data['res']/mnorm, marker='o', markersize=4, color='k', alpha=0.25)
    ax3.plot(filtered['treatment'], filtered['res']/mnorm, marker='o', markersize=4, color='k')
    if lines:
        for i in range(len(lines)):
            ax3.plot(lpts[i]['treatment'], lpts[i]['res']/mnorm, marker='o', linestyle='None', markersize=5, color=lcolors[i])
    if planes:
        for i in range(len(planes)):
            ax3.plot([gcpts[i].iloc[0]['treatment'], gcpts[i].iloc[-1]['treatment']], [0.5, 0.5], linestyle='--', color=gccolors[i], alpha=0.5, zorder=0, label=f'GC {gcnames[i]}')
    ax3.tick_params(axis='y', which='both', direction='in', length=6, labelleft=False, labelbottom=False)
    ax3.grid(which='both', axis='y', linestyle='--', color='gray', linewidth=0.5)
    ax3.set_xlabel('Treatment')
    ax3.set_ylabel('Normalized intensity')
    handles, labels = ax3.get_legend_handles_labels()
    if labels:  # only call legend if there are labels
        ax3.legend(loc='upper right')

    return fig


def linzij_plt(coordinates, projection, data, filtered, lines, fitted, coefficients, coefficients_norm):
    """ ... """

    X, negX, Y, Z, negZ = coord_vars(coordinates)
    Hx, Hy, Vx, Vy, lHx, lHy, lVx, lVy, hxs, vxs, vys, xlab, ylab = proj_vars(projection, X, negX, Y, Z, negZ)
    
    lnames, ldirs, lcolors = [], [], []
    for i in range(len(lines)): 
        lnames.append(lines[i][1])
        ldirs.append(lines[i][5])
        lcolors.append(lines[i][9])
    
    fig = plt.figure(constrained_layout=True, figsize=(14.5,8))
    gs = fig.add_gridspec(3, 2)
    
    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot(data[Hx]*hxs, data[Hy], marker='o', color='k', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
    ax1.plot(data[Vx]*vxs, data[Vy]*vys, marker='o', color='k',  markerfacecolor='white', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
    ax1.plot(filtered[Hx]*hxs, filtered[Hy], marker='o', color='k', linewidth=0.5,  label='horizontal')
    ax1.plot(filtered[Vx]*vxs, filtered[Vy]*vys, marker='o', color='k', linewidth=0.5, markerfacecolor='white', label='vertical')
    ax1.plot(fitted[:,lHx]*hxs, fitted[:,lHy], marker='o', markersize=3, color='purple', linewidth=0.25, linestyle='dashed', label='modelled')
    ax1.plot(fitted[:,lVx]*vxs, fitted[:,lVy]*vys, marker='o', markersize=3, color='purple', linewidth=0.25, linestyle='dashed', markerfacecolor='white')

    ax1.set_xlabel(f"{xlab}", loc='right')
    ax1.set_ylabel(f"{ylab}", rotation='horizontal', ha='left', va='center', x=0.5, y=0.99)
        
    xmin, xmax = min(min(data[Hx]*hxs), min(data[Vx]*vxs)), max(max(data[Hx]*hxs), max(data[Vx]*vxs))
    ymin, ymax = min(min(data[Hy]), min(data[Vy]*vys)), max(max(data[Hy]), max(data[Vy]*vys))
    ax1, tick_unit = get_scaled_axes(ax1, xmin, xmax, ymin, ymax)
    
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    xmax = ax1.get_xlim()[1]
    ax1.text(xmax, tick_unit*0.1, f"ticks: {tick_unit:.1e} A/m", ha='right', va='center', fontsize=9)
    
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[:2, 1])

    ldirs_scaled = [ldir * a for ldir, a in zip(ldirs, coefficients[0])]
    arrow_start = np.zeros(3, dtype=float)
    
    ax2.plot(data[Hx]*hxs, data[Hy], marker='o', color='k', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
    ax2.plot(data[Vx]*vxs, data[Vy]*vys, marker='o', color='k',  markerfacecolor='white', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
    ax2.plot(filtered[Hx]*hxs, filtered[Hy], marker='o', color='k', linewidth=0.5)
    ax2.plot(filtered[Vx]*vxs, filtered[Vy]*vys, marker='o', color='k', linewidth=0.5, markerfacecolor='white')        
    
    for i in range(len(ldirs)-1,-1,-1):
        ax2.quiver(arrow_start[lHx]*hxs, arrow_start[lHy], ldirs_scaled[i][lHx]*hxs, ldirs_scaled[i][lHy], angles='xy', scale_units='xy', scale=1, color=lcolors[i], alpha=0.5, label=f'comp. {lnames[i]}')
        ax2.quiver(arrow_start[lVx]*vxs, arrow_start[lVy]*vys, ldirs_scaled[i][lVx]*vxs, ldirs_scaled[i][lVy]*vys, angles='xy', scale_units='xy', scale=1, color=lcolors[i], alpha=0.5)
        arrow_start += ldirs_scaled[i][:3]

    ax2.set_xlabel(f"{xlab}", loc='right')
    ax2.set_ylabel(f"{ylab}", rotation='horizontal', ha='left', va='center', x=0.5, y=0.99)

    xmin, xmax = min(min(data[Hx]*hxs), min(data[Vx]*vxs)), max(max(data[Hx]*hxs), max(data[Vx]*vxs))
    ymin, ymax = min(min(data[Hy]), min(data[Vy]*vys)), max(max(data[Hy]), max(data[Vy]*vys))
    ax2, tick_unit = get_scaled_axes(ax2, xmin, xmax, ymin, ymax)
    
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    xmax = ax2.get_xlim()[1]
    ax2.text(xmax, tick_unit*0.1, f"ticks: {tick_unit:.1e} A/m", ha='right', va='center', fontsize=9)
    
    ax2.spines['left'].set_position('zero')
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
        
    ax3 = fig.add_subplot(gs[2:3, 1])
    steps = filtered['treatment']
    coeff_norm_arr = np.array(coefficients_norm)
    for i in range(len(ldirs)):
        ax3.plot(steps, abs(coeff_norm_arr[:,i]), color=lcolors[i], alpha=0.75, label=f'comp. {lnames[i]}')
    ax3.tick_params(axis='y', which='both', direction='in', length=6, labelbottom=False)
    ax3.grid(which='both', axis='y', linestyle='--', color='gray', linewidth=0.5)
    ax3.set_xlabel('Treatment')
    ax3.set_ylabel('Component contribution')
    ax3.legend(loc='upper right', fontsize=12, markerscale=1);

    ax3_twin = ax3.twinx()
    ax3_twin.plot(data['treatment'], data['res']/filtered['res'].max(), linestyle='--', color='k', alpha=0.25)
    ax3_twin.plot(filtered['treatment'], filtered['res']/filtered['res'].max(), linestyle='--', color='k')
    ax3_twin.set_ylabel('Normalized intensity')
    ax3_twin.set_yticks([])

    return fig

def interactive_zij_plt(coordinates, data, filtered, lines, planes, show_lines='y', show_planes='n'):

    if coordinates == 'specimen':
        X, Y, Z, negZ = 'X', 'Y', 'Z', '-Z'
    if coordinates == 'geographic' or coordinates == 'tectonic':
        X, Y, Z, negZ = 'N', 'E', 'Down', 'Up'
        
    fig = pgo.Figure()   # create figure

    # add the origin
    fig.add_trace(pgo.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='orange', opacity=1), showlegend=False))
    # add raw observations
    fig.add_trace(pgo.Scatter3d(x=data['x2'], y=data['x1'], z=data['x3'], mode='markers', marker=dict(size=4, color='grey', opacity=0.25), showlegend=False))
    # add filtered data
    fig.add_trace(pgo.Scatter3d(x=filtered['x2'], y=filtered['x1'], z=filtered['x3'], mode='markers', marker=dict(size=4, color='black', opacity=0.8), showlegend=False))

    if show_lines == 'y':
    # add PCA segments
        for i in range(len(lines)):
            lseg = lines[i][7]
            color = lines[i][9]
            fig.add_trace(pgo.Scatter3d(x=[lseg[0][1], lseg[1][1]], y=[lseg[0][0], lseg[1][0]], z=[lseg[0][2], lseg[1][2]], mode="lines",
                line=dict(width=5, color=color), opacity=0.5, name=f'Comp. {lines[i][1]}'))

    if show_planes == 'y':
    # add GC planes
        for i in range(len(planes)):
            nvec = planes[i][5]
            xmax = data['x2'].abs().max()
            ymax = data['x1'].abs().max()
            zmax = data['x3'].abs().max()
            xrange = np.linspace(-xmax, xmax, 5)
            yrange = np.linspace(-ymax, ymax, 5)
            xpl, ypl = np.meshgrid(xrange, yrange)
            if nvec[2] != 0:
                zpl = (-nvec[1] * xpl - nvec[0] * ypl) / nvec[2]
            else:
                return  # skipped if plane is perfectly horizontal
            fig.add_trace(pgo.Surface(x=xpl, y=ypl, z=zpl, colorscale=[[0, 'purple'], [1, 'purple']], opacity=0.2, showscale=False))

    xarr = data['x2'].values
    xarr = np.append(xarr, 0)
    yarr = data['x1'].values
    yarr = np.append(yarr, 0)
    zarr = data['x3'].values
    zarr = np.append(zarr, 0)

    x_span = xarr.max()-xarr.min()
    y_span = yarr.max()-yarr.min()
    z_span = zarr.max()-zarr.min()
    max_span = max(x_span, y_span, z_span)

    x_mid = x_span / 2
    y_mid = y_span / 2
    z_mid = z_span / 2
    
    x_range = [x_mid - max_span / 2, x_mid + max_span / 2]
    y_range = [y_mid - max_span / 2, y_mid + max_span / 2]
    z_range = [z_mid + max_span / 2, z_mid - max_span / 2]
    
    fig.update_layout(scene=dict(xaxis_title=Y, yaxis_title=X, zaxis_title=Z, xaxis=dict(range=x_range), yaxis=dict(range=y_range), zaxis=dict(range=z_range),
                                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=1)),
                                 width=1200, height=800, margin=dict(l=20, r=20, t=15, b=15), legend=dict(font=dict(size=14), x=0.90,  y=0.90))
    
    fig.show()


############# site-/study-level plots #############

def stat_stereo(coordinates, selected, mean_data, show_lines=False, show_planes=False, show_normals=False, show_mad=False):

    if coordinates == 'specimen': 
        dec, inc = 'Ds', 'Is'
        gcpts = 'gcs'
    elif coordinates == 'geographic': 
        dec, inc = 'Dg', 'Ig'
        gcpts = 'gcg'
    elif coordinates == 'tectonic': 
        dec, inc = 'Dt', 'It'
        gcpts = 'gct'

    c_labels, comp_dfs, lin_means, gc_ints, mix_means, gc_endpts, show_samples, c_colors = [], [], [], [], [], [], [], []
    for i in range(len(mean_data)): 
        c_labels.append(mean_data[i][0])
        comp_dfs.append(mean_data[i][1])
        lin_means.append(mean_data[i][2])
        gc_ints.append(mean_data[i][3])
        mix_means.append(mean_data[i][4])
        gc_endpts.append(mean_data[i][5])
        show_samples.append(mean_data[i][6])
        c_colors.append(mean_data[i][7])

    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    plot_net()

    # plot background data...
    if show_lines:
        lines = selected[selected['fit_type']=='line']
        if show_mad:
            for i, line in lines.iterrows():
                plot_di_mean(dec=line[dec], inc=line[inc], a95=line['mad'], marker=None, color='k', alpha=0.1)
        plot_di(dec=[x for x in lines[dec]], inc=[x for x in lines[inc]], markersize=20, color='k', alpha=0.3)

    if show_planes:
        planes = selected[selected['fit_type']=='plane']
        gcs = planes[gcpts]
        for gc in gcs:
            plot_gc(dec=[x for x in gc[:,0]], inc=[x for x in gc[:,1]], color='k', up_linestyle='--', linewidth=1.25, alpha=0.3)

    if show_normals:
        planes = selected[selected['fit_type']=='plane']
        if show_mad:
            for i, plane in planes.iterrows():
                plot_di_mean(dec=plane[dec], inc=plane[inc], a95=plane['mad'], marker=None, color='k', alpha=0.1)    
        plot_di(dec=[x for x in planes[dec]], inc=[x for x in planes[inc]], marker='D', markersize=20, color='k', alpha=0.3)

    # plot data contributing to mean calculation ...
    for i in range(len(mean_data)):
        if show_samples[i]:
            df = comp_dfs[i]
            if lin_means[i] or mix_means[i]:
                lines = df[df['fit_type']=='line']
                plot_di(dec=[x for x in lines[dec]], inc=[x for x in lines[inc]], markersize=20, color=c_colors[i], alpha=0.3)

            if gc_ints[i] or mix_means[i]:
                planes = df[df['fit_type']=='plane']
                gcs = planes[gcpts]
                for gc in gcs:
                    plot_gc(dec=[x for x in gc[:,0]], inc=[x for x in gc[:,1]], color=c_colors[i], up_linestyle='--', linewidth=1.25, alpha=0.3)

    # plot means 
    for i in range(len(mean_data)):  # looping again to ensure that the means plot on top of all the constituent data from previous loop
        # plot means
        if lin_means[i]:
            plot_di_mean(dec=lin_means[i]['dec'], inc=lin_means[i]['inc'], a95=lin_means[i]['alpha95'], marker='*', markersize=100, 
                         color=c_colors[i], alpha=1, label=f"comp.{c_labels[i]} fisher mean")
        if gc_ints[i]:
            plot_di_mean(dec=gc_ints[i]['dec'], inc=gc_ints[i]['inc'], a95=gc_ints[i]['mad'], marker='^', markersize=100, linestyle='--', 
                         color=c_colors[i], alpha=1, label=f"comp.{c_labels[i]} GC intersection")
        if mix_means[i]:
            plot_di_mean(dec=mix_means[i]['dec'], inc=mix_means[i]['inc'], a95=mix_means[i]['alpha95'], marker='P', markersize=100, 
                         color=c_colors[i], alpha=1, label=f"comp.{c_labels[i]} mixed Fisher mean")
        if gc_endpts[i]:
            gc_array = np.array(gc_endpts[i])
            plot_di(dec=gc_array[:,0], inc=gc_array[:,1], marker='s', markersize=20, color=c_colors[i], alpha=0.3)

    if any(label is not None for label in [artist.get_label() for artist in plt.gca().get_legend_handles_labels()[0]]):
        plt.legend()

    plt.tight_layout()
    plt.show()

    

def decay_spectra(components, df, mean_treatments, mean_coefficients, mean_dMdD, show_dMdD=False, AF_log=False):
 
    n = len(components)
    fig, axes = plt.subplots(n, 2, figsize=(18, 3.5 * n))
    axes = axes.flatten()

    k = 0
    for i in range(n):
        comp = components[i]
        ldf = df[(df['component'] == comp) & (df['fit_type'] == 'line')]
        
        for j in range(2):
            ax = axes[k]
            ax_twin = ax.twinx()
            
            if j == 0:
                AF_ldf = ldf[ldf['demag'] == 'AF']
                for treatment, coefficients in zip(AF_ldf['treatment'], AF_ldf['coefficients']):
                    ax.plot(treatment, abs(coefficients), color='blue', alpha=0.5)
                if mean_treatments[k] is not None and mean_coefficients[k] is not None and len(mean_treatments[k]) > 0 and len(mean_coefficients[k]) > 0:
                    ax.plot(mean_treatments[k], mean_coefficients[k], color='darkblue', linewidth=2, label=f'comp. {comp} mean decay')
                    if show_dMdD:
                        ax_twin.plot(mean_treatments[k], mean_dMdD[k], color='darkblue', linestyle='--', linewidth=1, alpha=0.5, label=f'comp. {comp} dM/dD')
                if AF_log: ax.set_yscale('log')
                ax.set_ylabel('remanent contribution')
                if show_dMdD:
                    ax_twin.set_ylabel('dM/dD')
                    ax_twin.invert_yaxis()
                    ax_twin.set_yticks([])
                    ax_twin.set_yticklabels([])
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = ax_twin.get_legend_handles_labels()
                    ax.legend(lines + lines2, labels + labels2, loc='upper right')
                else: ax.legend(loc='upper right')
                
                if i == 0:
                    ax.set_title('AF demagnetization spectra')
                if i == n-1:
                    ax.set_xlabel('treatment (mT)')
                else: ax.set_xticks([])
                        
            else:
                TH_ldf = ldf[ldf['demag'] == 'TH']
                for treatment, coefficients in zip(TH_ldf['treatment'], TH_ldf['coefficients']):
                    ax.plot(treatment, abs(coefficients), color='red', alpha=0.5)
                if mean_treatments[k] is not None and mean_coefficients[k] is not None and len(mean_treatments[k]) > 0 and len(mean_coefficients[k]) > 0:
                    ax.plot(mean_treatments[k], mean_coefficients[k], color='darkred', linewidth=2, label=f'comp. {comp} mean deacy')
                    if show_dMdD:
                        ax_twin.plot(mean_treatments[k], mean_dMdD[k], color='darkred', linestyle='--', linewidth=1, alpha=0.5, label=f'comp. {comp} dM/dD')
                if show_dMdD:
                    ax_twin.set_ylabel('dM/dD')
                    ax_twin.invert_yaxis()
                    ax_twin.set_yticks([])
                    ax_twin.set_yticklabels([])
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = ax_twin.get_legend_handles_labels()
                    ax.legend(lines + lines2, labels + labels2, loc='upper right')
                else: ax.legend(loc='upper right')
                
                if i == 0:
                    ax.set_title('Thermal demagnetization spectra')
                if i == n-1:
                    ax.set_xlabel('treatment (deg.)')
                else: ax.set_xticks([])
            k+=1
              
    plt.show()

############# things stripped out of pmagpy that need to be revisited #############

def dimap(dec, inc):
    """
    Map declination and inclination directions to x, y pairs in equal-area projection.
    """
    dirs = np.column_stack((dec, inc))  # Stack declination and inclination if necessary
    cart_coords = pro.to_car(dirs)
    R = np.sqrt(1 - np.abs(cart_coords[:, 2])) / np.sqrt(cart_coords[:, 0]**2 + cart_coords[:, 1]**2)
    return cart_coords[:, 1] * R, cart_coords[:, 0] * R

def plot_net(fignum=None, tick_spacing=10):
    """
    Draws circle and tick marks for equal area projection.
    """
    if fignum is not None:
        plt.figure(num=fignum)
        plt.clf()

    plt.axis("off")
    Dcirc = np.arange(0, 361.)
    Icirc = np.zeros(361)
    XY = [dimap(D, I) for D, I in zip(Dcirc, Icirc)]
    Xcirc, Ycirc = zip(*XY)
    plt.plot(Xcirc, Ycirc, 'k')

    def plot_ticks(longitude, tick_range, marker, size):
        XY = [dimap(longitude, I) for I in range(tick_range[0], tick_range[1], tick_spacing)]
        Xsym, Ysym = zip(*XY)
        plt.scatter(Xsym, Ysym, color='black', marker=marker, s=size)

    plot_ticks(0., (tick_spacing, 100), '_', 10)
    plot_ticks(90., (tick_spacing, 100), '|', 10)
    plot_ticks(180., (tick_spacing, 90), '_', 10)
    plot_ticks(270., (tick_spacing, 90), '|', 10)

    for D in range(0, 360, tick_spacing):
        Xtick, Ytick = zip(*[dimap(D, I) for I in range(4)])
        plt.plot(Xtick, Ytick, 'k')

    plt.axis("equal")
    plt.axis((-1.05, 1.05, -1.05, 1.05))

def plot_di(dec, inc, color='k', marker='o', markersize=20, legend='no', label='', title=None, edge=None, alpha=1, zorder=2, connect_points=False, line_color=None, line_style='-', line_width=1):
    """
    Plot declination, inclination data on an equal area plot.
    """
    if np.isscalar(dec):
        dec = [dec]
        inc = [inc]

    # Define a consistent color list based on the input color
    if isinstance(color, str):
        color_list = [color] * len(dec)
    elif len(color) == len(dec):
        color_list = color
    else:
        color_list = [color] * len(dec)
    
    X_down, X_up, Y_down, Y_up = [], [], [], []
    color_up, color_dn = [], []
    all_X, all_Y = [], []

    for d, i, c in zip(dec, inc, color_list):
        X, Y = dimap(d, i)
        all_X.append(X)
        all_Y.append(Y)
            
        if i >= 0:
            X_down.append(X)
            Y_down.append(Y)
            color_dn.append(c)
        else:
            X_up.append(X)
            Y_up.append(Y)
            color_up.append(c)

    if marker:
        if X_up: 
            plt.scatter(X_up, Y_up, facecolors='none', edgecolors=color_up, s=markersize, marker=marker, label=label, alpha=alpha, zorder=zorder)
        if X_down:
            plt.scatter(X_down, Y_down, facecolors=color_dn, edgecolors=edge, s=markersize, marker=marker, label=label, alpha=alpha, zorder=zorder)

    if connect_points and len(all_X) > 1:
        plt.plot(all_X, all_Y, color=line_color if line_color else color, linestyle=line_style, linewidth=line_width, alpha=alpha, zorder=zorder)

    if legend == 'yes': plt.legend(loc=2)
    if title: plt.title(title)

    plt.tight_layout()

def plot_di_mean(dec, inc, a95, color='k', marker='o', markersize=20, linewidth=1, linestyle='-', alpha=1, label='', legend='no', zorder=2):
    """
    Plot a mean direction with alpha_95 ellipse.
    """
    DI_dimap = dimap(dec, inc)
    if marker:
        plt.scatter(DI_dimap[0], DI_dimap[1], edgecolors=color, facecolors='white' if inc < 0 else color, marker=marker, s=markersize, label=label, alpha=alpha, zorder=zorder)

    # Get the circle points corresponding to a95 and plot them
    Da95, Ia95 = circ(dec, inc, a95)
    Xcirc, Ycirc = zip(*[dimap(D, I) for D, I in zip(Da95, Ia95)])
    plt.plot(Xcirc, Ycirc, c=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

    plt.tight_layout()


def plot_gc(dec, inc, color='k', up_color=None, dn_color=None, linestyle='-', up_linestyle=None, dn_linestyle=None, linewidth=1, up_linewidth=None, dn_linewidth=None, 
            alpha=1, up_alpha=None, dn_alpha=None, legend='no', label='', title=None, zorder=2):
    """
    Plot declination, inclination data along a great circle or great circle arc with lines
    """
    
    dirs = np.column_stack((dec, inc))

    # split into upper and lower hemisphere arrays
    upper = dirs[dirs[:, 1] < 0]
    lower = dirs[dirs[:, 1] >= 0]

    # sort the points in each arc
    upper_sorted = sort_gc_pts(upper)
    lower_sorted = sort_gc_pts(lower)
    
    # convert to X,Y stereonet coordinates    
    X_dn, X_up, Y_dn, Y_up = [], [], [], []

    for pt in upper_sorted:
        X, Y = dimap(pt[0], pt[1])
        X_up.append(X)
        Y_up.append(Y)

    for pt in lower_sorted:
        X, Y = dimap(pt[0], pt[1])
        X_dn.append(X)
        Y_dn.append(Y)
    
    if len(X_up) > 1:
        plt.plot(X_up, Y_up, color=up_color if up_color else color, linestyle=up_linestyle if up_linestyle else linestyle, linewidth=up_linewidth if up_linewidth else linewidth, 
                 alpha=up_alpha if up_alpha else alpha, zorder=zorder)
    if len(X_dn) > 1:
        plt.plot(X_dn, Y_dn, color=dn_color if dn_color else color, linestyle=dn_linestlye if dn_linestyle else linestyle, linewidth=dn_linewidth if dn_linewidth else linewidth, 
                 alpha=dn_alpha if dn_alpha else alpha, zorder=zorder)

    if legend == 'yes': plt.legend(loc=2)
    if title: plt.title(title)

    plt.tight_layout()
    

def sort_gc_pts(dirs):

    # get lowest inclination value
    incs = abs(dirs[:, 1])
    min_idx = np.argmin(incs)
    min_dir = dirs[min_idx]
    min_car = pro.to_car([min_dir])

    # now convert the whole array to Cartesian coordinates
    cars = pro.to_car(dirs)
    angs = np.array([pro.angle(min_car[0], v) for v in cars])
    
    # get sorted indices and sort dirs
    sorted_indices = np.argsort(angs)
    sorted_dirs = dirs[sorted_indices]
    
    return sorted_dirs


def circ(dec, inc, alpha, npts=201):
    """
    Calculates points on a circle about declination and inclination with angle alpha.
    """
    # Convert inputs to radians
    dec, inc, alpha = np.radians([dec, inc, alpha])

    # Compute reference declination and inclination
    dec1 = dec + np.pi / 2
    dip1 = inc - np.sign(inc) * (np.pi / 2)

    # Construct transformation matrix
    t = np.array([
        [np.cos(dec1), np.cos(dec) * np.cos(dip1), np.cos(dec) * np.cos(inc)],
        [np.sin(dec1), np.sin(dec) * np.cos(dip1), np.sin(dec) * np.cos(inc)],
        [0, np.sin(dip1), np.sin(inc)]
    ])

    # Generate circle points
    psi = np.linspace(0, 2 * np.pi, npts)  # Sweep angle from 0 to 180°
    v = np.column_stack([
        np.sin(alpha) * np.cos(psi),
        np.sin(alpha) * np.sin(psi),
        np.sqrt(1 - np.sin(alpha)**2) * np.ones(npts)
    ])

    # Apply transformation to all points
    elli = v @ t.T

    # Convert resulting points back to declination and inclination
    D_out, I_out = zip(*pro.to_sph(elli))  
    
    return list(D_out), list(I_out)    