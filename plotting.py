import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
import processing as pro

############# specimen-level plots #############

def zij_plt(coordinates, projection, data, filtered, lines, planes):
    """
    Makes an orthogonal vector (e.g. Zijderveld) plot together with a stereoplot and a remanence decay plot.
    """
    if coordinates == 'specimen':
        X, Y, Z, negZ = 'X', 'Y', 'Z', '-Z'
    elif coordinates in ['geographic', 'tectonic']:
        X, Y, Z, negZ = 'N', 'E', 'Down', 'Up'
        
    if not lines:
        lnames, lpts, ldirs, lmads, lsegs, lcolors = [], [], [], [], [], []
    else:
        lnames, lpts, ldirs, lmads, lsegs, lcolors = zip(*[(line[1], line[2], line[5], line[6], line[7], line[9]) for line in lines])
    if not planes:
        gcnames, gcpts, ndirs, gcmads, gcsegs, gccolors = [], [], [], [], [], []
    else:
        gcnames, gcpts, ndirs, gcmads, gcsegs, gccolors = zip(*[(plane[1], plane[2], plane[5], plane[6], plane[7], plane[9]) for plane in planes])

    fig = plt.figure(constrained_layout=True, figsize=(16,8))
    gs = fig.add_gridspec(3, 2)
    
    ax1 = fig.add_subplot(gs[:3, 0]) # make the zijderveld diagram
    
    if projection==1:  
        ax1.plot(data['x2'], data['x1'], marker='o', color='k', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0) # plot all the data
        ax1.plot(data['x3'], data['x1'], marker='o', color='k',  markerfacecolor='white', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax1.plot(filtered['x2'], filtered['x1'], marker='o', color='k', linewidth=0.5,  label='horizontal')  # plot the filtered subset of data
        ax1.plot(filtered['x3'], filtered['x1'], marker='o', color='k', linewidth=0.5, markerfacecolor='white', label='vertical')

        for i in range(len(lines)):
            ax1.plot(lpts[i]['x2'], lpts[i]['x1'], marker='o', color=lcolors[i], linestyle='none', label=f'comp. {lnames[i]}') # plot the fitted points
            ax1.plot(lpts[i]['x3'], lpts[i]['x1'], marker='o', markerfacecolor='none', markeredgecolor=lcolors[i], markeredgewidth=1.5, linestyle='none')
            lstart = lsegs[i][0]
            lend = lsegs[i][1]
            ax1.plot([lstart[1], lend[1]], [lstart[0], lend[0]], color=lcolors[i], lw=3, alpha=0.5) # plot the principal component
            ax1.plot([lstart[2], lend[2]], [lstart[0], lend[0]], color=lcolors[i], lw=3, alpha=0.5)

        x_lim = ax1.get_xlim()
        y_lim = ax1.get_ylim()
        for i, (y, x) in enumerate(zip(data['x2'], data['x1'])):
            x_offset = 0.01 * (x_lim[1] - x_lim[0]) 
            y_offset = 0.01 * (y_lim[1] - y_lim[0])
            ax1.text(y - x_offset, x + y_offset, str(i), fontsize=7, color='k', ha='right', va='bottom')  # plot axes labels
        for i, (z, x) in enumerate(zip(data['x3'], data['x1'])):
            x_offset = 0.01 * (x_lim[1] - x_lim[0])
            z_offset = 0.01 * (y_lim[1] - y_lim[0])
            ax1.text(z + x_offset, x + z_offset, str(i), fontsize=7, color='grey', ha='left', va='bottom')
        ax1.annotate(f"{Y}, {Z}", xy=(x_lim[1] - x_offset, 2*y_offset), ha='center', va='center', fontsize=12, color='k')
        ax1.annotate(f"{X}", xy=(2*x_offset, y_lim[1] + 2*y_offset), ha='center', va='center', fontsize=12, color='k')
        
    if projection==2:
        ax1.plot(data['x2'], data['x1'], marker='o', color='k', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0) # plot all the data
        ax1.plot(data['x2'], -data['x3'], marker='o', color='k',  markerfacecolor='white', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax1.plot(filtered['x2'], filtered['x1'], marker='o', color='k', linewidth=0.5,  label='horizontal') # plot the filtered subset of data
        ax1.plot(filtered['x2'], -filtered['x3'], marker='o', color='k', linewidth=0.5, markerfacecolor='white', label='vertical')

        for i in range(len(lines)):
            ax1.plot(lpts[i]['x2'], lpts[i]['x1'], marker='o', color=lcolors[i], linestyle='none', label=f'comp. {lnames[i]}') # plot the fitted points
            ax1.plot(lpts[i]['x2'], -lpts[i]['x3'], marker='o', markerfacecolor='none', markeredgecolor=lcolors[i], markeredgewidth=1.5, linestyle='none')
            lstart = lsegs[i][0]
            lend = lsegs[i][1]
            ax1.plot([lstart[1], lend[1]], [lstart[0], lend[0]], color=lcolors[i], lw=3, alpha=0.5) # plot the principal component
            ax1.plot([lstart[1], lend[1]], [-lstart[2], -lend[2]], color=lcolors[i], lw=3, alpha=0.5)

        x_lim = ax1.get_xlim()
        y_lim = ax1.get_ylim()
        for i, (y, x) in enumerate(zip(data['x2'], data['x1'])):
            x_offset = 0.01 * (x_lim[1] - x_lim[0])
            y_offset = 0.01 * (y_lim[1] - y_lim[0])
            ax1.text(y - x_offset, x + y_offset, str(i), fontsize=7, color='k', ha='right', va='bottom') # plot axes labels
        for i, (y, z) in enumerate(zip(data['x2'], -data['x3'])):
            y_offset = 0.01 * (y_lim[1] - y_lim[0])
            z_offset = 0.01 * (x_lim[1] - x_lim[0])
            ax1.text(y + z_offset, z + y_offset, str(i), fontsize=7, color='grey', ha='left', va='bottom')
        ax1.annotate(f"{Y}", xy=(x_lim[1] - x_offset, 2*y_offset), ha='center', va='center', fontsize=12, color='k')
        ax1.annotate(f"{X}, {negZ}", xy=(-0.1*x_offset, y_lim[1] + 2*y_offset), ha='center', va='center', fontsize=12, color='k')

    xticks = [tick for tick in ax1.get_xticks() if tick != 0]
    yticks = [tick for tick in ax1.get_yticks() if tick != 0]
    ax1.set_xticks(xticks[::2])
    ax1.set_yticks(yticks[::2]) 
    
    ax1.axhline(0, color='k',linewidth=0.8)  
    ax1.axvline(0, color='k',linewidth=0.8)
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()
    if labels:  # Only call legend if there are labels
        ax1.legend()
    
    ax2 = fig.add_subplot(gs[:2, 1])  # plot stereonet
    plot_net()
    plot_di(dec=[x for x in data['dec']], inc=[x for x in data['inc']], marker=None, connect_points=True, line_color='grey', line_width=0.5, alpha=0.50) # plot lines connecting all data
    plot_di(dec=[x for x in data['dec']], inc=[x for x in data['inc']], markersize=50, alpha=0.20) # plot all the data
    plot_di(dec=[x for x in filtered['dec']], inc=[x for x in filtered['inc']], markersize=50)     # plot the filtered data
    plot_di(dec=filtered['dec'].iloc[0], inc=filtered['inc'].iloc[0], marker='+', markersize=200)

    for i in range(len(lines)):
        ldir = pro.to_sph([ldirs[i]])
        plot_di_mean(dec=ldir[0][0], inc=ldir[0][1], a95=lmads[i], marker='*', markersize=75, color=lcolors[i], alpha=0.75, label=f'comp. {lnames[i]}') # plot the linear fits

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
    ax2.text(0.05, 0.95, f"{data['coordinates'][0]}\ncoordinates", transform=ax2.transAxes, fontsize=12, ha='left', va='top', color='black')
    
    ax3 = fig.add_subplot(gs[2:3, 1])   # plot the remanence decay plot
    ax3.plot(data['treatment'], data['res']/filtered['res'].max(), marker='o', markersize=4, color='k', alpha=0.25)
    ax3.plot(filtered['treatment'], filtered['res']/filtered['res'].max(), marker='o', markersize=4, color='k')
    ax3.tick_params(axis='y', which='both', direction='in', length=6, labelleft=False, labelbottom=False)
    ax3.grid(which='both', axis='y', linestyle='--', color='gray', linewidth=0.5)
    ax3.set_xlabel('Treatment')
    ax3.set_ylabel('Normalized intensity')

    plt.show();


def linzij_plt(coordinates, projection, data, filtered, lines, fitted, coefficients, coefficients_norm):

    if coordinates == 'specimen':
        X, Y, Z, negZ = 'X', 'Y', 'Z', '-Z'
    if coordinates == 'geographic' or coordinates == 'tectonic':
        X, Y, Z, negZ = 'N', 'E', 'Down', 'Up'
    
    lnames, ldirs, lcolors = [], [], []
    for i in range(len(lines)): 
        lnames.append(lines[i][1])
        ldirs.append(lines[i][5])
        lcolors.append(lines[i][9])
    
    fig = plt.figure(constrained_layout=True, figsize=(16,8))
    gs = fig.add_gridspec(3, 2)
    
    ax1 = fig.add_subplot(gs[:3, 0])
    if projection==1:  
        ax1.plot(data['x2'], data['x1'], marker='o', color='k', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax1.plot(data['x3'], data['x1'], marker='o', color='k',  markerfacecolor='white', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax1.plot(filtered['x2'], filtered['x1'], marker='o', color='k', linewidth=0.5,  label='horizontal')
        ax1.plot(filtered['x3'], filtered['x1'], marker='o', color='k', linewidth=0.5, markerfacecolor='white', label='vertical')
        ax1.plot(fitted[:,1], fitted[:,0], marker='o', markersize=3, color='purple', linewidth=0.25, linestyle='dashed', label='modelled')
        ax1.plot(fitted[:,2], fitted[:,0], marker='o', markersize=3, color='purple', linewidth=0.25, linestyle='dashed', markerfacecolor='white')

        x_lim = ax1.get_xlim()
        y_lim = ax1.get_ylim()
        x_offset = 0.01 * (x_lim[1] - x_lim[0])
        y_offset = 0.01 * (y_lim[1] - y_lim[0])
        ax1.annotate(f"{Y}, {Z}", xy=(x_lim[1] - x_offset, 2*y_offset), ha='center', va='center', fontsize=12, color='k')
        ax1.annotate(f"{X}", xy=(2*x_offset, y_lim[1] + 2*y_offset), ha='center', va='center', fontsize=12, color='k')
        
    if projection==2:
        ax1.plot(data['x2'], data['x1'], marker='o', color='k', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax1.plot(data['x2'], -data['x3'], marker='o', color='k',  markerfacecolor='white', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax1.plot(filtered['x2'], filtered['x1'], marker='o', color='k', linewidth=0.5,  label='horizontal')
        ax1.plot(filtered['x2'], -filtered['x3'], marker='o', color='k', linewidth=0.5, markerfacecolor='white', label='vertical')
        ax1.plot(fitted[:,1], fitted[:,0], marker='o', markersize=3, color='purple', linewidth=0.25, linestyle='dashed', label='modelled')
        ax1.plot(fitted[:,1], -fitted[:,2], marker='o', markersize=3, color='purple', linewidth=0.25, linestyle='dashed', markerfacecolor='white')

        x_lim = ax1.get_xlim()
        y_lim = ax1.get_ylim()
        x_offset = 0.01 * (x_lim[1] - x_lim[0])
        y_offset = 0.01 * (y_lim[1] - y_lim[0])
        ax1.annotate(f"{Y}", xy=(x_lim[1] - x_offset, 2*y_offset), ha='center', va='center', fontsize=12, color='k')
        ax1.annotate(f"{X}, {negZ}", xy=(-0.1*x_offset, y_lim[1] + 2*y_offset), ha='center', va='center', fontsize=12, color='k')

    xticks = [tick for tick in ax1.get_xticks() if tick != 0]
    yticks = [tick for tick in ax1.get_yticks() if tick != 0]
    ax1.set_xticks(xticks[::2])
    ax1.set_yticks(yticks[::2]) 
    
    ax1.axhline(0, color='k',linewidth=0.8)  
    ax1.axvline(0, color='k',linewidth=0.8)
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[:2, 1])

    ldirs_scaled = [ldir * a for ldir, a in zip(ldirs, coefficients[0])]
    arrow_start = np.zeros(3, dtype=float)
    
    if projection==1:  
        ax2.plot(data['x2'], data['x1'], marker='o', color='k', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax2.plot(data['x3'], data['x1'], marker='o', color='k',  markerfacecolor='white', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax2.plot(filtered['x2'], filtered['x1'], marker='o', color='k', linewidth=0.5)
        ax2.plot(filtered['x3'], filtered['x1'], marker='o', color='k', linewidth=0.5, markerfacecolor='white')        
        
        for i in range(len(ldirs)-1,-1,-1):
            ax2.quiver(arrow_start[1], arrow_start[0], ldirs_scaled[i][1], ldirs_scaled[i][0], angles='xy', scale_units='xy', scale=1, color=lcolors[i], alpha=0.5, label=f'comp. {lnames[i]}')
            ax2.quiver(arrow_start[2], arrow_start[0], ldirs_scaled[i][2], ldirs_scaled[i][0], angles='xy', scale_units='xy', scale=1, color=lcolors[i], alpha=0.5)
            arrow_start += ldirs_scaled[i][:3]

        x_lim = ax2.get_xlim()
        y_lim = ax2.get_ylim()
        x_offset = 0.01 * (x_lim[1] - x_lim[0])
        y_offset = 0.01 * (y_lim[1] - y_lim[0])
        ax2.annotate(f"{Y}, {Z}", xy=(x_lim[1] - x_offset, 2*y_offset), ha='center', va='center', fontsize=12, color='k')
        ax2.annotate(f"{X}", xy=(2*x_offset, y_lim[1] + 2*y_offset), ha='center', va='center', fontsize=12, color='k')
        
    if projection==2:
        ax2.plot(data['x2'], data['x1'], marker='o', color='k', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax2.plot(data['x2'], -data['x3'], marker='o', color='k',  markerfacecolor='white', linestyle='--', linewidth='0.75', alpha=0.25, zorder=0)
        ax2.plot(filtered['x2'], filtered['x1'], marker='o', color='k', linewidth=0.5)
        ax2.plot(filtered['x2'], -filtered['x3'], marker='o', color='k', linewidth=0.5, markerfacecolor='white')
        
        for i in range(len(ldirs)-1,-1,-1):
            ax2.quiver(arrow_start[1], arrow_start[0], ldirs_scaled[i][1], ldirs_scaled[i][0], angles='xy', scale_units='xy', scale=1, color=lcolors[i], alpha=0.5)
            ax2.quiver(arrow_start[1], -arrow_start[2], ldirs_scaled[i][1], -ldirs_scaled[i][2], angles='xy', scale_units='xy', scale=1, color=lcolors[i], alpha=0.5)
            arrow_start += ldirs_scaled[i][:3]

        x_lim = ax2.get_xlim()
        y_lim = ax2.get_ylim()
        x_offset = 0.01 * (x_lim[1] - x_lim[0])
        y_offset = 0.01 * (y_lim[1] - y_lim[0])
        ax2.annotate(f"{Y}", xy=(x_lim[1] - x_offset, 2*y_offset), ha='center', va='center', fontsize=12, color='k')
        ax2.annotate(f"{X}, {negZ}", xy=(-0.1*x_offset, y_lim[1] + 2*y_offset), ha='center', va='center', fontsize=12, color='k')

    xticks = [tick for tick in ax2.get_xticks() if tick != 0]
    yticks = [tick for tick in ax2.get_yticks() if tick != 0]
    ax2.set_xticks(xticks[::2])
    ax2.set_yticks(yticks[::2]) 
    
    ax2.axhline(0, color='k',linewidth=0.8)  
    ax2.axvline(0, color='k',linewidth=0.8)
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

    plt.show();

def interactive_zij_plt(coordinates, data, filtered, lines, planes, show_lines='y', show_planes='n'):

    if coordinates == 'specimen':
        X, Y, Z, negZ = 'X', 'Y', 'Z', '-Z'
    if coordinates == 'geographic' or coordinates == 'tectonic':
        X, Y, Z, negZ = 'N', 'E', 'Down', 'Up'
        
    fig = pgo.Figure()   # create figure

    # add the origin
    fig.add_trace(pgo.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='orange', opacity=1), showlegend=False))
    # add raw observations
    fig.add_trace(pgo.Scatter3d(x=data['x1'], y=data['x2'], z=data['x3'], mode='markers', marker=dict(size=4, color='grey', opacity=0.25), showlegend=False))
    # add filtered data
    fig.add_trace(pgo.Scatter3d(x=filtered['x1'], y=filtered['x2'], z=filtered['x3'], mode='markers', marker=dict(size=4, color='black', opacity=0.8), showlegend=False))

    if show_lines == 'y':
    # add PCA segments
        for i in range(len(lines)):
            lseg = lines[i][7]
            color = lines[i][9]
            fig.add_trace(pgo.Scatter3d(x=[lseg[0][0], lseg[1][0]], y=[lseg[0][1], lseg[1][1]],  z=[lseg[0][2], lseg[1][2]], mode="lines",
                line=dict(width=5, color=color), opacity=0.5, name=f'Comp. {lines[i][1]}'))

    if show_planes == 'y':
    # add GC planes
        for i in range(len(planes)):
            nvec = planes[i][5]
            xmax = data['x1'].abs().max()
            ymax = data['x2'].abs().max()
            zmax = data['x3'].abs().max()
            xrange = np.linspace(-xmax, xmax, 5)
            yrange = np.linspace(-ymax, ymax, 5)
            xpl, ypl = np.meshgrid(xrange, yrange)
            if nvec[2] != 0:
                zpl = (-nvec[0] * xpl - nvec[1] * ypl) / nvec[2]
            else:
                return  # skipped if plane is perfectly horizontal
            fig.add_trace(pgo.Surface(x=xpl, y=ypl, z=zpl, colorscale=[[0, 'purple'], [1, 'purple']], opacity=0.2, showscale=False))
    
    fig.update_layout(scene=dict(xaxis_title=X, yaxis_title=Y, zaxis_title=Z, aspectmode="cube"), width=1200, height=800,
                     margin=dict(l=20, r=20, t=15, b=15), legend=dict(font=dict(size=14), x=0.90,  y=0.90))
    
    fig.show()


############# site-/study-level plots #############


def overview_plt(comp_dfs, fmeans, pests, mean_treatments, mean_coefficients, coordinates, colors):

    if coordinates == 'specimen': 
        dec, inc = 'Ds', 'Is'
        gcpts = 'gcs'
    elif coordinates == 'geographic': 
        dec, inc = 'Dg', 'Ig'
        gcpts = 'gcg'
    elif coordinates == 'tectonic': 
        dec, inc = 'Dt', 'It'
        gcpts = 'gct'
    
    n = len(comp_dfs) 
    components = []
    for df in comp_dfs: components.append(df['component'].iloc[0])
    
    fig = plt.figure(constrained_layout=True, figsize=(12, 18))
    gs = fig.add_gridspec(nrows=n+3, ncols=2, height_ratios=[3] + [1]*(n+2))

    ax1 = fig.add_subplot(gs[0, :])
    plot_net()
    for i, df in enumerate(comp_dfs):
        
        #lines
        ldf = df[df['fit_type'] == 'line']
        plot_di(dec=[x for x in ldf[dec]], inc=[x for x in ldf[inc]], markersize=50, color=colors[i], alpha=0.5, label=f'comp. {components[i]}')
        
        #planes
        gcdf = df[df['fit_type'] == 'plane']
        gcs = gcdf[gcpts]
        pesti = pests[i]
        for j, gc in enumerate(gcs):
            up_gcpts = gc[gc[:, 1] < 0]
            dn_gcpts = gc[gc[:, 1] > 0]
            plot_di(dec=[x for x in up_gcpts[:,0]], inc=[x for x in up_gcpts[:,1]], markersize=0.5, color='grey', alpha=0.2)
            plot_di(dec=[x for x in dn_gcpts[:,0]], inc=[x for x in dn_gcpts[:,1]], markersize=0.5, color='purple', alpha=0.5)
            plot_di(dec=pesti[j][0], inc=pests[j][1], marker='s', markersize=25, color='purple')
        
        #mean
        plot_di_mean(dec=fmeans[i]['dec'], inc=fmeans[i]['inc'], a95=fmeans[i]['alpha95'], marker='*', markersize=100, color=colors[i]) #label=f'Comp. {components[i]} mean')

    ax1.legend(loc='upper right', fontsize=12, markerscale=1.25)

    
    axes = []
    subplt_idx = 0
    for i in range(n):
        df = comp_dfs[i]
        df = df[df['fit_type']=='line']
        
        for j in range(2):
            ax = fig.add_subplot(gs[i + 1, j])
            axes.append(ax)
            if j == 0:
                dfAF = df[df['demag'] == 'AF']
                for treatment, coefficients in zip(dfAF['treatment'], dfAF['coefficients']):
                    ax.plot(treatment, abs(coefficients), color=colors[i], alpha=0.5)
                ax.plot(mean_treatments[subplt_idx], mean_coefficients[subplt_idx], color=colors[i], linewidth=2, label=f'comp. {components[i]}')
                if len(mean_treatments[subplt_idx]) > 0: 
                    ax.legend(loc='upper right')
                ax.set_ylabel('remanent contribution')
                if i == 0:
                    ax.set_title('AF demagnetization spectra')
                if i == n-1:
                    ax.set_xlabel('treatment (mT)')
                else: ax.set_xticks([])
                        
            else:
                dfTH = df[df['demag'] == 'TH']
                for treatment, coefficients in zip(dfTH['treatment'], dfTH['coefficients']):
                    ax.plot(treatment, abs(coefficients), color=colors[i], alpha=0.5)
                ax.plot(mean_treatments[subplt_idx], mean_coefficients[subplt_idx], color=colors[i], linewidth=2, label=f'comp. {components[i]}')
                if len(mean_treatments[subplt_idx]) > 0: 
                    ax.legend(loc='upper right')
                if i == 0:
                    ax.set_title('Thermal demagnetization spectra')
                if i == n-1:
                    ax.set_xlabel('treatment (deg.)')
                else: ax.set_xticks([]) 

            subplt_idx+=1
    
    for i in range(len(axes)):
        if i % 2 == 0: 
            axes[i].sharex(axes[-2])
        else:
            axes[i].sharex(axes[-1])
    
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
              
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

def plot_di_mean(dec, inc, a95, color='k', marker='o', markersize=20, label='', legend='no', alpha=1, zorder=2):
    """
    Plot a mean direction with alpha_95 ellipse.
    """
    DI_dimap = dimap(dec, inc)
    plt.scatter(DI_dimap[0], DI_dimap[1], edgecolors=color, facecolors='white' if inc < 0 else color, marker=marker, s=markersize, label=label, zorder=zorder)

    # Get the circle points corresponding to a95 and plot them
    Da95, Ia95 = circ(dec, inc, a95)
    Xcirc, Ycirc = zip(*[dimap(D, I) for D, I in zip(Da95, Ia95)])
    plt.plot(Xcirc, Ycirc, c=color)

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