import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import string

import kit.globals as G
from kit.plot import A4_width, A4_height, rm_axes_elements, \
    plot_legend, plot_legend_scatter, plot_green_red_areas, plot_borders_in_stripplot
from kit.path import join
from kit.data.utils import invert_dict
from kit.bioinf.proteins.plots import plot_aa_dist, plot_pwm_dist, plot_spider_seqs_aa_dist

from CAPE.Eval.MPNN import E
from CAPE.Eval.MPNN.design import Design, get_infos_df, reduce_dict
from CAPE.Eval.MPNN.utils import \
    c_avg_vis_mhc_1_v_g, c_avg_vis_mhc_1_pc_v_g, c_avg_rec_v_g, c_avg_tm_v_s, \
    c_avg_vis_mhc_1_v_s, c_avg_tm_t_s, c_avg_vis_mhc_1_t_s


rn_ = {
    'avg_vis_mhc_1_v_g': 'mean visibility (abs)',     # 'mean visibility MHC-1 (general)'
    'avg_vis_mhc_1_pc_v_g': 'mean visiblity',   # 'mean relative visibility MHC-1 (general)'
    'avg_rec_v_g': 'mean sequence recovery',    # 'mean sequence recovery (general)'
    'avg_tm_v_s': 'mean TM-score',              # 'mean TM score (specific VAL)'
    'avg_vis_mhc_1_v_s': 'mean visibility',     # 'mean visibility MHC-1 (specific VAL)'
    'avg_tm_t_s': 'mean TM-score',              # 'mean TM score (specific TEST)'
    'avg_vis_mhc_1_t_s': 'mean visibility',     # 'mean visibility MHC-1 (specific TEST)'
}

base_model_ids = None

color_data = 'green'
color_model_base = None
color_model_ids = None
color_ckpt = None
color_bounds = None

marker_data = 'o'
marker_ckpt = None
marker_model_base = None
markers_epochs = None


def set_colors(_color_ckpt, _color_model_base, _color_model_ids, _color_bounds):
    global color_ckpt, color_model_base, color_model_ids, color_bounds
    color_ckpt = _color_ckpt
    color_model_base = _color_model_base
    color_model_ids = _color_model_ids
    color_bounds = _color_bounds


def set_markers(_marker_ckpt, _marker_model_base, _markers_epochs):
    global marker_ckpt, marker_model_base, markers_epochs
    marker_ckpt = _marker_ckpt
    marker_model_base = _marker_model_base
    markers_epochs = _markers_epochs
    markers_epochs[-1] = marker_model_base


def get_palette(source_ids, for_legend=False):
    p = {}
    for source_id in source_ids:
        if source_id.startswith('data'):
            if source_id == 'data':
                p[source_id] = color_data  # Data
            else:
                p[source_id] = f"light{color_data}"  # Data
        else:
            if source_id in base_model_ids:  # Base model
                if len(base_model_ids) > 1 or not for_legend:  # more than one base model
                    p[source_id] = color_model_base
                else:
                    p['ProteinMPNN'] = color_model_base
            else:
                model_id = source_id.split(":")[0]

                if model_id in color_model_ids:
                    p[source_id] = color_model_ids[model_id]  # model with special color
                else:
                    p[source_id] = color_ckpt  # other model
    return p


def get_markers(source_ids, for_legend=False):
    result = {}
    for source_id in source_ids:
        if source_id.startswith('data'):
            result[source_id] = marker_data  # Data
        else:
            if source_id in base_model_ids:  # Base model
                if len(base_model_ids) > 1 or not for_legend:  # more than one base model
                    result[source_id] = marker_model_base
                else:
                    result['ProteinMPNN'] = marker_model_base
            else:
                epoch = int(source_id.split(":")[1].split("_")[1])
                result[source_id] = markers_epochs[epoch]
    return result


def get_source_color_marker(source_ids):
    result = {}

    for source_id in source_ids:
        if source_id.startswith('data'):
            result[source_id] = (color_data, marker_data)
            continue
        s = source_id.split(":")
        if len(s) == 2:
            model_id, epoch = s
            epoch = int(epoch.split("_")[1])
            color_model, marker_epoch = color_model_ids[model_id], markers_epochs[epoch]
            result[source_id] = (color_model, marker_epoch)
        elif source_id in base_model_ids:
            result[source_id] = (color_model_base, marker_model_base)
        else:
            raise(f"Unknown checkpoint id: {source_id}")
    return result


def rn(df, columns):
    return df[columns].rename(columns=rn_)


def plot_hyp_search(df_plot, dpo_hyp_parameters, epochs, save_fig=False):
    df_plot = df_plot.query(f'epoch in {epochs}')

    p = get_palette(df_plot.model_id.unique())
    
    fig = plt.figure(figsize=(A4_width, 0.6*A4_height))

    n_cols = len(dpo_hyp_parameters)

    ax_y_1, ax_y_2 = None, None
    for n, (dpo_hyp, dpo_scale) in enumerate(dpo_hyp_parameters):  
        # Plot bottom row first - to share x axis
        ax = fig.add_subplot(2, n_cols, n_cols + n+1, sharey=ax_y_2)
        sns.scatterplot(data=rn(df_plot, [c_avg_vis_mhc_1_pc_v_g, dpo_hyp, 'epoch', 'model_id']), 
                    x=dpo_hyp, y=rn_[c_avg_vis_mhc_1_pc_v_g], 
                    color=color_ckpt, 
                    hue="model_id", palette=p,
                    style="epoch", markers=markers_epochs,
                    ax=ax)
        plt.xscale(dpo_scale)
        ax.legend([], frameon=False) 
        if n > 0:
            ax.set_ylabel("")
            #ax.set_yticklabels("")
        else:
            ax_y_2 = ax
        
        ax = fig.add_subplot(2, n_cols, n+1, sharey=ax_y_1, sharex=ax)
        sns.scatterplot(data=rn(df_plot, [c_avg_tm_v_s, dpo_hyp, 'epoch', 'model_id']), 
                    x=dpo_hyp, y=rn_[c_avg_tm_v_s], 
                    color=color_ckpt, 
                    hue="model_id", palette=p,
                    style="epoch", markers=markers_epochs,
                    ax=ax)
        plt.xscale(dpo_scale)
        ax.legend([], frameon=False)
        ax.set_xlabel("")
        #ax.set_xticklabels("")
        if n > 0:
            ax.set_ylabel("")
            #ax.set_yticklabels("")
        else:
            ax_y_1 = ax

    if save_fig:
        fig.savefig(join(G.ENV.ARTEFACTS, "figures", "CAPE-MPNN", G.DOMAIN, f"Figure_H.pdf"), bbox_inches='tight')


def plot_trade_off(df_plot, ckpt_ids_selected, save_fig=False):
    ckpt_ids_selected = get_source_color_marker(ckpt_ids_selected)
    fig = plt.figure(figsize=(A4_width, A4_height/4))
    all_axes = []

    #
    #  Plot mean visibility (x) vs mean sequence recovery (y)
    #
    ax = fig.add_subplot(1,2,1)
    all_axes.append(ax)
    # plot all checkpoints
    sns.scatterplot(data=rn(df_plot, [c_avg_vis_mhc_1_pc_v_g, c_avg_rec_v_g]), 
                    x=rn_[c_avg_vis_mhc_1_pc_v_g], y=rn_[c_avg_rec_v_g], 
                    color=color_ckpt, marker=marker_ckpt, ax=ax)
    # plot base models
    sns.scatterplot(data=rn(df_plot.query(f'source in {base_model_ids}'), [c_avg_vis_mhc_1_pc_v_g, c_avg_rec_v_g]),
                    x=rn_[c_avg_vis_mhc_1_pc_v_g], y=rn_[c_avg_rec_v_g], 
                    color=color_model_base, ax=ax)
    # plot selected checkpoints
    sns.scatterplot(data=rn(df_plot.query(f'source in {list(ckpt_ids_selected)}'), [c_avg_vis_mhc_1_pc_v_g, c_avg_rec_v_g, "epoch"]), 
                    x=rn_[c_avg_vis_mhc_1_pc_v_g], y=rn_[c_avg_rec_v_g], 
                    hue="source", palette={k: v[0] for k, v in ckpt_ids_selected.items()},
                    style="epoch", markers=markers_epochs,
                    ax=ax)
    ax.legend([], frameon=False)  # switch off legend for left plot
    
    # plt.hlines(c_avg_rec_v_g_lower_bound, xmin=0., xmax=1.1, color=color_bounds, linestyle='--')
    # plt.vlines(c_avg_vis_mhc_1_pc_v_g_upper_bound, ymin=0., ymax=.5, color=color_bounds, linestyle='--')

    #
    #  Plot mean visibility (x) vs mean TM score (y)
    #
    ax = fig.add_subplot(1,2,2)
    all_axes.append(ax)
    # plot all checkpoints
    sns.scatterplot(data=rn(df_plot, [c_avg_vis_mhc_1_pc_v_g, c_avg_tm_v_s]), 
                    x=rn_[c_avg_vis_mhc_1_pc_v_g], y=rn_[c_avg_tm_v_s], 
                    color=color_ckpt, marker=marker_ckpt, ax=ax)
    # plot base models
    sns.scatterplot(data=rn(df_plot.query(f'source in {base_model_ids}'), [c_avg_vis_mhc_1_pc_v_g, c_avg_tm_v_s]), 
                    x=rn_[c_avg_vis_mhc_1_pc_v_g], y=rn_[c_avg_tm_v_s], 
                    color=color_model_base, ax=ax)
    # plot selected checkpoints
    sns.scatterplot(data=rn(df_plot.query(f'source in {list(ckpt_ids_selected)}'), [c_avg_vis_mhc_1_pc_v_g, c_avg_tm_v_s, "epoch"]), 
                    x=rn_[c_avg_vis_mhc_1_pc_v_g], y=rn_[c_avg_tm_v_s], 
                    hue="source", palette={k: v[0] for k, v in ckpt_ids_selected.items()},
                    style="epoch", markers=markers_epochs,
                    ax=ax)

    base_model_names = ['ProteinMPNN'] if len(base_model_ids) == 1 else base_model_ids
    plot_legend_scatter(ax, 
        list(ckpt_ids_selected) + ['other checkpoints'] + base_model_names, 
        [v[1] for v in ckpt_ids_selected.values()] + ['.', 'o'], 
        [v[0] for v in ckpt_ids_selected.values()] + [color_ckpt, color_model_base]
        )

    labels = [f'{c})' for c in string.ascii_lowercase]
    for i, ax in enumerate(all_axes):
        ax.text(-0., 1.08, labels[i], transform=ax.transAxes,
                fontweight='bold', va='top', ha='right') #, color='red')
    if save_fig:
        fig.savefig(join(G.ENV.ARTEFACTS, "figures", "CAPE-MPNN", G.DOMAIN, f"Figure_A.pdf"), bbox_inches='tight')


def plot_info_boxplot(
        source_ids,
        protein_ids,
        best,
        info,
        min_info=None,
        min_info_value=None,
        title=None,
        ax=None
):
    if ax is None:
        fig = plt.figure(figsize=(A4_width, A4_height / 4))
        ax = fig.add_subplot(1, 1, 1)

    info_name = info[1]

    infos = [info] if min_info is None else [info, min_info]
    df_plot = get_infos_df(source_ids, protein_ids, best, infos)
    if min_info_value is not None:
        df_plot.loc[df_plot[min_info[1]] < min_info_value, info_name] = None  # set to None where the threshold is not met

    source_ids_designs = get_source_color_marker(source_ids)

    if title is not None:
        ax.set_title(title)

    sns.boxenplot(
        data=df_plot.query(f'source not in {base_model_ids}'),
        x='protein',
        y='visibility',
        ax=ax,
        color='lightgrey'
    )

    _markers_epochs = markers_epochs
    _markers_epochs[-1] = marker_model_base
    for e, m in _markers_epochs.items():
        _df = df_plot.copy()
        _df[info_name] = _df.apply(
            lambda row: row[info_name] if row.epoch == e else None, axis=1)
        if len(_df.query(f"not `{info_name}`.isnull()")) > 0:
            sns.stripplot(
                data=_df,
                x='protein',
                y=info_name,
                hue='source', palette={k: v[0] for k, v in source_ids_designs.items()},
                marker=m,
                alpha=1.,
                jitter=0.25,
                ax=ax
            )
    ax.legend([], frameon=False)
    _xmax = len(df_plot.protein.unique())-0.5
    ax.axhline(1., color=color_data, linestyle='--')
    ax.set_xlim((-0.5, _xmax))

    return ax


def plot_info_vs_info(source_ids, protein_ids, best, info_x, info_y, title=None, show_legend=False, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(A4_width, A4_height / 4))
        ax = fig.add_subplot(1, 1, 1)

    df_plot = get_infos_df(source_ids, protein_ids, best, [info_x, info_y])

    source_id_designs = get_source_color_marker(source_ids)

    if title is not None:
        ax.set_title(title)

    sns.scatterplot(data=df_plot,
                    y=info_y[1], x=info_x[1],
                    hue='source', palette={k: v[0] for k, v in source_id_designs.items()},
                    style='epoch', markers=markers_epochs,
                    ax=ax
                    )

    if info_x[3]:
        ax.axvline(1., color=color_data, linestyle='--')

    if info_y[3]:
        ax.axhline(1., color=color_data, linestyle='--')

    source_ids = list(source_id_designs.keys())
    if show_legend:
        plot_legend(ax,
                    get_markers(source_ids, for_legend=True),
                    get_palette(source_ids, for_legend=True),
        )

    return ax



def plot_specific(
        source_ids,
        protein_ids,
        best,
        info_x,
        info_y,
        min_info_y_value=0.9,
        save_fig=None
):
    fig = plt.figure(figsize=(A4_width, 3*A4_height/4))
    all_axes = []
    gs = mpl.gridspec.GridSpec(
            4,
            1,
            height_ratios=[2, 6, 2, 6],
            width_ratios=[1],
            wspace=0.1,
            hspace=0.0,
    )

    ax = fig.add_subplot(gs[1, :])
    all_axes.append((ax, 0.))

    plot_info_vs_info(
        source_ids,
        protein_ids,
        best,
        info_x,
        info_y,
        'Visibility vs. quality of specific designs',
        True,
        ax=ax
    )

    # (x,y), width, height
    rect = plt.Rectangle((0.0, min_info_y_value), 0.2, 1-min_info_y_value, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)
    
    ax = fig.add_subplot(gs[3, :])
    all_axes.append((ax, 0.))

    plot_info_boxplot(
        source_ids,
        protein_ids,
        best,
        info_x,
        info_y,
        min_info_y_value,
        title=f"Visibility for designs with TM-score above {min_info_y_value}",
        ax=ax
    )
    
    labels = [f'{c})' for c in string.ascii_lowercase]
    for i, (ax, x_offset) in enumerate(all_axes):
        ax.text(x_offset, 1.2, labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='right')

    if save_fig is not None:
        fig.savefig(join(G.ENV.ARTEFACTS, "figures", "CAPE-MPNN", G.DOMAIN, save_fig), bbox_inches='tight')

    return all_axes


def plot_spider_seq_aa_dist_gen_data_base(source_id, protein_id, base_model_name, best):
    seq = E.DB.get_seq(best[source_id][protein_id])
    data_seq = E.DB.get_seq(best['data'][protein_id])
    base_seq = E.DB.get_seq(best[base_model_name][protein_id])

    plot_spider_seqs_aa_dist({source_id: seq, 'data': data_seq, base_model_name: base_seq})


def prettify(text):
    return text.replace('delta', r'$\Delta$')

def plot_info(source_ids,
              protein_ids,
              best,
              info_request,
              y_scale='linear',
              hline=None,
              good=1,
              alpha=0.01,
              show_legend=False,
              show_background_range=None,
              show_background=True,
              show_data=False,
              show_xticklabels=True,
              area=None,
              jitter=0.3, pretty=True):

    info_name = info_request[1]
    if area is None:
        fig = plt.figure(figsize=(A4_width, A4_height / 4))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = plt.gcf()
        ax = fig.add_subplot(area)

    # get_best_seq_hashes(self, best_trial=BestTrial.MAX_TM, designed_positions='all'):

    source_ids = source_ids + ['data_similar', 'data']
    exclude_from_scatter = []
    if not show_background:
        exclude_from_scatter += ['data_similar']
    if not show_data:
        exclude_from_scatter += ['data']

    palette = get_palette(source_ids)
    markers = get_markers(source_ids)
    i_markers = invert_dict(markers)

    df_plot = get_infos_df(source_ids, protein_ids, best, [info_request])

    for m, _source_ids in i_markers.items():
        _df = df_plot.query(f"source in {_source_ids} and source not in {exclude_from_scatter}")
        _unique_proteins = _df.protein.unique()

        _info_name = info_name
        if pretty:
            _df.columns = [prettify(c) for c in _df.columns]
            _info_name = prettify(info_name)

        assert len(set(protein_ids)^set(_unique_proteins)) == 0
        sns.stripplot(
            data=_df,
            x='protein',
            y=_info_name,
            hue='source', palette=palette,
            jitter=jitter * (len(_source_ids) > 1),
            marker=m,
            size=6,
            ax=ax
        )
        ax.legend([], frameon=False)

    if show_background_range is not None:
        for i, protein_id in enumerate([item.get_text() for item in ax.get_xticklabels()]):
            _df = df_plot.query(f"source.str.startswith('data_similar') and protein == '{protein_id}' and not `{info_name}`.isnull()")
            if len(_df) >= show_background_range:
                min_val = _df[info_name].min()
                max_val = _df[info_name].max()
                ax.vlines(x=i, ymin=min_val, ymax=max_val, color=f"light{color_data}", linewidth=150/len(protein_ids))

    ax.set_yscale(y_scale)
    if hline is not None:
        ax.axhline(hline, linestyle='dashed', color=color_data)

    plot_borders_in_stripplot(len(protein_ids), ax)

    top = max(df_plot[info_name].max() * 1.1, 0)
    bottom = min(df_plot[info_name].min() * 1.1, 0)
    if good != 0 and good is not None:
        plot_green_red_areas(top, bottom, good, ax, alpha=alpha)
    ax.set_ylim((bottom, top))
    ax.set_xlim((-0.5, len(protein_ids)-0.5))
    ax.set(xlabel=None)

    if show_legend:
        plot_legend(ax,
                    get_markers(source_ids, for_legend=True),
                    get_palette(source_ids, for_legend=True),
        )

    ax.xaxis.tick_top()
    xticklabels = [x.get_text() for x in ax.get_xticklabels()]
    if not show_xticklabels:
        ax.set_xticklabels([])

    return ax, xticklabels


def plot_infos(source_ids, protein_ids, best, infos_to_plot,
               show_background_range=None, save_fig=None, wspace=0.1, hspace=0.1, plot_good=True):
    n_cols = 2
    n_rows = int(np.ceil(len(infos_to_plot) / n_cols)) + 1
    fig = plt.figure(figsize=(A4_width, 0.8*min(4, n_rows) * A4_height / 4))
    gs = mpl.gridspec.GridSpec(
        n_rows,
        n_cols,
        height_ratios=[1] * (n_rows-1) + [0.3],
        width_ratios=[1] * n_cols,
        wspace=wspace, hspace=hspace)

    all_axes, xticklabels_check = [], None
    for idx, (info_attr, info_name) in enumerate(infos_to_plot.items()):
        _alpha = 0.01
        _delta = "delta" in info_name or "dist" in info_name
        _hline = 0. if "delta" in info_name else None
        _relative = "rel." in info_name
        _show_background_range = show_background_range
        _show_background = False
        _y_scale = 'linear'
        _good = 0

        if 'BLOSUM-62 score' in info_name:
            _y_scale = 'linear'
        elif 'dssp' in info_name:
            _y_scale, _good, _alpha = 'linear', -1, 0.005
        elif 'rosetta total' in info_name:
            _y_scale, _good = 'symlog', -1
        elif 'hydrophobic fitness' in info_name:
            _good = -1
        elif 'packing density' in info_name:
            _good = +1
        elif 'aggrescan3d' in info_name:
            _good = -1
        elif 'dfire2 total' in info_name:
            _y_scale, _good = 'symlog', -1
        elif 'evoef2 total' in info_name:
            _y_scale, _good = 'symlog', -1
        elif 'isoelectric point' in info_name:
            pass
        else:
            raise Exception(f"info_name {info_name} unknown")

        if not _delta or not plot_good:
            _good = None

        info_request = (info_attr, info_name, _delta, _relative)
        ax, xticklabels = plot_info(
            source_ids,
            protein_ids,
            best,
            info_request,
            y_scale=_y_scale,
            hline=_hline,
            good=_good,
            alpha=_alpha,
            area=gs[idx // n_cols, idx % n_cols],
            show_background_range=_show_background_range,
            show_background=_show_background,
            show_data=not _delta,
            show_xticklabels=((idx // n_cols) == 0)
        )

        if xticklabels_check is None:
            xticklabels_check = xticklabels
        assert xticklabels == xticklabels_check

        if idx % n_cols == 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        plt.xticks(rotation=90)

        all_axes.append(ax)

    ax = fig.add_subplot(gs[-1,:])

    _source_ids = ['data'] + source_ids
    rm_axes_elements(ax, "plain")
    ncol = int(np.ceil(len(_source_ids)/2))
    plot_legend(ax,
                get_markers(_source_ids, for_legend=True),
                get_palette(_source_ids, for_legend=True),
                ncol=ncol,
                loc='center'
    )

    if save_fig is not None:
        fig.savefig(join(G.ENV.ARTEFACTS, "figures", "CAPE-MPNN", G.DOMAIN, save_fig), bbox_inches='tight')
