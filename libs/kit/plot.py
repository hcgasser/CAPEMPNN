""" offers some helper functions for plotting """

import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

A4_width = 8.27
A4_height = 11.69

def rm_axes_elements(axes, elements, y_axis=True, x_axis=True):
    """removes elements from the axes

    :param axes: list of axes to remove the elements from
    :param elements: elements to remove (e.g. "ticks,ticklabels,plain,labels,grid,legend")
    :param y_axis: whether to remove the elements from the y-axis
    :param x_axis: whether to remove the elements from the x-axis
    """

    elements = elements.split(",")

    # pylint: disable=unidiomatic-typecheck
    if type(axes) != list:
        axes = [axes]
    for ax in axes:
        if ax is not None:
            if "ticks" in elements:
                if x_axis:
                    ax.set_xticks([])
                if y_axis:
                    ax.set_yticks([])
            if "ticklabels" in elements:
                if x_axis:
                    ax.set_xticklabels("")
                if y_axis:
                    ax.set_yticklabels("")
            if "plain" in elements:
                ax.axis("off")
            if "labels" in elements:
                if x_axis:
                    ax.set_xlabel("")
                if y_axis:
                    ax.set_ylabel("")
            if "grid" in elements:
                if x_axis:
                    ax.xaxis.grid(False)
                if y_axis:
                    ax.yaxis.grid(False)
            if "legend" in elements:
                ax.legend_.remove()


def plot_text(text, ax, font_scale=1.0, rotation=0, x_pos=0.5, y_pos=1.0):
    """plots text in the specified axes"""

    rm_axes_elements(ax, "plain")
    ax.text(
        x_pos,
        y_pos,
        text,
        fontsize=plt.rcParams["font.size"] * 1.5 * font_scale,
        ha="center",
        va="top",
        rotation=rotation,
        rotation_mode='anchor'
    )


def plot_legend_patches(legend, ax, location="center"):
    """plots a legend in the specified axes

    the legend is showing patches with the specified colors and labels
    """

    patches = []
    for key, value in legend.items():
        patches.append(mpatches.Patch(color=value, label=key))

    ax.legend(
        handles=patches,  # loc='upper right')
        loc=location,
        fancybox=False,
        shadow=False,
        ncol=1,
    )

    rm_axes_elements(ax, "plain")


def plot_legend_scatter(ax, labels, markers, colors, **kwargs):
    """plots a legend in the specified axes

    the legend is showing scatter plot elements with the specified colors and labels
    """

    legend_elements = [
        plt.scatter([], [], label=l, linewidth=2, marker=m, color=c)
        for l, m, c in zip(labels, markers, colors)
    ]
    ax.legend(handles=legend_elements, title=None, fancybox=False, **kwargs)


def plot_legend(ax, markers, colors, **kwargs):
    assert len(set(markers.keys()) ^ set(colors.keys())) == 0

    _labels, _markers, _colors = [], [], []
    for l in markers.keys():
        _labels.append(l)
        _markers.append(markers[l])
        _colors.append(colors[l])

    plot_legend_scatter(ax, _labels, _markers, _colors, **kwargs)


def plot_green_red_areas(top, bottom, good, ax, alpha=0.01):
    # Add a red area for values above 0
    color_positive, color_negative = ("green", "red") if good > 0 else ("red", "green")
    
    for lower in np.linspace(top, 0, 50):
        ax.axhspan(lower, top, facecolor=color_positive, alpha=alpha)
    
    for upper in np.linspace(bottom, 0, 50):
        ax.axhspan(bottom, upper, facecolor=color_negative, alpha=alpha)


def plot_borders_in_stripplot(cnt, ax):
    for category in range(cnt - 1):
        ax.axvline(category + 0.5, color='gray', linestyle='--')


def spider_plot(df_values,
                fill=None,
                area=None, ax=None,
                ylim=None, ysections=None,
                face_center=True, fy=1.15):
    """ plots a spider web figure

    :param df_values: indices are the categories and columns are the series
    :param fill:
    :param area:
    :param ax:
    :param ylim:
    :param ysections:
    :param face_center:
    :param fy:
    :return:
    """

    if area is None and ax is None:
        fig = plt.figure(figsize=(A4_width, A4_width))
        ax = fig.add_subplot(1, 1, 1, projection='polar')
    elif ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(area, projection='polar')

    categories = list(df_values.index)

    N = len(categories)  # number of variables
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    for c in df_values.columns:  # plot all series
        # plot data
        values = list(df_values[c])
        values += [values[0]]
        ax.plot(angles, values, linewidth=2, linestyle='solid')

        # fill area
        if fill is not None and fill[i]:
            ax.fill(angles, values, 'b', alpha=0.1)

    # add value labels inside
    ax.set_rlabel_position(0)
    # plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=7)
    if ylim is not None:
        ax.set_ylim(ylim)
        if ysections is not None:
            sec = np.array(range(ysections+1))*(ylim[1] - ylim[0])/ysections
            sec = sec[:-1]
            ax.set_yticks(sec)
            ax.set_yticklabels([f"{s:.2f}" for s in sec], color="grey", size=7)
    # plt.ylim(0, 5)

    # add category labels outside the circle
    if not face_center:
        ax.set_xticks(angles[:-1], categories)
    else:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        for i, category in enumerate(categories):
            angle = angles[i]
            angle_label = (-pi / 2 + angle) if angle <= pi else (-pi / 2 + angle - pi)
            rotation = np.degrees(angle_label)
            ha = 'center'
            va = 'center'  #'bottom' if angle <= pi / 2 or angle > 3 * pi / 2 else 'top'
            ax.text(
                angle,
                ax.get_ylim()[1]*fy,
                f"{category}",  # _{angle:.2f}_{rotation:.2f}",
                size=12,
                horizontalalignment=ha,
                verticalalignment=va,
                rotation=rotation,
                rotation_mode='anchor'
            )

    return ax


