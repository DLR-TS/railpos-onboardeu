"""
| Copyright (c) 2025
| Deutsches Zentrum für Luft- und Raumfahrt e.V. (DLR)
| Linder Höhe
| 51147 Köln

:Authors: Judith Heusel, Michael Roth, Keivan Kiyanfar, Erik Hedberg

:Version: 1.4.0

:Project: railpos

License
-------
This software is released under the 3-Clause BSD License.
Details about the license can be found in the LICENSE file.

Description
Functions for interactive visualizations.
-------
"""

import folium
import branca.colormap as cm


def folium_figure(width: int = 1200, height: int = 800):
    """
    Create a folium figure with a given width and height

    Parameters
    ----------
    width : int, optional
        Figure width. The default is 1200.
    height : int, optional
        Figure height. The default is 800.

    Returns
    -------
    branca.element.Figure
        Folium figure with given size where a folium.folium.Map can be added.

    """
    return folium.Figure(width=width, height=height)


def folium_layer_control(fmap: folium.folium.Map):
    """
    Adds layer control to a folium map. Can only be done once for a folium map.

    Parameters
    ----------
    fmap : folium.folium.Map
        Input folium map where the layer control should be added.

    Returns
    -------
    None.

    """
    folium.LayerControl().add_to(fmap)


def folium_graz_image(fmap: folium.folium.Map):
    """
    Add aerial image of Graz to a folium map.
    Tested on 2025-06-11.

    Parameters
    ----------
    fmap : folium.folium.Map
        Input map to add the aerial image to.

    Returns
    -------
    None.

    """
    folium.raster_layers.WmsTileLayer(
        url='https://geodaten.graz.at/arcgis/services/OGD/LUFTBILD_WMS/MapServer/WmsServer?',
        layers='1',
        transparent=False,
        control=True,
        fmt="image/png",
        name='Orthophoto 2022',
        overlay=False,
        attr='Data by https://geodaten.graz.at/',
        show=True,
        max_zoom=30
    ).add_to(fmap)


def folium_niedersachsen_image(fmap: folium.folium.Map):
    """
    Adds an aerial image of Niedersachsen (or a part, e.g. a city) to a folium map.
    Tested on 2025-06-11.

    Parameters
    ----------
    fmap : folium.folium.Map
        Input folium map to add the aerial image to.

    Returns
    -------
    None.

    """

    folium.raster_layers.WmsTileLayer(url='https://opendata.lgln.niedersachsen.de/doorman/noauth/dop_wms',
                                      layers='ni_dop20',
                                      transparent=False,
                                      control=True,
                                      fmt="image/png",
                                      name='Niedersachsen, WMS DOP 20',
                                      overlay=False,
                                      attr='https://opendata.lgln.niedersachsen.de',
                                      show=True,
                                      max_zoom=30
                                      ).add_to(fmap)


def folium_switzerland_relief(fmap: folium.folium.Map, layer_string: str = None, layer: str = 'relief'):
    """
    Adds a WMS layer with a relief map of Switzerland to a folium map.
    Tested on 2025-06-11.

    Parameters
    ----------
    fmap : folium.folium.Map
        Input folium map to add the aerial image to.
    layer : str, optional
        Name of the layer to read. If there is no layer_string specified, a layer_string is chosen according to the
        shortcut argument layer (provides two layer strings, for a relief and a vegetation map).
        If the layer_string is not None, it overwrites the layer argument.
        The default is None.
    layer: str, optional
        Either 'relief' or 'vegetation'. Defines the map layer string if it is not specified by layer.
        The default is 'relief'.

    Returns
    -------
    None.

    """
    if layer_string is None:
        if layer == 'relief':
            layer_string = 'ch.swisstopo.leichte-basiskarte_reliefschattierung'
        elif layer == 'vegetation':
            layer_string = 'ch.bafu.landesforstinventar-vegetationshoehenmodell_relief'

    folium.raster_layers.WmsTileLayer(url='https://wms.geo.admin.ch/?',
                                      layers=layer_string,
                                      transparent=False,
                                      control=True,
                                      fmt="image/png",
                                      name=layer_string,
                                      overlay=False,
                                      attr='Data by https://map.geo.admin.ch/',
                                      show=True,
                                      max_zoom=30
                                      ).add_to(fmap)


def create_linear_colormap(color_list: list = None, **kwargs):
    """
    Create a linear colormap that can be used with the folium cmap keyword argument.

    Parameters
    ----------
    color_list : list, optional
        Colors for the colormap creation. The default is ["orange", "red", "blue"].
    ** kwargs:
        Keyword arguments to be handed over to cm.LinearColormap, e.g., vmin, vmax, caption or max_labels.

    Returns
    -------
    cmap : branca.colormap.LinearColormap
        Linear colormap using the given colors.

    """
    if color_list is None:
        color_list = ["orange", "red", "blue"]
    cmap = cm.LinearColormap(color_list, **kwargs)
    return cmap


if __name__ == '__main__':
    pass
