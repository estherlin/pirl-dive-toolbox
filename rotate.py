''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve rotate.py --args filepath
at your command prompt. Then navigate to the URL
    http://localhost:5006/rotate
in your browser.
'''

import numpy as np
import sys
from joblib import Parallel, delayed
import pandas as pd

#sys.path.insert(0, '../utils/')
from interference import *
import videoparser as vd

from bokeh.plotting import figure, show, curdoc
from bokeh.io import output_notebook
from bokeh.layouts import row, column, widgetbox
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer, ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.resources import CDN
from bokeh.embed import file_html


# CONSTANTS
TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,save"

# Get filename
filename = "//win.desy.de/group/cfel/4all/mpsd_drive/massspec/Users/Esther/19092018/fringes/19092018_prolene_1468_0_1.bmp"
dx = 125

# Load in image
img = vp.VideoEdit(filename, cut=False, angle=0.0).frames[0][:,:,1]
print(img.shape)
x_center, y_center = vd.VideoEdit.find_center(img)
print(x_center, y_center)
x = np.linspace(1, 800, num=800)
y = np.linspace(1, 600, num=600)
xx, yy = np.meshgrid(x, y)

# DATA
# Variables of data
x_proj, y_proj = vp.VideoEdit.xy_projections(img)
x_proj = x_proj / np.max(x_proj)
y_proj = y_proj / np.max(y_proj)

source = ColumnDataSource(data=dict(x=x, y=x_proj))
image_source = [img]


# Create the figure
p = figure(tools = TOOLS,
           x_range=(0,int(np.max(x))),
           y_range=(0,int(np.max(y))),
           plot_width=int(np.max(x)/2),
           plot_height=int(np.max(y)/2),
           toolbar_location="above",
           title="Interference Image")

# More figure properties
p.background_fill_color = '#000000'
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False

# give it an image
p.image(image=image_source, x=0, y=0, dw=int(np.max(x)), dh=int(np.max(y)), palette="Greys9")

# Create the horizontal graph with normalized data
ph = figure(plot_height=200,
           plot_width=p.plot_width,
           toolbar_location="left",
           x_range=(x_center-dx,x_center+dx),
           y_range=(np.min(np.abs(x_proj)), np.max(x_proj)+0.2),
           y_axis_location="right",
           tools=TOOLS)
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"

ph.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
ph.scatter('x', 'y', source=source, size=3, color="#3A5785", alpha=0.6)

# Create the Sliders
angle = Slider(title='angle', value=0.0, start=-5.0, end=5.0,step=0.1)

def update_data(attrname, old, new):
    a = angle.value
    image = vd.VideoEdit.rotate_img(img, angle.value, x_center, y_center)

    # DATA
    # Variables of data
    x_proj, y_proj = vp.VideoEdit.xy_projections(image)
    x_proj = x_proj / np.max(x_proj)
    y_proj = y_proj / np.max(y_proj)

    source.data = dict(x=x, y=x_proj)
    image_source[0] = image

    ph.title.text = filename + ": " + str(angle.value)

for w in [angle]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = widgetbox(angle)
input_col = column(Spacer(width=200, height=200), inputs)
layout = column(row(p, input_col), row(ph,Spacer(width=200, height=600)))
curdoc().add_root(layout)

"""

# Set up plot
plot = figure(tools = TOOLS,
           x_range=(0,int(np.max(x))),
           y_range=(0,int(np.max(y))),
           plot_width=int(np.max(x)),
           plot_height=int(np.max(y)),
           toolbar_location="above",
           title="Interference Image")

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)


# Set up widgets
text = TextInput(title="title", value='my sine wave')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    a = amplitude.value
    b = offset.value
    w = phase.value
    k = freq.value

    # Generate the new curve
    x = np.linspace(0, 4*np.pi, N)
    y = a*np.sin(k*x + w) + b

    source.data = dict(x=x, y=y)

for w in [offset, amplitude, phase, freq]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = widgetbox(text, offset, amplitude, phase, freq)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"
"""
"""
# CONSTANTS
TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,save"

#filename = "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/30082018_prolene_1458_0.avi"

# Get filename
filename = sys.argv[1]

# Load in image
img = vp.VideoEdit(filename, cut=False, angle=0.0).frames[0]
x_center, y_center = vd.VideoEdit.find_center(img)
x = np.linspace(1, 800, num=800)
y = np.linspace(1, 600, num=600)
xx, yy = np.meshgrid(x, y)

# DATA
# Variables of data
x_proj, y_proj = vp.VideoEdit.xy_projections(img)
x_proj = x_proj / np.max(x_proj)
y_proj = y_proj / np.max(y_proj)

hsource = ColumnDataSource(data=dict(x=x, y=x_proj))
isource = ColumnDataSource({'value': img})

# Create the figure
p = figure(tools = TOOLS,
           x_range=(0,int(np.max(x))),
           y_range=(0,int(np.max(y))),
           plot_width=int(np.max(x)),
           plot_height=int(np.max(y)),
           toolbar_location="above",
           title="Interference Image")

# More figure properties
p.background_fill_color = '#000000'
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False

# give it an image
p.image(image=[img], x=0, y=0, dw=int(np.max(x)), dh=int(np.max(y)), palette="Greys9")

# give it an image
#p.image(image=[img], x=0, y=0, dw=int(np.max(x)), dh=int(np.max(y)), palette="Greys9")

# Create the horizontal graph with normalized data
ph = figure(plot_height=300,
           plot_width=p.plot_width,
           toolbar_location="left",
           x_range=p.x_range,
           y_range=(-np.max(x_proj)+1.5, np.max(x_proj)+0.1),
           y_axis_location="right",
           tools=TOOLS)
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"

ph.line('x', 'y', source=hsource, line_width=1, line_alpha=0.6)
ph.scatter('x', 'y', source=hsource, size=3, color="#3A5785", alpha=0.6)

# WIDGETS
angle = Slider(title="angle", value=0.0, start=-2.5, end=2.5, step=0.1)

# CALLBACKS
def update_data(attr, old, new):
    if text.value != '0.0':
        p.title.text = "Image at " + text.value + "degrees"
        angle.value = float(text.value)
    text.value = str(angle.value)
    p.title.text = "Image at " + str(angle.value) + " degrees"

    # Variables of data
    img = vd.VideoEdit.rotate_img(img, angle.value, x_center, y_center)
    x_proj, y_proj = vp.VideoEdit.xy_projections(img)
    x_proj = x_proj / np.max(x_proj)
    y_proj = y_proj / np.max(y_proj)

    hsource.data = dict(x=x, y=x_proj)

angle.on_change('value', update_data)

# Set up widget
inputs = widgetbox(angle)
input_col = column(Spacer(width=200, height=100), inputs)
layout = column(row(p, input_col), row(ph,Spacer(width=200, height=600)))
curdoc().add_root(layout)
"""
