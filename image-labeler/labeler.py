#!/usr/bin/env python
# Image Labeling Script
# Takes output from emotion/pain estimator script and overlays on each
# individual frame.

import os
import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from scipy.io import savemat, loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MATLAB_FILENAME = "pipeline_out.mat"


def load_MAT(var_name):
    loaded = loadmat(file_name=MATLAB_FILENAME, variable_names=[var_name])
    if var_name in loaded:
        return loaded[var_name]
    else:
        print("MATLAB File Load Error")
        return None


# Font must be provided in working directory
FONT_SIZE = 12
FONT = ImageFont.truetype("fonts/FreeMono.otf", size=FONT_SIZE)
FONT_BOLD = ImageFont.truetype("fonts/FreeMonoBold.otf", size=FONT_SIZE)
FONT_EM = ImageFont.truetype(
    "fonts/FreeMonoBoldOblique.otf", size=2 * FONT_SIZE)


class LabeledFrame(object):
    """
    Wrapper class for PIL.Image
    """

    def __init__(self, filename, frame_id, intensity_predicted=-1, intensity_actual=-1, bounds=(-2, -1)):
        # bounds are the bounds of the frame id range (for arange)

        # PIL.Image "isn't meant to be subclassed", so we have to wrap it
        self.frame_id = frame_id
        self.filename = filename
        self.intensity_predicted = intensity_predicted
        self.intensity_actual = intensity_actual
        self.bounds = bounds

        # filename=None when testing. Generate an empty test image
        if filename is None:
            self.clean_image = Image.new("RGB", (320, 240), "navy")
            draw = ImageDraw.Draw(self.clean_image)
            draw.text((120, 100), "test" + str(self.frame_id),
                      "wheat", font=FONT_EM)

            self.filename = "test" + str(self.frame_id)

        else:
            try:
                self.clean_image = Image.open(filename)
                self.clean_image = self.clean_image.convert(mode="RGB")

            except IOError:
                print("ERROR: Failed to open " + filename)
                self.clean_image = Image.new("RGB", (400, 400), "grey")

    def label(self):
        """
        Draw information onto the frame
        """
        
        error = abs(self.intensity_actual - self.intensity_predicted)
        # if error == 0:
        #     e_color = "aqua"
        # elif error < 0.75:
        #     e_color = "chartreuse"
        # elif error < 2.5:
        #     e_color = "gold"
        # else:
        #    e_color = "crimson"
        e_color = "white"
        self.labeled_image = self.clean_image.copy()
        draw = ImageDraw.Draw(self.labeled_image)

        draw.text((10, 490 - 5 * (FONT_SIZE + 10)), "Filename: " +
                  os.path.basename(self.filename), "white", font=FONT)
        draw.text((10, 490 - 4 * (FONT_SIZE + 10)), "Frame ID: " +
                  str(self.frame_id), "white", font=FONT)

        # draw.text((10, 3 * (FONT_SIZE + 10)),
        #          "Intensities", "white", font=FONT_BOLD)
        draw.text((10, 490 - 3 * (FONT_SIZE + 10)), "Ground Truth: " +
                  str(self.intensity_actual), "white", font=FONT)
        draw.text((10, 490 - 2 * (FONT_SIZE + 10)), "Predicted:    " +
                  str(self.intensity_predicted), "white", font=FONT)
        draw.text((10, 490 - 1 * (FONT_SIZE + 10)),
                  "Error:        " + str(error), e_color, font=FONT)

        return self.labeled_image

    def overlay_image(self, image):
        """
        Overlay an image (like a graph) in the bottom right-hand corner of frame
        
        :param image: the image to insert
        :returns: the new image
        """
        self.labeled_image.paste(image, (470, 380), image)

        return self.labeled_image


def test_lf_fdata_fframes():
    """
    Test for the LabeledFrame class with fake data & frames generated on the fly
    """

    # IO dirs should exist
    output_dir = "out/"

    # What to append to a frame ID to get the corresponding image file
    file_suffix = ".png"
    # IDs for each frame
    frame_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Intensities for each frame
    intensities_predicted = [1, 1, 1, 2, 4, 6, 6, 6, 6, 3, 2, 1, 0]
    intensities_actual = [1, 0, 0, 2, 3, 5, 6, 6, 5, 3, 1, 1, 1]

    gif_images = []

    # Loop through provided frame ids
    for f_id, i_pred, i_act in zip(frame_ids, intensities_predicted, intensities_actual):
        #print("Loading " + input_dir + str(f_id) + file_suffix)
        #frame = LabeledFrame(input_dir + str(f_id) + file_suffix, f_id, i_pred, i_act)
        frame = LabeledFrame(None, f_id, i_pred, i_act)
        #print("Labelling " + frame.filename)
        l_image = frame.label()
        print("Saving labeled " + str(frame.frame_id) +
              file_suffix + " to " + output_dir)
        l_image.save(output_dir + str(frame.frame_id) + file_suffix)

        gif_images.append(l_image)

    # Generate GIF
    print("Saving animated GIF")
    gif_images[0].save(output_dir + "animated.gif", format="gif",
                       save_all=True, append_images=gif_images[1:], duration=500)


def test_lf_rdata_fframes():
    """
    Test for the LabeledFrame class with real data, but frames generated on the fly
    """

    # IO dirs should exist
    output_dir = "out/"

    # What to append to a frame ID to get the corresponding image file
    file_suffix = ".png"

    # Intensities for each frame
    intensities_predicted = load_MAT("dec_values").flatten().tolist()
    intensities_actual = load_MAT("test_label").flatten().tolist()
    # IDs for each frame
    frame_ids = range(len(intensities_predicted))

    gif_images = []

    # Loop through provided frame ids
    for f_id, i_pred, i_act in zip(frame_ids, intensities_predicted, intensities_actual):
        #print("Loading " + input_dir + str(f_id) + file_suffix)
        #frame = LabeledFrame(input_dir + str(f_id) + file_suffix, f_id, i_pred, i_act)
        frame = LabeledFrame(None, f_id, i_pred, i_act)
        print("Labeling " + frame.filename)
        l_image = frame.label()
        #print("Saving labeled " + str(frame.frame_id) + file_suffix + " to " + output_dir)
        # l_image.save(output_dir+str(frame.frame_id)+file_suffix)

        gif_images.append(l_image)

    # Generate GIF
    print("Saving animated GIF")
    gif_images[0].save(output_dir + "animated.gif", format="gif",
                       save_all=True, append_images=gif_images[1:], duration=500)


def test_lf_rdata_rframes_nc():
    """
    Test for the LabeledFrame class with real data, and real frames, but the data and frames don't correspond
    """

    # IO dirs should exist
    input_dir = "images/jh123t1aeaff"
    output_dir = "out/"

    # What to append to a frame ID to get the corresponding image file
    file_suffix = ".png"

    # Intensities for each frame
    intensities_predicted = load_MAT("dec_values").flatten().tolist()
    intensities_actual = load_MAT("test_label").flatten().tolist()
    # IDs for each frame
    frame_ids = range(56, 360)

    gif_images = []

    plt.figure(figsize=(1.5, 1.15), dpi=100)
    plt.axis('off')
    plt.plot(frame_ids, intensities_predicted, "b-", label="predicted")
    plt.plot(frame_ids, intensities_actual, "r-", label="actual")
    #plt.vlines(self.frame_id,-1, 10)
    #plt.legend(loc='upper right')
    data_max = max(intensities_predicted + intensities_actual)
    data_min = min(intensities_predicted + intensities_actual)

    # Loop through provided frame ids
    for f_id, i_pred, i_act in zip(frame_ids, intensities_predicted, intensities_actual):
        #print("Loading " + input_dir + str(f_id) + file_suffix)
        frame = LabeledFrame(input_dir + ('0' if f_id < 100 else '') +
                             str(f_id) + file_suffix, f_id, i_pred, i_act)
        #frame = LabeledFrame(None, f_id, i_pred, i_act)
        print("Labeling " + frame.filename)
        l_image = frame.label()

        # Add vertical line for this frame
        ln = plt.vlines(f_id, data_min, data_max,
                        linestyles='solid', linewidth=".5", zorder=3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True,
                    bbox_inches='tight', pad_inches=0)
        # Remove the vert line for the next figure
        ln.remove()

        buf.seek(0)

        overlay = Image.open(buf)

        l_image = frame.overlay_image(overlay)

        #print("Saving labeled " + str(frame.frame_id) + file_suffix + " to " + output_dir)
        # l_image.save(output_dir+str(frame.frame_id)+file_suffix)

        gif_images.append(l_image)

    # Generate GIF
    print("Saving animated GIF")
    gif_images[0].save(output_dir + "animated.gif", format="gif",
                       save_all=True, append_images=gif_images[1:], duration=120)

def test_77out(images, output_dir):
    """
    Test for the LabeledFrame class with real data and real frames from the leave 77
    out experiment
    
    :param images: a list of full paths to the frames used
    :param output_dir: the path to the directory where the animated GIF should be saved
    """

    # Intensities for each frame
    intensities_predicted = load_MAT("dec_values").flatten().tolist()
    intensities_actual = load_MAT("test_label").flatten().tolist()
    # IDs for each frame
    frame_ids = range(0, len(intensities_actual))

    gif_images = []

    plt.figure(figsize=(1.5, 1.15), dpi=100)
    plt.axis('off')
    plt.plot(frame_ids, intensities_predicted, "b-", label="predicted")
    plt.plot(frame_ids, intensities_actual, "r-", label="actual")
    #plt.vlines(self.frame_id,-1, 10)
    #plt.legend(loc='upper right')
    data_max = max(intensities_predicted + intensities_actual)
    data_min = min(intensities_predicted + intensities_actual)

    # Loop through provided frame ids
    for p, f_id, i_pred, i_act in zip(images, frame_ids, intensities_predicted, intensities_actual):
        #print("Loading " + input_dir + str(f_id) + file_suffix)
        frame = LabeledFrame(p, f_id, i_pred, i_act)
        print("Labeling " + frame.filename)
        l_image = frame.label()

        # Add vertical line for this frame
        ln = plt.vlines(f_id, data_min, data_max,
                        linestyles='solid', linewidth=".5", zorder=3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True,
                    bbox_inches='tight', pad_inches=0)
        # Remove the vert line for the next figure
        ln.remove()

        buf.seek(0)

        overlay = Image.open(buf)

        l_image = frame.overlay_image(overlay)

        gif_images.append(l_image)

    # Generate GIF
    print("Saving animated GIF")
    gif_images[0].save(output_dir + "animated.gif", format="gif",
                       save_all=True, append_images=gif_images[1:], duration=220)
