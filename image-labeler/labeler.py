#!/usr/bin/env python
# Image Labeling Script
# Takes output from emotion/pain estimator script and overlays on each
# individual frame.

import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# Font must be provided in working directory
FONT_SIZE = 10
FONT = ImageFont.truetype("fonts/FreeMono.otf", size=FONT_SIZE)
FONT_BOLD = ImageFont.truetype("fonts/FreeMonoBold.otf", size=FONT_SIZE)


class LabeledFrame(object):
    """
    Wrapper class for PIL.Image
    """

    def __init__(self, filename, frame_id, intensity_predicted=-1, intensity_actual=-1):
        # PIL.Image "isn't meant to be subclassed", so we have to wrap it
        self.frame_id = frame_id
        self.filename = filename
        self.intensity_predicted = intensity_predicted
        self.intensity_actual = intensity_actual

        try:
            self.clean_image = Image.open(filename)

        except IOError:
            print("ERROR: Failed to open " + filename)
            self.clean_image = Image.new("RGB", (100, 100), "grey")

    def label(self):
        error = abs(self.intensity_actual - self.intensity_predicted)
        if error == 0:
            e_color = "aqua"
        elif error < 0.75:
            e_color = "chartreuse"
        elif error < 2.5:
            e_color = "gold"
        else:
            e_color = "crimson"

        self.labeled_image = self.clean_image.copy()
        draw = ImageDraw.Draw(self.labeled_image)

        draw.text((10, 1 * (FONT_SIZE + 10)), "Filename: " +
                  self.filename, "white", font=FONT)
        draw.text((10, 2 * (FONT_SIZE + 10)), "Frame ID: " +
                  str(self.frame_id), "white", font=FONT)

        draw.text((10, 4 * (FONT_SIZE + 10)),
                  "Intensities", "white", font=FONT_BOLD)
        draw.text((10, 5 * (FONT_SIZE + 10)), "Ground Truth: " +
                  str(self.intensity_actual), "white", font=FONT)
        draw.text((10, 6 * (FONT_SIZE + 10)), "Predicted:    " +
                  str(self.intensity_predicted), "white", font=FONT)
        draw.text((10, 7 * (FONT_SIZE + 10)),
                  "Error:        " + str(error), e_color, font=FONT)

        return self.labeled_image

def test_lf():
    """
    Test for the LabeledFrame class
    """

    # IO dirs should exist
    input_dir = "images/"
    output_dir = "out/"

    # What to append to a frame ID to get the corresponding image file
    file_suffix = ".png"
    # IDs for each frame
    frame_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Intensities for each frame
    intensities_predicted = [1, 1, 1, 2, 4, 6, 6, 6, 6, 3, 2, 1, 0]
    intensities_actual = [1, 0, 0, 2, 3, 5, 6, 6, 5, 3, 1, 1, 1]
    frames = []

    # Loop through provided frame ids
    for f_id, i_pred, i_act in zip(frame_ids, intensities_predicted, intensities_actual):
        #print("Loading " + input_dir + str(f_id) + file_suffix)
        frame = LabeledFrame(input_dir + str(f_id) +
                                   file_suffix, f_id, i_pred, i_act)
        #print("Labelling " + frame.filename)
        l_image = frame.label()
        print("Saving labelled " + str(frame.frame_id) + file_suffix + " to " + output_dir)
        l_image.save(output_dir+str(frame.frame_id)+file_suffix)

test_lf()