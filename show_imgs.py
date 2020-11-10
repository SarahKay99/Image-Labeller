# ==== copied from https://stackoverflow.com/questions/4135928/pygame-display-position ==== #
import pygame
import sys
import os
import pandas as pd
from os import path

from shutil import copytree, copyfile

"""
READY-TO-USE FEATURES:
    -> Scrolling backwards and forwards.
    -> Adding boxes
    -> Removing boxes
    -> Adding classes
    -> Removing classes
    -> Changing box classes
    -> Removing images from the dataset
"""

"""
FEATURES IN TESTING:
    -> Merging classes
    -> Renaming classes
    -> Changing box classes en masse. 
    -> Showing dataset info
"""

"""
FEATURES IN PROGRESS:
    -> Recording metadata. (including requests)
    -> Unflagging images for removal
"""

"""
    All will be ready by 03/03/2020
"""

# globals
pygame.init()
DISPLAYSURF = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)


def ele_to_string(lst):
    for ele in enumerate(lst):
        ind = ele[0]
        ele = str(ele[1])
        lst[ind] = ele
    return lst


def centers_to_corners(box, width, height):
    """
    :param box: String list representing a box. ["class", "xc", "yc", "norm_box_width", "norm_box_height"]
    :return: (x0, y0), (x1, y1)
    Converting (xc, yc, norm_box_width, norm_box_height) to (x0, x1), (y0, y1)
    """

    # extracting values
    norm_xc = float(box[1])
    norm_yc = float(box[2])
    norm_box_width = float(box[3])
    norm_box_height = float(box[4])

    # calculating original box_width
    box_width = norm_box_width * width
    box_height = norm_box_height * height

    # getting top-left coordinates
    x0 = norm_xc * width - box_width / 2
    y0 = norm_yc * height - box_height / 2
    x1 = norm_xc * width + box_width
    y1 = norm_yc * height + box_height

    return (x0, y0), (x1, y1)


def write_to_file(coords, width, height, filename, category):
    """
    :param coords: [(x0, x1), (y0, y1)]
    :param width: int or float. whole image width in px.
    :param height: int or float. whole image height in px.
    :param filename: string.
    :param category: int. category index.
    :return:
    """
    rectangle_size = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])

    # calculating center and normalizing values
    norm_box_width = (rectangle_size[0]) / width  # (x1 - x0)
    norm_box_height = (rectangle_size[1]) / height  # (y1 - y0)
    x_center = (coords[0][0] + rectangle_size[0] / 2) / width  # (x0 + box_width/2)
    y_center = (coords[0][1] + rectangle_size[1] / 2) / height  # (y0 + box_width/2)

    # writing the coords to the correct label file.
    with open(filename + ".txt", "a") as label:
        line = str(category) + " " + str(x_center) + " " + str(y_center) + " " + str(norm_box_width) + " " + str(
            norm_box_height) + "\n"
        label.write(line)
        label.close()


def remove_box(pos, label_file, width, height):
    """
    The user can remove a box by right-clicking on it.
    :param pos: tuple (x,y). describes where the user clicked.
    :param label_file: string. name of label_file to edit.
    :param width: int. whole image.
    :param height int. whole image.
    :return:
    """
    with open(label_file, "r") as l:
        box_coords = l.readlines()
        l.close()

    for ind, box in enumerate(box_coords):
        box = box.split(" ")
        if len(box) == 1:
            box = box[0].split("\t")

        top_left, bottom_right = centers_to_corners(box, width, height)

        if top_left[0] < float(pos[0]) < bottom_right[0] and top_left[1] < float(pos[1]) < bottom_right[1]:
            box_coords.pop(ind)
            print("Line removed from box_coords.")
            break

    with open(label_file, "w") as l:
        l.writelines(box_coords)

    print("Box removed.")


class Application:

    def __init__(self, class_names, addresses, names_file=r'yolov3\data\networkeq.names', label_box=False,
                 skip_annotated=True, our_data_only=True):
        self.addresses = addresses
        self.our_data_only = our_data_only     # if True, index annotations from 0. If false, index annotations from 80.
        self.class_names = class_names
        self.names_file = names_file
        self.label_box = label_box
        self.skip_annotated = skip_annotated
        self.inds = list(range(0, len(class_names)))

    def define_class_names(self, names_file):
        """
        Edits variable self.class_names
        :param names_file:
        :return:
        """
        self.class_names = []
        with open(names_file) as n:
            names = n.readlines()
            for name in names:
                name = name.rstrip()
                self.class_names.append(name)

    def show_menu(self):
        """A new window. Constantly open during main().
        Self-upgrading when a class is removed.
        USES KIVY."""
        pass

    def add_class(self, class_name):
        """
        :param class_name: string. class name.
        :return:
        Adds one class to the END of self.class_name.
        """
        # editing self.names_file, self.class_names and self.inds
        with open(self.names_file) as n:
            lines = n.readlines()
            lines.append("\n" + class_name)
            n.close()
        with open(self.names_file, "w") as n:
            n.writelines(lines)
            n.close()

        self.class_names.append(class_name)
        self.inds = list(range(0, len(self.class_names)))

    def rename_class(self, ind, new_name):
        """
        :param ind: int. index of the class being renamed.
        :param new_name: string. the new name for the class.
        :return:
        """
        with open(self.names_file) as n:
            lines = n.readlines()
            lines[ind] = new_name + "\n"
            n.close()
        with open(self.names_file, "w") as n:
            n.writelines(lines)
            n.close()

        self.define_class_names(self.names_file)

    def merge_classes(self, merge_list, new_name):
        """
        :param merge_list: List of integers. Must all be in range of len(self.class_names)
        :param new_name: String. The name of the new class.
        Copies all boxes where line[0] in merge_list[:-1], with line[0] = merge_list[-1]
        then, for ind = merge_list[:-1], self.remove_class(ind)
        New class index is the index of the last class in merge_list.
        :return:
        """
        # RENAMING THE CLASS
        ind = merge_list[-1]
        self.rename_class(ind, new_name)

        # COPYING & REMOVING.
        for address in self.addresses:

            # COPYING BOXES
            for dirpath, dirnames, filenames in os.walk(address):
                for filename in filenames:
                    file_ad = path.join(dirpath, filename)
                    pre, ext = path.splitext(file_ad)

                    # if text file is empty, skip it
                    if os.stat(pre + ".txt").st_size == 0:
                        continue

                    # if text file doesn't contain any relevant boxes, skip it
                    elif not self.is_found(pre + ".txt", merge_list[:-1]):
                        continue

                    # if the txt file passes all tests, it is opened
                    else:
                        newlines = []
                        with open(pre + ".txt", "r") as l:
                            lines = l.readlines()
                            for line in lines:
                                line = line.split(" ")
                                if len(line) == 1:
                                    line = line[0].split("\t")

                                # creating a new line with class merge_list[-1], and adding it to the txt file.
                                if line[0] in merge_list[:-1]:
                                    newline = line
                                    newline[0] = str(merge_list[-1])
                                    newline = " ".join(newline)
                                    newlines.append(newline)
                        # appending newlines to lines.
                        with open(pre + ".txt", "a") as l:
                            l.writelines(newlines)

            # REMOVING CLASSES
            for ind in merge_list[:-1]:
                self.remove_class(ind)

    def is_found(self, file_ad, show_class):
        """
        :param file_ad: string. .txt file address to look through.
        :param show_class: int list []. list of desired classes. can't be blank.
        :return:
        Checks if the .txt file contains boxes of the desired classes.
        """
        if not show_class:
            print("***CATCH ERROR HERE*** Application.is_found ERROR: show_class is blank [].")
            return None

        found = False
        with open(file_ad, "r") as l:
            lines = l.readlines()
            for ind, line in enumerate(lines):
                line = line.split(" ")
                if len(line) == 1:
                    line = line[0].split("\t")

                if int(line[0]) in show_class:
                    found = True
                    return found
        return found

    def remove_class(self, ind):
        """
        Removes one class and adjusts all label files to their new class numbers.
        :param ind: int. index of removed class in self.class_names
        :return:
        Removes a class AND all its boxes from the dataset.
        Creates backups -->
        """
        # error check
        if ind >= len(self.class_names) or ind < 0 or type(ind) != 'int':
            print("***CATCH ERROR HERE*** class number out of range.")

        # removing the unwanted class from names_file.
        with open(self.names_file) as n:
            lines = n.readlines()
            lines.pop(ind)
            n.close()
        with open(self.names_file, "w") as n:
            n.writelines(lines)
            n.close()

        self.class_names.pop(ind)
        self.inds = list(range(0, len(self.class_names)))

        for address in self.addresses:
            for dirpath, dirname, filenames in os.walk(address):
                for filename in filenames:
                    file_ad = path.join(dirpath, filename)
                    pre, ext = path.splitext(file_ad)
                    if ext == ".txt":
                        # if text file isn't empty
                        if os.stat(pre + ".txt").st_size != 0:

                            with open(pre + ".txt") as l:
                                boxes = l.readlines()
                                for box in enumerate(boxes):
                                    box_ind = box[0]
                                    box = box[1]

                                    # splitting the line
                                    box = box.split(" ")
                                    if len(box) == 1:
                                        box = box[0].split("\t")

                                    # getting classname
                                    current_ind = int(box[0])

                                    # each case
                                    if current_ind == ind:    # remove the box.
                                        box = ''

                                    elif current_ind > ind:   # decrease current_ind by one.
                                        current_ind -= 1
                                        box[0] = str(current_ind)

                                    # join box
                                    box = " ".join(box)

                                    # add adjusted line back.
                                    boxes[box_ind] = box
                                l.close()

                            boxes = [a for a in boxes if a != '']
                            print("BOXES (NEW) =", boxes)

                            with open(pre + ".txt", "w") as l:
                                 l.writelines(boxes)

    def change_box_class(self, pos, label_file, width, height):
        """
        :param pos: tuple list [(x0, y0), (x1, y1)]
        :param label_file: string. file you're editing.
        :param width: int. width of whole image.
        :param height: int. width of whole image.
        :return:
        """

        # showing classes.
        new_class = ''
        prompt = '===Press C to cancel this box===\n'
        for word in enumerate(self.class_names):
            prompt += str(word[0] + self.add) + ": " + word[1] + "\n"
        while new_class not in ele_to_string(self.inds) and new_class not in ["C", "c"]:
            new_class = int(input("Insert the new class number."))

        with open(label_file, "r") as l:
            box_coords = l.readlines()
            l.close()

        for ind, box in enumerate(box_coords):
            box = box.split(" ")
            if len(box) == 1:
                box = box[0].split("\t")

            top_left, bottom_right = centers_to_corners(box, width, height)

            if top_left[0] < float(pos[0]) < bottom_right[0] and top_left[1] < float(pos[1]) < bottom_right[1]:
                print("OLD BOX = ", box)
                box[0] = str(new_class)
                box = " ".join(box)
                print("NEW BOX = ", box)
                box_coords[ind] = box
                print("Line edited in box_coords.")

                with open(label_file, "w") as l:
                    l.writelines(box_coords)

    def en_masse_change_box_class(self, address, og_class, change):
        """
        :param address: string. Address
        :param og_class: int. class to change from.
        :param change: int. class to change to.
        :return:
        Useful if you group your data by filenames.
        i.e. you name all your USB images with "USB" in them, but class them as "plugged cable" or "unplugged".
        If you add the USB class, you can easily put USB imgs in their own folder and run this function to change them.
        """

        if not len(self.class_names) < og_class < 0 or not len(self.class_names) < change < 0:
            print("***CATCH ERROR HERE*** input classes out of range. og_class = ", og_class, ", change =", change,
                  ", len(self.class_names) = ", len(self.class_names))
            return None

        for dirpath, dirnames, filenames in os.walk(address):
            for filename in filenames:
                file_ad = path.join(dirpath, filename)
                pre, ext = path.splitext(file_ad)
                if ext == ".txt":
                    with open(file_ad, "r") as l:
                        boxes = l.readlines()
                        l.close()

                    for ind, box in enumerate(boxes):
                        box = box.split(" ")
                        if len(box) == 1:
                            box = box[0].split("\t")

                        if box[0] == str(og_class):
                            box[0] = str(change)

                        box = " ".join(box)
                        boxes[ind] = box

                        with open(file_ad, "w") as l:
                            l.writelines(boxes)

    def show_boxes(self, surface, label_file, width, height, show_class=[]):
        """
        :param surface: pygame object.
        :param label_file: string. address to text file.
        :param width: int or float. whole image width.
        :param height: int or float. whole image height.
        :param show_class: int list containing classes to show. [] if all classes are shown.
        :return:
        """
        with open(label_file, "r") as l:
            boxes = l.readlines()
            for ind, box in enumerate(boxes):

                # split into a list
                box = box.split(" ")
                if len(box) == 1:
                    box = box[0].split("\t")

                # translate coords from label_file, only if it's the desired class.
                if not show_class or int(box[0]) in show_class:
                    box[3] = float(box[3]) * width     # box_width
                    box[4] = float(box[4]) * height    # box_height
                    box[1] = (float(box[1]) * width) - float(box[3]) / 2  # x_corner
                    box[2] = (float(box[2]) * height) - float(box[4]) / 2  # y_corner

                    # add it back to the list, edited
                    boxes[ind] = box

        # storage variable for return
        return_boxes = boxes

        # draw rectangle
        for box in boxes:
            # only if it's the desired class.
            if not show_class or int(box[0]) in show_class:
                # rectangle_size
                x = round(float(box[1]))
                y = round(float(box[2]))
                rectangle_size = (round(float(box[3])), round(float(box[4])))
                pygame.draw.rect(surface, green, [x, y, rectangle_size[0], rectangle_size[1]], 3)

                if self.label_box:
                    # getting classname
                    ind = int(box[0])
                    text = self.class_names[ind]
                    class_rectangle_size = (100, 30)
                    pygame.draw.rect(surface, green, [x, y + rectangle_size[1], class_rectangle_size[0], class_rectangle_size[1]], 3)

                    # writing classname
                    font = pygame.font.SysFont('comicsans', 20)
                    text = font.render(text, 1, (0, 0, 0))
                    surface.blit(text, (x, y + rectangle_size[1]))
        return return_boxes

    def record_metadata(self):
        """
        Speeds up the process of viewing images.
        Keeps data in JSON string.
        {
            {
            'class': int,
            'class_count': int,
            'found_in_img': int,
            'imgs': [vector of strings, all image addresses this class can be found in.]
            }
            ...
            {
            'class': int,
            'class_count': int,
            'found_in_img': int,
            'imgs': [vector of strings, all image addresses this class is found in.]
            }
            ... * len(classes)
        }
        :return:
        """
        pass

    def get_info(self, addresses=['Dataset']):
        """
        :param addresses: string list []. directories to go through.
        :return:
        """
        class_count = [0]*len(self.class_names)
        found_in_img = [0]*len(self.class_names)

        for address in addresses:
            for dirpath, dirnames, filenames in os.walk(address):
                for filename in filenames:
                    file_ad = path.join(dirpath, filename)
                    pre, ext = path.splitext(file_ad)
                    if ext != ".txt":
                        continue
                    else:
                        with open(file_ad, "r") as f:
                            lines = f.readlines()
                            added = False
                            for line in lines:

                                # splitting line
                                line = line.split(" ")
                                if len(line) == 1:
                                    line = line[0].split("\t")

                                # extracting class number
                                ind = int(line[0])

                                # adding points to class_count
                                class_count[ind] += 1

                                # adding points to img if not added already
                                if not added:
                                    found_in_img[ind] += 1
        return class_count, found_in_img

    def show_info(self, addresses=[]):
        """
        :param addresses: string list []. directories to go through.
        :return:
        """
        for address in addresses:
            class_count, found_in_img = self.get_info(address)
            title = address
            d = {'class count': class_count,
                 'found in x images': found_in_img}
            df = pd.DataFrame(data=d)
            print("DATA AT: " + title + "\n" + df.to_string())

    def main(self, show_class=[]):
        """
        :param addresses: string list []. directories of the dataset(s) you'll annotate.
        :param show_class: int list []. classes to show. If [], all classes are shown.
        :return:
        """

        # initializing coords, box_ops, pos, category and add
        coords = []
        pos = []
        box_ops = []
        removed = []

        # add relates to indices
        if self.our_data_only:
            self.add = 0
        else:
            self.add = 80

        # looping through the dataset
        imgs = []

        for address in self.addresses:
            for dirpath, dirnames, filenames in os.walk(address):
                for filename in filenames:
                    file_ad = path.join(dirpath, filename)

                    # checking file extension; must be image to proceed
                    pre, ext = path.splitext(file_ad)
                    if ext not in [".txt", ".zip"] and path.isfile(file_ad):
                        # check if text file is blank.
                        if os.stat(pre + ".txt").st_size != 0 and self.skip_annotated:
                            print("File already annotated.")
                            continue

                        # check text file for desired boxes.
                        elif show_class:
                            found = self.is_found(pre + ".txt", show_class)
                            if not found:
                                print("Desired box(es) not found.")
                                continue
                            else:
                                imgs.append(file_ad)

                        # include image for annotation if its label file passes the above tests.
                        else:
                            imgs.append(file_ad)
                        print(filename)

            i = 0
            while len(imgs) > i >= 0:
                # initializing
                file_ad = imgs[i]
                pre, ext = path.splitext(file_ad)
                img = pygame.image.load(file_ad)
                width, height = img.get_size()

                # creating window:
                window = pygame.display
                surface = window.set_mode((width, height), pygame.RESIZABLE)
                window.set_caption(file_ad)
                close = False

                while not close:
                    # === EVENT HANDLING === #
                    for event in pygame.event.get():
                        if event not in [{}, []]:
                            # MOUSE BUTTONS
                            if event.type == 6:     # mouseButtonUp --> click of any button. first requires mousebuttondown.
                                pos = event.dict["pos"]
                                print("POS = ", pos)

                                if event.dict["button"] == 3:   # right click
                                    box_ops = [0]
                                else:
                                    box_ops = []    # cancels any right-click operations we had.
                                    if len(coords) < 3:
                                        coords.append(pos)
                                    else:
                                        coords = []     # resetting coords
                                        print("Coords has been reset.")
                                        coords.append(pos)

                            # KEY PRESSES
                            if event.type == 2:  # KEYDOWN

                                # ARROW KEYS
                                if event.dict["unicode"] == '':

                                    # RIGHT ARROW
                                    if event.dict["key"] == 275:
                                        # go forward
                                        coords = []
                                        close = True
                                        if i < len(imgs) - 1:
                                            i += 1
                                        continue

                                    # LEFT ARROW
                                    elif event.dict["key"] == 276:
                                        # go back
                                        coords = []
                                        close = True
                                        if i > 0:
                                            i -= 1
                                        continue

                                # NON-ARROW keys
                                # M KEY: Merge classes
                                if event.dict["unicode"] == "m":
                                    prompt = "===Press C to cancel===\n"
                                    for word in enumerate(self.class_names):
                                        prompt += str(word[0]) + ": " + word[1] + "\n"

                                    # merge: change from a string to a list of numbers
                                    merge = input("Choose which classes numbers to merge, seperated by commas: ")
                                    merge.split(",")

                                    for ind, ele in enumerate(merge):
                                        ele = ele.strip()
                                        ele = int(ele)  # changing it to a number
                                        if ele not in range(0):
                                            print("***CONTINUE ERROR:*** Number " + str(ele) + " out of range. Start again.")
                                            continue
                                        merge[ind] = ele

                                    if len(merge) <= 1:
                                        print("***CONTINUE ERROR:*** You need more than one number. Start again.")
                                        continue

                                    new_name = input("Name your merged class: ")

                                    self.merge_classes(merge, new_name)

                                # C KEY: Cancels / resets all operations.
                                if event.dict["unicode"] == "c":    # c-key
                                    if coords:
                                        coords = []
                                    if box_ops:
                                        box_ops = []

                                # I KEY: Shows info about the dataset
                                if event.dict["unicode"] == "i":
                                    """In Progress"""
                                    # self.show_info()
                                    pass

                                # D KEY: Delete a class
                                if event.dict["unicode"] == "d":
                                    prompt = "===Press C to cancel===\n"
                                    for word in enumerate(self.class_names):
                                        prompt += str(word[0]) + ": " + word[1] + "\n"
                                    ind = input(prompt + "Select a class to remove: ")

                                    if ind in ele_to_string(list(range(0, len(self.class_names)))):
                                        ind = int(ind)
                                        self.remove_class(ind)

                                    elif ind in ["C, c"]:
                                        continue

                                # A KEY: Add a class
                                if event.dict["unicode"] == "a":
                                    class_name = input("Name your new class: ")
                                    self.add_class(class_name)

                                # IF RIGHT CLICKED:
                                if box_ops == [0]:

                                    # RIGHT CLICK + D-KEY: Deletes whole image
                                    if event.dict["unicode"] == "d":
                                        status = ''
                                        while status not in ["Y", "y", "N", "n"]:
                                            status = input("Are you sure you want to delete this image? [Y/N]: ")

                                            # flag image for removal if Y
                                            if status in ["Y", "y"]:
                                                if file_ad not in removed:
                                                    removed.append(file_ad)
                                                    removed.append(pre + ".txt")

                                            # continue if N
                                            elif status in ["N", "n"]:
                                                continue

                                    # RIGHT CLICK + U-KEY: Restores image flagged for removal
                                    if event.dict["unicode"] == "u":
                                        if file_ad in removed:
                                            removed.remove(file_ad)
                                            removed.remove(pre + ".txt")

                                    # RIGHT CLICK + E-KEY: Edits box
                                    if event.dict["unicode"] == "e":
                                        print("Changing box class...")
                                        self.change_box_class(pos, pre + ".txt", width, height)

                                    # RIGHT CLICK + R-KEY: Removes box
                                    """In Progress. Fix bug where it removes the one outside."""
                                    if event.dict["unicode"] == "r":
                                        if not pos:
                                            print("WARNING: pos is blank. remove_box has no effect.")
                                        remove_box(pos, pre + ".txt", width, height)
                                        box_ops = []

                            if event.type == pygame.QUIT:
                                # keep track of which image we're currently annotating before we close.
                                with open("last_ad.txt", "w+") as last_ad:
                                    last_ad.write(file_ad)

                                # removes any images we flagged for removal.
                                for file in removed:
                                    os.remove(file)

                                sys.exit()  # exits the whole program

                    # if we have a valid bounding box in the format (top_left, bottom_right)...
                    if len(coords) == 2 and coords[0][0] < coords[1][0] and coords[0][1] < coords[1][1]:
                        # drawing the box
                        rectangle_size = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])     # (x1-x0, y1-y0)
                        pygame.draw.rect(surface, green, [coords[0][0], coords[0][1], rectangle_size[0], rectangle_size[1]], 3)
                        window.update()

                        # showing the number options
                        category = ''
                        prompt = '===Press C to cancel this box===\n'
                        for word in enumerate(self.class_names):
                            prompt += str(word[0] + self.add) + ": " + word[1] + "\n"
                        while category not in ["C", "c"] and category not in ele_to_string(self.inds) and len(coords) == 2:
                            category = input(prompt + "Which class is this?")

                        # once labelled, verify depending on if its C or not.
                        if category in ele_to_string(self.inds):
                            category = int(category)
                            write_to_file(coords, width, height, pre, category)
                        else:
                            category = ''

                        # reset coords after processing the box
                        coords = []

                    # if coordinates are not [top_left, bottom_right], invalid box. reset coords.
                    elif len(coords) == 2 and (coords[0][0] > coords[1][0] or coords[0][1] > coords[1][1]):
                        coords = []
                        print("Invalid box: Coords has been reset.\n"
                              "Please click on the top-left and bottom-right corners, in order.\n")

                    # constant operations
                    surface.fill(white)
                    surface.blit(img, (0, 0))
                    label_file = pre + ".txt"
                    self.show_boxes(surface, label_file, width, height, show_class)
                    window.update()


def convert_to_pretrained(og_data_address, save_address, copy=True):
    """
    IN PROGRESS
    Assume you have 15 classes in your new dataset.
    Use when you want to convert your data from classes 0-15 to classes 80-95.
    You'd use this when pre-training with COCO.
    :param og_data_address: string, the data you want to convert.
    :param save_address: string, where you want to save.
    :return:
    """
    # copy the whole directory to save_address
    if copy:
        copytree(og_data_address, save_address)

    # starting os.walk --> looping through whole dataset
    for dirpath, dirnames, filenames in os.walk(save_address):
        # looping through the images
        for filename in filenames:
            file_ad = path.join(dirpath, filename)

            # checking file extension; must be .txt to proceed
            pre, ext = path.splitext(file_ad)
            if path.isfile(file_ad) and ext == ".txt":
                # open file as a+: reading and appending
                with open(file_ad, 'r') as f:
                    lines = f.readlines()
                    for line in enumerate(lines):
                        ind = line[0]
                        arr = line[1].split()
                        classnum = arr[0]
                        fix = int(classnum) + 80
                        arr[0] = str(fix)
                        newline = ' '.join(arr)
                        print("NEWLINE = ", newline)
                        lines[ind] = newline
                    f.close()
                print("filename = ", file_ad, ", LINES = ", lines)
                with open(file_ad, 'w') as f:
                     f.writelines(lines)


def get_labelled(address):
    """
    :param address:
    :return: List of labelled image files + list of labelled text files.
    [(label address, label filename), (label address, label filename)........ (label address, label filename)]
    """
    imgs = []
    labels = []

    # looping through the data
    for dirpath, dirnames, filenames in os.walk(address):

        # checking file by file
        for filename in filenames:
            file_ad = path.join(dirpath, filename)

            # checking file extension; must be image to proceed
            pre, ext = path.splitext(file_ad)
            if ext != ".txt" and path.isfile(file_ad):

                # if file is not empty
                if os.stat(pre + ".txt").st_size != 0:
                    imgs.append((filename, file_ad))
                    pre2, ext2 = path.splitext(filename)
                    labels.append((pre2 + ".txt", pre + ".txt"))
    return imgs, labels


def write_to_darknet(data_folder, img_folder, label_folder, text_file, overwrite=True):
    """
    :param data_folder: Original dataset. (String)
    :param img_folder: Target image address. (string)
    :param label_folder: Target label address. (String)
    :param text_file: Lists all the new image addresses. (string)
    :param overwrite: NOT YET WORKING
    :return:
    """
    imgs, labels = get_labelled(data_folder)
    print("IMGS: ", imgs)
    print("LABELS: ", labels)

    if len(imgs) != len(labels):
        print("***CATCH ERROR HERE*** len(imgs) != len(labels)")

    for img in imgs:
        src = img[1]
        dest = path.join(img_folder, img[0])
        print("SRC = ", src)
        print("DEST = ", dest)
        copyfile(src, dest)

    for label in labels:
        src = label[1]
        dest = path.join(label_folder, label[0])
        print("SRC = ", src)
        print("DEST = ", dest)
        copyfile(src, dest)

    if overwrite:
        status = "w"    # clear folder and write anew.
    else:
        status = "a"    # append.

    with open(text_file, status) as text:
        for dirpath, dirnames, filenames in os.walk(img_folder):
            for file in filenames:
                abs_dir = path.join(path.abspath(path.curdir), dirpath)
                file_ad = path.join(abs_dir, file)
                print("FILEAD =", file_ad)
                text.write(file_ad + "\n")


def clean_duplicate_imgs(address, remove=False):
    """
    ASSUMES ONLY ONE DUPLICATE
    :param address: string, the file address you want
    :param remove: bool, True if files get deleted, False if one gets renamed.
    :return:
    """
    for dirpath, dirnames, filenames in os.walk(address):
        found = []
        for file in filenames:
            file_ad = path.join(dirpath, file)
            pre, ext = path.splitext(file_ad)
            if pre in found:
                if remove:
                    os.remove(file_ad)
                else:
                    os.rename(file_ad, pre + "_renamed.txt")
            else:
                found.append(pre)


def change_jfif(address, change=".png"):
    for dirpath, dirnames, filenames in os.walk(address):
        for filename in filenames:
            file_ad = path.join(dirpath, filename)
            pre, ext = path.splitext(file_ad)
            if ext in [".jfif", ".gif"]:
                os.rename(file_ad, pre + change)


def empty_files(address):
    for dirpath, dirnames, filenames in os.walk(address):
        for filename in filenames:
            file_ad = path.join(dirpath, filename)
            pre, ext = path.splitext(file_ad)
            if ext == ".txt":
                with open(file_ad, "w") as f:
                    if os.stat(pre + ".txt").st_size != 0:
                        print("WARNING: File was not emptied!")
                    f.close()


# def get_dataset_addresses():
#     """
#     Uses argv to get dataset address.
#     :return:
#     """
#     addresses = []
#     address = ''
#     while address not in ["N", "n", "Q", "q"]:
#         address = input("Enter the address (press N or Q to exit): ")
#         addresses.append(address)
#     return addresses
#
#
# # defining addresses
# addresses = get_dataset_addresses()     # dir

def create_label_files(addresses):
    """
    Creates corresponding .txt files for images in the dataset.
    (i.e. 0001.png, 0002.png ==> 0001.txt, 0002.txt)
    :param addresses: [ ] of strings
    :return:
    """
    for address in addresses:
        for dirpath, dirnames, filenames in os.walk(address):
            files = os.listdir(dirpath)
            for file in files:
                # create full file name
                this_dir = os.path.join(dirpath, file)

                # creating .txt file IF it doesn't already exist
                if path.isfile(this_dir):
                    pre, ext = path.splitext(this_dir)  # splits filepath into stem and extension.
                    open(pre + ".txt", 'a').close()  # creates a .txt file with the same name, different extension.

"""
    ADDRESSES TO FILES MANUALLY ENTERED HERE.
"""

addresses = [r'C:\Users\Sarah\Documents\GitHub\Visual-Mathematical-Equations-VME\photos']
create_label_files(addresses)
names_file = r'C:\Users\Sarah\Documents\GitHub\Visual-Mathematical-Equations-VME\formats\darknet\names.txt'

class_names = []
with open(names_file) as n:
    names = n.readlines()
    for name in names:
        name = name.rstrip()
        class_names.append(name)

print(class_names)

app = Application(class_names, addresses, names_file, False, False)
app.main()

# for address in addresses:
#     write_to_darknet(address, train_img_address, train_label_address, image_list_file, True)

# train_img_address = r''             # dir
# train_label_address = r''           # dir
# image_list_file = r''               # always .txt
# convert_to_pretrained('NBN Dataset - Our Data Only', 'NBN Dataset - Pre-trained', False)


# IN PRESENTATION, EXPAND ON:
    # RANDOM FOREST
    # https://github.com/tzutalin/labelImg
    # Send Balaji a link to the correct dataset.
