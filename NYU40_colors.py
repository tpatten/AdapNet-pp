# color palette for nyu40 labels (correspond to ScanNetV2 ground truth labels)
def create_color_palette_gt():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves
       (219, 219, 141),		# curtain
       (140, 57, 197),      # dresser
       (202, 185, 52),      # pillow
       (51, 176, 203),      # mirror
       (200, 54, 131),      # floor mat
       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # television
       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
       (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
       (178, 127, 135),     # whiteboard
       (120, 185, 128),     # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp
       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # otherstructure
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # otherprop
    ]

#color pallete for nyu40 labels (corresponding to the legend image provided by ScanNet)
def create_color_palette_legend():
   return [
      (0, 0, 0),
      (190, 153, 112),  # wall
      (189, 198, 255),  # floor
      (213, 255, 0),  # cabinet
      (158, 0, 142),  # bed
      (152, 255, 82),  # chair
      (119, 77, 0),  # sofa
      (122, 71, 130),  # table
      (0, 174, 126),  # door
      (0, 125, 181),  # window
      (0, 143, 156),  # bookshelf
      (107, 104, 130),  # picture
      (255, 229, 2),  # counter
      (117, 68, 177),  # blinds
      (1, 255, 254),  # desk
      (0, 21, 68),  # shelves
      (255, 166, 254),  # curtain
      (194, 140, 159),  # dresser
      (98, 14, 0),  # pillow
      (0, 71, 84),  # mirror
      (255, 219, 102),  # floor mat
      (0, 118, 255),  # clothes
      (67, 0, 44),  # ceiling
      (1, 208, 255),  # books
      (232, 94, 190),  # refrigerator
      (145, 208, 203),  # television
      (255, 147, 126),  # paper
      (95, 173, 78),  # towel
      (0, 100, 1),  # shower curtain
      (255, 238, 232),  # box
      (0, 155, 255),  # whiteboard
      (255, 0, 86),  # person
      (189, 211, 147),  # night stand
      (133, 169, 0),  # toilet
      (149, 0, 58),  # sink
      (255, 2, 157),  # lamp
      (187, 136, 0),  # bathtub
      (0, 185, 23),  # bag
      (1, 0, 103),  # otherstructure
      (0, 0, 255),  # otherfurn
      (255, 0, 246)  # otherprop
   ]

