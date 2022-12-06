import json
import glob
import cv2 as cv2
import math
import copy
import numpy as np

def lastChar(s, c):
    return -(s[::-1].index(c) + 1)

class MaskFill:
    maskedImage = None

    # {
    #     "shape_attributes": {
    #         "name": "ellipse",
    #         "cx": 2326,
    #         "cy": 1172,
    #         "rx": 108.738,
    #         "ry": 115.355,
    #         "theta": -2.664
    #     },
    #     "region_attributes": {
    #         "Classification": "Label"
    #     }
    # },

    def ellipseMask(self, region):
        x = region["shape_attributes"]["cx"]
        y = region["shape_attributes"]["cy"]
        rx = region["shape_attributes"]["rx"]
        ry = region["shape_attributes"]["ry"]
        theta = region["shape_attributes"]["theta"]
        self.maskedImage = cv2.ellipse(self.maskedImage, (int(x), int(y)), (int(rx), int(ry)), math.degrees(theta), 0, 360, self.fillColor, -1)

    # {
    #     "shape_attributes": {
    #         "name": "circle",
    #         "cx": 2569,
    #         "cy": 2790,
    #         "r": 82.503
    #     },
    #     "region_attributes": {
    #         "Type": "Button"
    #     }
    # },

    def circleMask(self, region):
        x = region["shape_attributes"]["cx"]
        y = region["shape_attributes"]["cy"]
        r = region["shape_attributes"]["r"]
        self.maskedImage = cv2.circle(self.maskedImage, (int(x), int(y)), int(r), self.fillColor, -1)

    # {
    #     "shape_attributes": {
    #         "name": "polygon",
    #         "all_points_x": [
    #             1565,
    #             1438,
    #             1439,
    #             1563
    #         ],
    #         "all_points_y": [
    #             2651,
    #             2651,
    #             2768,
    #             2769
    #         ]
    #     },
    #     "region_attributes": {
    #         "Type": "button"
    #     }
    # },

    # {
    #     "shape_attributes": {
    #         "name": "polyline",
    #         "all_points_x": [
    #             2232,
    #             2223,
    #             2434,
    #             2437,
    #             2232
    #         ],
    #         "all_points_y": [
    #             2677,
    #             2879,
    #             2888,
    #             2684,
    #             2675
    #         ]
    #     },
    #     "region_attributes": {
    #         "Type": "Label"
    #     }
    # },

    # np.array([[220, 120], [130, 200], [130, 300],
    #                [220, 380], [310, 300], [310, 200]])

    def polyMask(self, region):
        points = np.array([list(a) for a in zip(region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"])])
        print(points)
        self.maskedImage = cv2.fillPoly(self.maskedImage, pts=[points], color=self.fillColor)

    # {
    #     "shape_attributes": {
    #         "name": "rect",
    #         "x": 1362,
    #         "y": 2038,
    #         "width": 86,
    #         "height": 84
    #     },
    #     "region_attributes": {
    #         "Type": "label"
    #     }
    # },

    def rectMask(self, region):
        x = region["shape_attributes"]["x"]
        y = region["shape_attributes"]["y"]
        height = region["shape_attributes"]["height"]
        width = region["shape_attributes"]["width"]
        self.maskedImage = cv2.rectangle(self.maskedImage, (int(x), int(y)), (int(x + width), int(y + height)), self.fillColor, -1)

    def __init__(self, fillColor = (255, 255, 255)):
        self.fillColor = fillColor

    def __call__(self, val, path, imgName, image):
        jsonArr = []

        blockRegions = []

        for region in val["regions"]:
            blockRegions.append([region["shape_attributes"]["name"], region])
            # if region["shape_attributes"]["name"] == "rect":
            #     blockRegions.append(["rect", region])
            # elif region["shape_attributes"]["name"] == "ellipse":
            #     ellipseRegions.append(region)
            # elif region["shape_attributes"]["name"] == "circle":
            #     circleRegions.append(region)
            # elif region["shape_attributes"]["name"] == "polyline" or region["shape_attributes"]["name"] == "polygon":
            #     polyRegions.append(region)
        print("\n\nDone Regioning\n\n\n")

        # print(rectRegions)
        # print(ellipseRegions)
        # print(circleRegions)
        # print(polyRegions)

        fig = -1

        print(str(fig + 1) + "miss" + key)

        self.maskedImage = copy.deepcopy(image)

        try:
            value = copy.deepcopy(val)

            value["filename"] = str(fig + 1) + "miss" + imgName

            for i in range(len(blockRegions)):
                if blockRegions[i][0] == "rect":
                    self.rectMask(blockRegions[i][1])
                elif blockRegions[i][0] == "ellipse":
                    self.ellipseMask(blockRegions[i][1])
                elif blockRegions[i][0] == "circle":
                    self.circleMask(blockRegions[i][1])
                elif blockRegions[i][0] == "polyline" or blockRegions[i][0] == "polygon":
                    self.polyMask(blockRegions[i][1])
                    
            cv2.imwrite(path + value["filename"], self.maskedImage)
            
            value["regions"] = []

            jsonArr.append(value)

        except Exception as e:
            print(e)



        for fig, det in enumerate(blockRegions):
            print(str(fig + 1) + "miss" + key)

            self.maskedImage = copy.deepcopy(image)

            try:
                value = copy.deepcopy(val)
                # print(value)

                value["filename"] = str(fig + 1) + "miss" + imgName

                newShape = det[1]

                for i in range(len(blockRegions)):
                    if i != fig:
                        if blockRegions[i][0] == "rect":
                            self.rectMask(blockRegions[i][1])
                        elif blockRegions[i][0] == "ellipse":
                            self.ellipseMask(blockRegions[i][1])
                        elif blockRegions[i][0] == "circle":
                            self.circleMask(blockRegions[i][1])
                        elif blockRegions[i][0] == "polyline" or blockRegions[i][0] == "polygon":
                            self.polyMask(blockRegions[i][1])
                        
                cv2.imwrite(path + value["filename"], self.maskedImage)
                
                value["regions"] = [newShape]
                # print(value)
                
                # cv2.imshow("Temp", self.maskedImage)
                # cv2.waitKey()

                jsonArr.append(value)

            except Exception as e:
                print(e)
        # print(jsonArr)

        return jsonArr

if __name__ == "__main__":
    buildingDirectories = glob.glob("./*/")
    
    for path in buildingDirectories:
        pictures = glob.glob(path+"*.jpg")
        jsonFile = glob.glob(path+"*.json")
        filler = MaskFill((255, 255, 255))
        if len(jsonFile) > 0:
            with open(jsonFile[0], "r") as read_file:
                data = copy.deepcopy(json.load(read_file))

                print(data)


                picJSON = copy.deepcopy(data["_via_img_metadata"])
                data["_via_img_metadata"].clear()
                data["_via_image_id_list"] = []

                for pic in pictures:
                    image = cv2.imread(pic)
                    
                    tempPic = pic[lastChar(pic, "\\") + 1:]
                    for key, value in picJSON.items():
                        if tempPic in key:
                            print(key)
                            for i, d in enumerate(filler(value, path, tempPic, image)):
                                data["_via_img_metadata"][str(i) + "miss" + key] = d
                                if data["_via_image_id_list"].count(str(i) + "miss" + key) == 0:
                                    data["_via_image_id_list"].append(str(i) + "miss" + key)
                            break






                with open(jsonFile[0][:-5] + "Missing.json", 'w') as newFile:
                    json.dump(data, newFile)

            read_file.close()