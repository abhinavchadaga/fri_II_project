import json
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import math
import copy


def lastChar(s, c):
    return -(s[::-1].index(c) + 1)


class PerspectiveTransform:
    transformedIMG = None
    tempIMG = None

    tIMG = None
    # draw = None

    def transformedPoint(self, x, y, angle, cy, cx):
        self.tempIMG = torch.zeros_like(self.transformedIMG)
        cy = int(cy)
        cx = int(cx)
        self.tempIMG[0][cy][cx] = torch.tensor(255)

        newPoint = transforms.functional.perspective(
            self.tempIMG,
            [[0, 0], [0, x], [y, x], [y, 0]],
            [
                [abs(y * angle), x * angle],
                [abs(y * angle), x - x * angle],
                [y - abs(y * angle), x + x * angle],
                [y - abs(y * angle), -x * angle],
            ],
            fill=(self.fillColor),
        )
        return torch.nonzero(newPoint)[0][1:].tolist()

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

    def ellipseTransform(self, x, y, angle, ellipseRegions):
        for i in range(len(ellipseRegions)):
            soh = math.sin(ellipseRegions[i]["shape_attributes"]["theta"])
            cah = math.cos(ellipseRegions[i]["shape_attributes"]["theta"])
            (cy, cx) = self.transformedPoint(
                x,
                y,
                angle,
                ellipseRegions[i]["shape_attributes"]["cy"],
                ellipseRegions[i]["shape_attributes"]["cx"],
            )
            ellipseRegions[i]["shape_attributes"]["rx"] = math.dist(
                self.transformedPoint(
                    x,
                    y,
                    angle,
                    ellipseRegions[i]["shape_attributes"]["cy"]
                    + (soh * ellipseRegions[i]["shape_attributes"]["rx"]),
                    ellipseRegions[i]["shape_attributes"]["cx"]
                    + (cah * ellipseRegions[i]["shape_attributes"]["rx"]),
                ),
                (cy, cx),
            )
            ellipseRegions[i]["shape_attributes"]["ry"] = math.dist(
                self.transformedPoint(
                    x,
                    y,
                    angle,
                    ellipseRegions[i]["shape_attributes"]["cy"]
                    + (cah * ellipseRegions[i]["shape_attributes"]["ry"]),
                    ellipseRegions[i]["shape_attributes"]["cx"]
                    + (soh * ellipseRegions[i]["shape_attributes"]["ry"]),
                ),
                (cy, cx),
            )
            (
                ellipseRegions[i]["shape_attributes"]["cy"],
                ellipseRegions[i]["shape_attributes"]["cx"],
            ) = (cy, cx)

            # self.draw.line((ellipseRegions[i]["shape_attributes"]["cx"] + (cah * ellipseRegions[i]["shape_attributes"]["rx"]), ellipseRegions[i]["shape_attributes"]["cy"] + (soh * ellipseRegions[i]["shape_attributes"]["rx"]), ellipseRegions[i]["shape_attributes"]["cx"] - (cah * ellipseRegions[i]["shape_attributes"]["ry"]), ellipseRegions[i]["shape_attributes"]["cy"] - (soh * ellipseRegions[i]["shape_attributes"]["ry"])), width = 3)
            # self.draw.line((ellipseRegions[i]["shape_attributes"]["cx"] + (soh * ellipseRegions[i]["shape_attributes"]["rx"]), ellipseRegions[i]["shape_attributes"]["cy"] + (cah * ellipseRegions[i]["shape_attributes"]["rx"]), ellipseRegions[i]["shape_attributes"]["cx"] - (soh * ellipseRegions[i]["shape_attributes"]["ry"]), ellipseRegions[i]["shape_attributes"]["cy"] - (cah * ellipseRegions[i]["shape_attributes"]["ry"])), width = 3)

        # print(ellipseRegions)
        return ellipseRegions

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

    def circleTransform(self, x, y, angle, circleRegions):
        for i in range(len(circleRegions)):
            circleRegions[i]["shape_attributes"]["name"] = "ellipse"
            (cy, cx) = self.transformedPoint(
                x,
                y,
                angle,
                circleRegions[i]["shape_attributes"]["cy"],
                circleRegions[i]["shape_attributes"]["cx"],
            )
            circleRegions[i]["shape_attributes"]["rx"] = (
                self.transformedPoint(
                    x,
                    y,
                    angle,
                    circleRegions[i]["shape_attributes"]["cy"],
                    circleRegions[i]["shape_attributes"]["cx"]
                    + circleRegions[i]["shape_attributes"]["r"],
                )[1]
                - cx
            )
            circleRegions[i]["shape_attributes"]["ry"] = (
                self.transformedPoint(
                    x,
                    y,
                    angle,
                    circleRegions[i]["shape_attributes"]["cy"]
                    + circleRegions[i]["shape_attributes"]["r"],
                    circleRegions[i]["shape_attributes"]["cx"],
                )[0]
                - cy
            )
            (
                circleRegions[i]["shape_attributes"]["cy"],
                circleRegions[i]["shape_attributes"]["cx"],
            ) = (cy, cx)
            circleRegions[i]["shape_attributes"].pop("r")
            circleRegions[i]["shape_attributes"]["theta"] = 0

            # self.draw.line((circleRegions[i]["shape_attributes"]["cx"] - circleRegions[i]["shape_attributes"]["rx"], circleRegions[i]["shape_attributes"]["cy"], circleRegions[i]["shape_attributes"]["cx"] + circleRegions[i]["shape_attributes"]["rx"], circleRegions[i]["shape_attributes"]["cy"]), width = 3)
            # self.draw.line((circleRegions[i]["shape_attributes"]["cx"], circleRegions[i]["shape_attributes"]["cy"] - circleRegions[i]["shape_attributes"]["ry"], circleRegions[i]["shape_attributes"]["cx"], circleRegions[i]["shape_attributes"]["cy"] + circleRegions[i]["shape_attributes"]["ry"]), width = 3)

        # print(circleRegions)
        return circleRegions

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

    def polyTransform(self, x, y, angle, polyRegions):
        for i in range(len(polyRegions)):
            for j in range(len(polyRegions[i]["shape_attributes"]["all_points_x"])):
                (
                    polyRegions[i]["shape_attributes"]["all_points_y"][j],
                    polyRegions[i]["shape_attributes"]["all_points_x"][j],
                ) = self.transformedPoint(
                    x,
                    y,
                    angle,
                    polyRegions[i]["shape_attributes"]["all_points_y"][j],
                    polyRegions[i]["shape_attributes"]["all_points_x"][j],
                )

                # if j > 0:
                #     self.draw.line((polyRegions[i]["shape_attributes"]["all_points_x"][j-1], polyRegions[i]["shape_attributes"]["all_points_y"][j-1], polyRegions[i]["shape_attributes"]["all_points_x"][j], polyRegions[i]["shape_attributes"]["all_points_y"][j]), width = 3)

        # print(polyRegions)
        return polyRegions

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

    def rectTransform(self, x, y, angle, rectRegions):
        for i in range(len(rectRegions)):
            rectRegions[i]["shape_attributes"]["name"] = "polygon"

            pointList = []

            pointList.append(
                self.transformedPoint(
                    x,
                    y,
                    angle,
                    rectRegions[i]["shape_attributes"]["y"],
                    rectRegions[i]["shape_attributes"]["x"],
                )
            )
            pointList.append(
                self.transformedPoint(
                    x,
                    y,
                    angle,
                    rectRegions[i]["shape_attributes"]["y"]
                    + rectRegions[i]["shape_attributes"]["height"],
                    rectRegions[i]["shape_attributes"]["x"]
                    + rectRegions[i]["shape_attributes"]["width"],
                )
            )
            pointList.append(
                self.transformedPoint(
                    x,
                    y,
                    angle,
                    rectRegions[i]["shape_attributes"]["y"],
                    rectRegions[i]["shape_attributes"]["x"]
                    + rectRegions[i]["shape_attributes"]["width"],
                )
            )
            pointList.append(
                self.transformedPoint(
                    x,
                    y,
                    angle,
                    rectRegions[i]["shape_attributes"]["y"]
                    + rectRegions[i]["shape_attributes"]["height"],
                    rectRegions[i]["shape_attributes"]["x"],
                )
            )

            rectRegions[i]["shape_attributes"].pop("x")
            rectRegions[i]["shape_attributes"].pop("y")
            rectRegions[i]["shape_attributes"].pop("width")
            rectRegions[i]["shape_attributes"].pop("height")

            rectRegions[i]["shape_attributes"]["all_points_x"] = [i[1] for i in pointList]
            rectRegions[i]["shape_attributes"]["all_points_y"] = [i[0] for i in pointList]

            # for j in range(len(rectRegions[i]["shape_attributes"]["all_points_x"])):
            #     if j > 0:
            #         self.draw.line((rectRegions[i]["shape_attributes"]["all_points_x"][j-1], rectRegions[i]["shape_attributes"]["all_points_y"][j-1], rectRegions[i]["shape_attributes"]["all_points_x"][j], rectRegions[i]["shape_attributes"]["all_points_y"][j]), width = 3)

        # print(rectRegions)
        return rectRegions

    def __init__(self, angles, fillColor=(0)):
        self.angles = angles
        self.fillColor = fillColor

    def __call__(self, val, path, imgName, image):
        jsonArr = []
        y = image.size(dim=2)
        x = image.size(dim=1)

        circleRegions = []
        polyRegions = []
        ellipseRegions = []
        rectRegions = []

        # if [] in value["regions"]:
        #     for val in value["regions"]:
        #         if val != []:
        #             value["regions"] = val

        # print("\n\n\nRegioning")
        for region in val["regions"]:
            if region["shape_attributes"]["name"] == "rect":
                rectRegions.append(region)
            elif region["shape_attributes"]["name"] == "ellipse":
                ellipseRegions.append(region)
            elif region["shape_attributes"]["name"] == "circle":
                circleRegions.append(region)
            elif (
                region["shape_attributes"]["name"] == "polyline"
                or region["shape_attributes"]["name"] == "polygon"
            ):
                polyRegions.append(region)
        # print("Done Regioning\n\n\n")

        # print(rectRegions)
        # print(ellipseRegions)
        # print(circleRegions)
        # print(polyRegions)

        for i, angle in enumerate(self.angles):
            print(str(i + 2) + key)
            try:
                value = copy.deepcopy(val)

                value["filename"] = str(i + 2) + imgName
                # print("\n\n\n\nValue")
                # print(value)
                # print("Value2\n\n\n\n")

                tensorIMG = transforms.functional.perspective(
                    image,
                    [[0, 0], [0, x], [y, x], [y, 0]],
                    [
                        [abs(y * angle), x * angle],
                        [abs(y * angle), x - x * angle],
                        [y - abs(y * angle), x + x * angle],
                        [y - abs(y * angle), -x * angle],
                    ],
                    fill=(self.fillColor),
                )

                self.transformedIMG = tensorIMG

                self.tIMG = transforms.functional.to_pil_image(self.transformedIMG)

                self.tIMG.save(path + value["filename"])

                # self.draw = ImageDraw.Draw(self.tIMG)

                newShape = []

                print("Rect Transform")
                newShape = newShape + self.rectTransform(x, y, angle, copy.deepcopy(rectRegions))
                print("Ellipse Transform")
                newShape = newShape + self.ellipseTransform(
                    x, y, angle, copy.deepcopy(ellipseRegions)
                )
                print("Circle Transform")
                newShape = newShape + self.circleTransform(
                    x, y, angle, copy.deepcopy(circleRegions)
                )
                print("Poly Transform")
                newShape = newShape + self.polyTransform(x, y, angle, copy.deepcopy(polyRegions))
                value["regions"] = newShape

                jsonArr.append(value)

                # self.tIMG.show()
                # self.draw = None

            except Exception as e:
                print(e)
        # print(jsonArr)
        return jsonArr


if __name__ == "__main__":
    buildingDirectories = glob.glob("./*/")

    for path in buildingDirectories:
        pictures = glob.glob(path + "*.jpg")
        jsonFile = glob.glob(path + "*.json")
        transformer = PerspectiveTransform([-0.06, -0.03, 0.03, 0.06])
        if len(jsonFile) > 0:
            with open(jsonFile[0], "r") as read_file:
                data = copy.deepcopy(json.load(read_file))
                picJSON = data["_via_img_metadata"]
                for pic in pictures:
                    image = Image.open(pic).rotate(-90, expand=True)

                    transform = transforms.Compose([transforms.PILToTensor()])
                    img_tensor = transform(image)

                    tempPic = pic[lastChar(pic, "\\") + 1 :]
                    for key, value in picJSON.items():
                        if tempPic in key:
                            print(key)
                            for i, d in enumerate(transformer(value, path, tempPic, img_tensor)):
                                data["_via_img_metadata"][str(i + 2) + key] = d
                                if data["_via_image_id_list"].count(str(i + 2) + key) == 0:
                                    data["_via_image_id_list"].append(str(i + 2) + key)
                            break

                    image.close()

                # print(data)
                with open(jsonFile[0][:-5] + "Revised.json", "w") as newFile:
                    json.dump(data, newFile)

            read_file.close()
