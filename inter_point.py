import os
import numpy as np
import shapely.geometry as shgeo
import math
def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return list(map(int,combinate[sorted[0]]))

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

        

def GetPoly4FromPoly5(poly):
    distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
    distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
    pos = np.array(distances).argsort()[0]
    count = 0
    outpoly = []
    while count < 5:
            #print('count:', count)
        if (count == pos):
            outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
            outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
            count = count + 1
        elif (count == (pos + 1)%5):
            count = count + 1
            continue
        else:
            outpoly.append(poly[count * 2])
            outpoly.append(poly[count * 2 + 1])
            count = count + 1
    return outpoly

def inter_poly(box1):
    poly1 = shgeo.Polygon([(box1[0], box1[1]), (box1[2], box1[3]), (box1[4], box1[5]),
                                     (box1[6], box1[7])])
    poly2 = shgeo.Polygon([(0,0), (1023, 0), (1023, 1023),
                                     (0,1023)])
    inter_poly = poly1.intersection(poly2)
    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
    out_poly = list(inter_poly.exterior.coords)[0: -1] #输出相交点
    if len(out_poly) < 4:
        print('Error')
        return None

    out_poly2 = []
    for i in range(len(out_poly)):
        out_poly2.append(out_poly[i][0])
        out_poly2.append(out_poly[i][1])

    if (len(out_poly) == 5):
        out_poly2 = GetPoly4FromPoly5(out_poly2) #5个点转4个点

    out_poly2 = choose_best_pointorder_fit_another(out_poly2, box1) #选择距离原来坐标最近的起始点
    return out_poly2
#print(" ".join(map(str,inter_poly([648,957,572,944,502,1319,577,1334]))   ))
#print(" ".join(map(str,inter_poly([164,189,106,190,100,-54,157,-54]))   ))

#print(" ".join(map(str,inter_poly([656,2,601,109,998,14,977,-74]))   ))


