#% visit -cli
"""
    /home/vsevolod/visit_built/bin/visit -nowin -cli -s /home/vsevolod/numrel/prj_visualisation/test4.py
"""
import os
from visit_utils import *

print("hi!")

dpath = "/media/vsevolod/Everlasting_Longning/transit/"
fname = "2121728.h5"
dpath = "/home/vsevolod/visit_data/tmp/"
fname = "iter_0002162688.vtr"

OpenDatabase(dpath + fname)
DefineScalarExpression("density","6.176269145886162e17*<rho>")

AddPlot("Volume", "density")
DrawPlots()

VolumeAtts = VolumeAttributes()

VolumeAtts.samplesPerRay = 1500 #standard 500, need to use 1500
VolumeAtts.rendererType = VolumeAtts.RayCasting  # Splatting, Texture3D, RayCasting, RayCastingIntegration

VolumeAtts.useColorVarMin = 1
VolumeAtts.colorVarMin = 1e8
VolumeAtts.useColorVarMax = 1
VolumeAtts.colorVarMax = 1e15
VolumeAtts.colorControlPoints.ClearControlPoints()
p = ColorControlPoint()
p.colors = (71, 71, 219, 255)
p.position = 0.0
VolumeAtts.colorControlPoints.AddControlPoints(p)
p = ColorControlPoint()
p.colors = (0, 0, 91, 255)
p.position = 0.2
VolumeAtts.colorControlPoints.AddControlPoints(p)
p = ColorControlPoint()
p.colors = (0, 255, 255, 255)
p.position = 0.35
VolumeAtts.colorControlPoints.AddControlPoints(p)
p = ColorControlPoint()
p.colors = (0, 127, 0, 255)
p.position = 0.5
VolumeAtts.colorControlPoints.AddControlPoints(p)
p = ColorControlPoint()
p.colors = (255, 255, 0, 255)
p.position = 0.65
VolumeAtts.colorControlPoints.AddControlPoints(p)
p = ColorControlPoint()
p.colors = (255, 96, 0, 255)
p.position = 0.8
VolumeAtts.colorControlPoints.AddControlPoints(p)
p = ColorControlPoint()
p.colors = (107, 0, 0, 255)
p.position = 1.0
VolumeAtts.colorControlPoints.AddControlPoints(p)

###VolumeAtts.opacityMode = VolumeAtts.FreeformMode
VolumeAtts.opacityMode = VolumeAtts.GaussianMode  # FreeformMode, GaussianMode, ColorTableMode
VolumeAtts.opacityControlPoints.ClearControlPoints()
p = GaussianControlPoint()
p.x = 1.0
p.height = 1.0
p.width = 0.99
p.xBias = 0
p.yBias = 0
VolumeAtts.opacityControlPoints.AddControlPoints(p)

VolumeAtts.scaling = VolumeAtts.Log

SetPlotOptions(VolumeAtts)
DrawPlots()

AddOperator("Box", 1)
BoxAtts = BoxAttributes()
BoxAtts.amount = BoxAtts.Some  # Some, All
BoxAtts.minx = -90
BoxAtts.maxx = 90
BoxAtts.miny = -90
BoxAtts.maxy = 90
BoxAtts.minz = 0.001
BoxAtts.maxz = 60
BoxAtts.inverse = 0
SetOperatorOptions(BoxAtts, 1)
DrawPlots()


AnnotationAtts = AnnotationAttributes()
AnnotationAtts.axes3D.setBBoxLocation = 1
AnnotationAtts.axes3D.bboxLocation = (0, 90, -90, 0, -90, 0)
#AnnotationAtts.axes3D.tickLocation = AnnotationAtts.axes3D.Outside  # Inside, Outside, Both
AnnotationAtts.axes3D.axesType = AnnotationAtts.axes3D.FurthestTriad # ClosestTriad, FurthestTriad, OutsideEdges, StaticTriad, StaticEdges
AnnotationAtts.axes3D.triadFlag = 0
AnnotationAtts.axes3D.bboxFlag = 0
AnnotationAtts.userInfoFlag = 0
AnnotationAtts.databaseInfoFlag = 0
AnnotationAtts.axes3D.autoSetTicks = 0
AnnotationAtts.axes3D.xAxis.title.visible = 0
AnnotationAtts.axes3D.yAxis.title.visible = 0
AnnotationAtts.axes3D.zAxis.title.visible = 0
AnnotationAtts.axes3D.xAxis.label.visible = 0
AnnotationAtts.axes3D.yAxis.label.visible = 0
AnnotationAtts.axes3D.zAxis.label.visible = 0
AnnotationAtts.axes3D.xAxis.tickMarks.visible = 1
AnnotationAtts.axes3D.xAxis.tickMarks.majorMinimum = 0
AnnotationAtts.axes3D.xAxis.tickMarks.majorMaximum = 150/1.4767
AnnotationAtts.axes3D.xAxis.tickMarks.minorSpacing = 2/1.4767
AnnotationAtts.axes3D.xAxis.tickMarks.majorSpacing = 10/1.4767
AnnotationAtts.axes3D.yAxis.tickMarks.visible = 1
AnnotationAtts.axes3D.yAxis.tickMarks.majorMinimum = -150/1.4767
AnnotationAtts.axes3D.yAxis.tickMarks.majorMaximum = 0
AnnotationAtts.axes3D.yAxis.tickMarks.minorSpacing = 2/1.4767
AnnotationAtts.axes3D.yAxis.tickMarks.majorSpacing = 10/1.4767
AnnotationAtts.axes3D.zAxis.tickMarks.visible = 1
AnnotationAtts.axes3D.zAxis.tickMarks.majorMinimum = -150/1.4767
AnnotationAtts.axes3D.zAxis.tickMarks.majorMaximum = 0
AnnotationAtts.axes3D.zAxis.tickMarks.minorSpacing = 2/1.4767
AnnotationAtts.axes3D.zAxis.tickMarks.majorSpacing = 10/1.4767
AnnotationAtts.axes3D.visible = 0
SetAnnotationAttributes(AnnotationAtts)
DrawPlots()

InvertBackgroundColor()

pL = GetPlotList()
legend = GetAnnotationObject(pL.GetPlots(0).plotName)
legend.xScale = 1.4
legend.yScale = 1.4
legend.managePosition = 0
legend.position = (0.02,0.98)
legend.drawTitle = 1
legend.drawMinMax = 0
legend.numberFormat = "%1.1e"

timeslider = CreateAnnotationObject("TimeSlider")
timeslider.position = (0.02,0.02)
timeslider.height = 0.07
timeslider.timeDisplay = timeslider.UserSpecified
m = GetMetaData(GetWindowInformation().activeSource)
tmin = 0
tmax = 5250.0/203.0129158711296
dt = 0.6

View3DAtts = View3DAttributes()
View3DAtts.viewNormal = (0.6, -0.8, -0.5)
View3DAtts.focus = (0, 0, 3)
View3DAtts.viewUp = (0, 0, -1)
View3DAtts.viewAngle = 30
View3DAtts.parallelScale = 1537.34
View3DAtts.nearPlane = -3074.68
View3DAtts.farPlane = 3074.68
View3DAtts.imagePan = (0, 0)
View3DAtts.imageZoom = 37
View3DAtts.perspective = 1
View3DAtts.eyeAngle = 2
View3DAtts.centerOfRotationSet = 0
View3DAtts.centerOfRotation = (4, 4, 512)
View3DAtts.axis3DScaleFlag = 0
View3DAtts.axis3DScales = (1, 1, 1)
View3DAtts.shear = (0, 0, 1)
View3DAtts.windowValid = 1
SetView3D(View3DAtts)
DrawPlots()

#### SAVE THE PLOT
SaveWindowAtts = SaveWindowAttributes()
SaveWindowAtts.advancedMultiWindowSave = 0
SaveWindowAtts.outputToCurrentDirectory = 0
SaveWindowAtts.outputDirectory = dpath
SaveWindowAtts.family = 0
SaveWindowAtts.format = SaveWindowAtts.PNG  # BMP, CURVE, JPEG, OBJ, PNG, POSTSCRIPT, POVRAY, PPM, RGB, STL, TIFF, ULTRA, VTK, PLY
SaveWindowAtts.resConstraint = SaveWindowAtts.NoConstraint
SaveWindowAtts.width = 1024
SaveWindowAtts.height = 768
SaveWindowAtts.screenCapture = 0

nts = TimeSliderGetNStates()



SaveWindowAtts.fileName = fname
SetSaveWindowAttributes(SaveWindowAtts)
SaveWindow()


DeleteAllPlots()

CloseDatabase(dpath + fname)

print("bye!")
exit(0)