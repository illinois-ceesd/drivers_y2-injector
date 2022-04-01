SetFactory("OpenCASCADE");

If(Exists(size))
    basesize=size;
Else
    //basesize=0.0002;
    basesize=0.0032;
EndIf

If(Exists(blratio))
    boundratio=blratio;
Else
    boundratio=2.0;
EndIf

If(Exists(blratiocavity))
    boundratiocavity=blratiocavity;
Else
    boundratiocavity=2.0;
EndIf

If(Exists(blratioinjector))
    boundratioinjector=blratioinjector;
Else
    boundratioinjector=2.0;
EndIf

If(Exists(injectorfac))
    injector_factor=injectorfac;
Else
    injector_factor=10.0;
EndIf

// horizontal injection
cavityAngle=45;
inj_h=4.e-3;  // height of injector (bottom) from floor
inj_t=1.59e-3; // diameter of injector
inj_d = 20e-3; // length of injector

bigsize = basesize*4;     // the biggest mesh size 
inletsize = basesize*2;   // background mesh size upstream of the nozzle
isosize = basesize;       // background mesh size in the isolator
nozzlesize = basesize/2;       // background mesh size in the isolator
cavitysize = basesize/2.; // background mesh size in the cavity region
injectorsize = inj_t/injector_factor; // background mesh size in the cavity region

Printf("basesize = %f", basesize);
Printf("inletsize = %f", inletsize);
Printf("isosize = %f", isosize);
Printf("nozzlesize = %f", nozzlesize);
Printf("cavitysize = %f", cavitysize);
Printf("injectorsize = %f", injectorsize);
Printf("boundratio = %f", boundratio);
Printf("boundratiocavity = %f", boundratiocavity);
Printf("boundratioinjector = %f", boundratioinjector);


//Cavity Start
//Point(450) = {0.65163,-0.0083245,0.0,basesize};
Point(450) = {0.67,-0.0083245,0.0,basesize};

//Bottom of cavity
//Point(451) = {0.65163,-0.0283245,0.0,basesize};
Point(451) = {0.67,-0.0283245,0.0,basesize};
Point(452) = {0.70163,-0.0283245,0.0,basesize};
// slanty cavity
//Point(453) = {0.72163,-0.0083245,0.0,basesize};
// straight cavity
Point(453) = {0.70163,-0.0083245,0.0,basesize};

// slanty cavity
//Point(500) = {0.70163+inj_h, -0.0283245+inj_h, 0., basesize};
//Point(501) = {0.70163+inj_h+inj_d, -0.0283245+inj_h, 0., basesize};
//Point(502) = {0.70163+inj_h+inj_d, -0.0283245+inj_h+inj_t, 0., basesize};
//Point(503) = {0.70163+inj_h+inj_t, -0.0283245+inj_h+inj_t, 0., basesize};
// straight cavity
Point(500) = {0.70163, -0.0283245+inj_h, 0., basesize};
Point(501) = {0.70163+inj_d, -0.0283245+inj_h, 0., basesize};
Point(502) = {0.70163+inj_d, -0.0283245+inj_h+inj_t, 0., basesize};
Point(503) = {0.70163, -0.0283245+inj_h+inj_t, 0., basesize};

//Make Cavity lines
Line(451) = {450,451};
Line(452) = {451,452};
Line(500) = {452,500};
Line(453) = {503,453};
Line(454) = {453,450};
// injector
Line(501) = {500,501};
Line(502) = {501,502};  // injector inlet
Line(503) = {502,503};

//Create lineloop of this geometry
// start on the bottom left and go around clockwise
Curve Loop(1) = {
-453, // cavity rear upper (slant)
-503, // injector top
-502, // injector inlet
-501, // injector bottom
-500, // cavity rear lower (slant)
-452, // cavity bottom
-451, // cavity front
-454 // cavity top
};

Plane Surface(1) = {1};

Physical Surface('domain') = {-1};

Physical Curve('injection') = {-502};
Physical Curve('outflow') = {454};
Physical Curve('wall') = {
451,
452,
453,
500,
503,
501
};

// Create distance field from curves, cavity only
Field[11] = Distance;
Field[11].CurvesList = {451:453, 500};
Field[11].NumPointsPerCurve = 100000;

//Create threshold field that varrries element size near boundaries
Field[12] = Threshold;
Field[12].InField = 11;
Field[12].SizeMin = cavitysize / boundratiocavity;
Field[12].SizeMax = cavitysize;
Field[12].DistMin = 0.0002;
Field[12].DistMax = 0.005;
Field[12].StopAtDistMax = 1;

// Create distance field from curves, injector only
Field[13] = Distance;
Field[13].CurvesList = {501, 503};
Field[13].NumPointsPerCurve = 10000;

//Create threshold field that varrries element size near boundaries
Field[14] = Threshold;
Field[14].InField = 13;
Field[14].SizeMin = injectorsize / boundratioinjector;
Field[14].SizeMax = injectorsize;
Field[14].DistMin = 0.000001;
Field[14].DistMax = 0.0005;
Field[14].StopAtDistMax = 1;

// background mesh size in the cavity region
cavity_start = 0.65;
cavity_end = 0.73;
Field[6] = Box;
Field[6].XMin = cavity_start;
Field[6].XMax = cavity_end;
Field[6].YMin = -1.0;
Field[6].YMax = -0.003;
Field[6].Thickness = 0.10;    // interpolate from VIn to Vout over a distance around the box
Field[6].VIn = cavitysize;
Field[6].VOut = bigsize;

// background mesh size for the injector
injector_start = 0.69;
injector_end = 0.75;
injector_bottom = -0.022;
injector_top = -0.025;
Field[7] = Box;
Field[7].XMin = injector_start;
Field[7].XMax = injector_end;
Field[7].YMin = injector_bottom;
Field[7].YMax = injector_top;
Field[7].Thickness = 0.05;    // interpolate from VIn to Vout over a distance around the box
Field[7].VIn = injectorsize;
Field[7].VOut = bigsize;


// take the minimum of all defined meshing fields
Field[100] = Min;
Field[100].FieldsList = {6, 7, 12, 14};
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

//Mesh.Smoothing = 3;
