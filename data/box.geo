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

If(Exists(injectorfac))
    injector_factor=injectorfac;
Else
    injector_factor=10.0;
EndIf

// horizontal injection
inj_h=4.e-3;  // height of injector (bottom) from floor
inj_t=1.59e-3; // diameter of injector
inj_d = 20e-3; // length of injector

bigsize = basesize*4;     // the biggest mesh size 
injectorsize = inj_t/injector_factor; // background mesh size in the cavity region

Printf("basesize = %f", basesize);
Printf("injectorsize = %f", injectorsize);
Printf("boundratio = %f", boundratio);


//Point(500) = {0.70163, -0.0283245+inj_h, 0., basesize};
//Point(501) = {0.70163+inj_d, -0.0283245+inj_h, 0., basesize};
//Point(502) = {0.70163+inj_d, -0.0283245+inj_h+inj_t/2, 0., basesize};
//Point(503) = {0.70163, -0.0283245+inj_h+inj_t/2, 0., basesize};

Point(500) = {0.70, -0.0283245+inj_h+inj_t/2, 0., basesize};
Point(501) = {0.715, -0.0283245+inj_h+inj_t/2, 0., basesize};
Point(502) = {0.715, -0.0283245+inj_h+inj_t, 0., basesize};
Point(503) = {0.70, -0.0283245+inj_h+inj_t, 0., basesize};

// injector
Line(501) = {500,501};  // injector midline
Line(502) = {501,502};  // injector inlet
Line(503) = {502,503};  // injector wall
Line(504) = {503,500}; // injector outlet

//Create lineloop of this geometry
// start on the bottom left and go around clockwise
Curve Loop(1) = {
-503,
-502,
-501,
-504
};

Plane Surface(1) = {1};

Physical Surface('domain') = {-1};

Physical Curve('injection') = {-502};
Physical Curve('outflow') = {-504};
Physical Curve('wall_injector') = {-503};
Physical Curve('injector_symmetry') = {501};

// Create distance field from curves, injector only
Field[13] = Distance;
Field[13].CurvesList = {503};
Field[13].NumPointsPerCurve = 10000;

//Create threshold field that varrries element size near boundaries
Field[14] = Threshold;
Field[14].InField = 13;
Field[14].SizeMin = injectorsize / boundratio;
Field[14].SizeMax = injectorsize;
Field[14].DistMin = 0.000001;
Field[14].DistMax = 0.0005;
Field[14].StopAtDistMax = 1;

// background mesh size for the injector
injector_start = 0.70;
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
Field[100].FieldsList = {7, 14};
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

//Mesh.Smoothing = 3;
