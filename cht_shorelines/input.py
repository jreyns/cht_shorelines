from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np

from .io import PathLike, normalize_optional_file_name, write_runfile


class Variables:
    def __init__(self) -> None:
        self.hso = 1
        self.tper = 6
        self.phiw0 = 330
        self.spread = 90
        self.dirspr = 12
        self.wvcfile = ""
        self.waveclimfile = ""
        self.hsbackground = 0
        self.ddeep = 25
        self.dnearshore = 8
        self.randomseed = -1
        self.randomseedsettings = ""
        self.interpolationmethod = "weighted_distance"  #'alongshore_mapping'
        self.mergeconditions = 0
        ## ---------------------simulation tide parameters ------------------------
        self.yestide = 0
        self.tidefile = ""
        self.tideprofile = ""
        self.tidedx = 10
        self.tiden: list[float] | str = []
        self.htide = 0
        self.vtide = 0
        self.refdep = 5
        ## ------------------- simulation coastline definition --------------------
        self.ldbcoastline = ""
        self.xmc = ""
        self.ymc = ""
        self.gisconvention = 0
        self.ds0 = 100
        self.griddingmethod = 2
        self.d = 10
        self.phif: list[float] | str = []
        self.maxangle = 60
        self.preserveorientation = 0
        ## ------------------- simulation transport parameters --------------------
        self.trform = "CERC"
        self.b = 1e6
        self.qscal = 1
        self.qscalgr = 0.3
        self.d50 = 2.0e-4
        self.d90 = 3.0e-4
        self.dgr: list[float] | str = []
        self.sedinput: list[float] | str = []
        self.multi = 0
        self.bedthick = 0.2
        self.bedwidth = 50
        self.bedgravelperc = 50
        self.printbed = 0
        self.porosity = 0.4
        self.tanbeta = 0.03
        self.tanbetagr = 0.25
        self.tanbetasetup = 1
        self.rhos = 2650
        self.rhow = 1025
        self.g = 9.81
        self.alpha = 1.8
        self.gamma = 0.72
        self.pswell = 20
        self.ks = 0.05
        self.hclosure = 8
        self.cf = 0.0023
        self.n = 0.02
        self.acal = 0.2
        self.hmin = 0.1
        self.sphimax: list[float] | str = []
        self.relaxationlength: list[float] | str = []
        self.suppresshighangle = 0
        ## ------------------- simulation time steps & numerical-------------------
        self.tc = 1
        self.dt = 0
        self.dtdune: list[float] | str = []
        self.reftime = "2020-01-01"
        self.endofsimulation = "2040-01-01"
        self.twopoints = 1
        self.smoothfac = 0
        self.smoothrefrac = 0
        ## -------------------------- boundary condition --------------------------
        self.boundaryconditionstart = "Fixed"
        self.boundaryconditionend = "Fixed"
        self.cyclic = 0
        ## ------------------------ climate change impact -------------------------
        self.ccslr: list[float] | str = []
        self.cchs: list[float] | str = []
        self.ccdir: list[float] | str = []
        ## ----------------------------- structures -------------------------------
        self.struct = 1
        self.ldbstructures = ""
        self.xhard: list[float] | str = []
        self.yhard: list[float] | str = []
        self.structtype: list[str] = []
        self.aw = 1.27
        self.awfixedhs = 5
        self.bypasscontrfac = 1
        self.bypassdistpwr = 1
        self.submerged = 0
        self.groinelev: list[float] = []
        ## ------------- wave transmission over submerged breakwater --------------
        self.transmission = 0
        self.transmform = "angr"
        self.transmdir = 1
        self.transmfile = ""
        self.transmbwdepth: list[float] = []
        self.transmcrestheight: list[float] = []
        self.transmslope: list[float] = []
        self.transmcrestwidth: list[float] = []
        self.transmd50 = 1
        ## ------------------------- permeable structures -------------------------
        self.perm = 0
        self.ldbpermeable = ""
        self.xperm: list[float] = []
        self.yperm: list[float] = []
        self.wavetransm: list[float] = [1]
        self.qstransm: list[float] = [1]
        ## ----------------------------- revetments -------------------------------
        self.revet = 1
        self.ldbrevetments = ""
        self.xrevet: list[float] = []
        self.yrevet: list[float] = []
        self.iterrev = 5
        self.critwidth = 5
        self.fillbeachatrevetment = 0
        ## --------------------------- wave diffraction ---------------------------
        self.diffraction = 0
        self.wdform = "Roelvink"
        self.kdform = "Kamphuis"
        self.rotfac = 1
        self.omegat = 1
        self.diffdist: list[float] = []
        self.diffsmooth = 0
        self.maxanglerotation = 90.1
        self.diffractiondist: list[float] = []
        ## ---------------------------- nourishments ------------------------------
        self.nourish = 0
        self.growth = 1
        self.norfile = ""
        self.nourmethod = "default"
        self.ldbnourish = ""
        self.nourratefile = ""
        self.nourstartfile = ""
        self.nourendfile = ""
        self.nourrate = 100
        ## ------------------------ shoreface nourishments ------------------------
        self.fnourish = 0
        self.fnorfile = ""
        self.sal = 35
        self.temp = 10
        self.mb = -0.5
        self.labda0 = 0.56e-6
        self.k: list[float] = []
        ## -------------------------- sources and Sinks ---------------------------
        self.sourcessinks = ""
        self.ssfile = ""
        ## -------------------- aeolian transport to the dunes --------------------
        self.dune = 0
        self.ldbdune = ""
        self.kf = 0.02
        self.cs = 5e-4
        self.cstill = 5e-6
        self.xtill: list[list[float]] = []
        self.perctill = 80
        self.d50r = 2.5e-4
        self.rhoa = 1.225
        self.duneaw = 0.1
        self.kw = 4.2
        self.k = 0.41
        self.segmaw = 0.1
        self.maxslope = 1 / 15
        self.aoverwash = 3
        self.xdune = 0
        self.ydune = 0
        self.wberm = 50
        self.dfelev = 3
        self.dcelev = 8
        self.csmodel = ""
        ## --------------------------- wind conditions ----------------------------
        self.wndfile = ""
        self.Windclimfile = ""
        self.cd = 0.002
        self.uz = 8
        self.z = 10
        self.phiwnd0 = 330
        ## --------------------- still water levels and run-up---------------------
        self.runupform = "Stockdon"
        self.runupfactor = 1
        self.watfile = ""
        self.watclimfile = ""
        self.watlocfile = ""
        self.wvdfile = ""
        self.WaveLocfile = ""
        self.swl0 = 0
        ##------------------------- Sediment limitations --------------------------
        self.sedlim = False
        self.ldbsedlim = ""
        self.xsedlim: list[float] = []
        self.ysedlim: list[float] = []
        self.widthsedlim: list[float] = []
        self.psedlim = 0
        ## --------------------------- mud properties -----------------------------
        self.mud = False
        self.mudtaucr = 0.3
        self.mudm = 1.0e-4
        self.mudb = 1000
        self.mudmhw = 1
        self.mudmsl = 0
        self.mudtfm = 10
        self.mudw = 1e-4
        self.mudbfcrit = 500
        self.mudbmmin = 1
        self.mudbmmax = 100000
        self.ldbriverdisch = ""
        self.ldbmangrove = ""
        ## ----------------------- physics of spit width --------------------------
        self.spitmethod = "default"
        self.spitwidth = 50
        self.spitheadwidth = 200
        self.owscale = 0.1
        self.owtimescale = 0.0
        self.spitdsf = self.d * 0.8
        self.spitdbb = 0.5 * self.spitdsf
        self.bheight = 2
        self.tideinteraction = False
        self.waveinteraction = False
        self.wavefile = ""
        self.surfwidth = 1000
        self.bathyupdate = ""
        ## ------------------------------- channel --------------------------------
        self.channel = False
        self.channelwidth = 550
        self.channelfac = 0.08
        self.channeldischrate = 0
        self.channeldischr = 300
        self.ldbchannel: str = ""
        self.rero = 300
        self.rdepo = 600
        self.tscale = 1
        self.xrmc = ""
        self.yrmc = ""
        ## ----------------------------- flood delta ------------------------------
        self.flooddelta = False
        self.ldbflood: str = ""
        self.xfloodpol: list[float] = []
        self.yfloodpol: list[float] = []
        self.ldbspit = ""
        self.xspitpol: list[float] = []
        self.yspitpol: list[float] = []
        self.dxf = 50
        self.overdepth = 2
        ## ------------------------ formatting / plotting -------------------------
        self.plotvisible = True
        self.xlimits: list[float] = []
        self.ylimits: list[float] = []
        self.xywave: list[float] = []
        self.xyoffset = [0, 0]
        self.pauselength: list[float] = []
        self.ldbplot: list[str] = []
        self.ploths = 0
        self.plotdir = 0
        self.plotqs = 0
        self.plotupw = 0
        self.plotsed = 0
        self.llocation = "SouthWest"
        self.ld = 3000
        self.usefill = 1
        self.usefillpoints = 0
        self.fignryear = 12
        self.plotdate = ""
        self.plotinterval = 1
        self.figplotfreq = 0
        self.fastplot = 1
        ## -------------------------------- output --------------------------------
        self.outputdir = r"Output"
        self.outputfile = "shorelines_output"
        self.rundir = r"Delft3D/def_model/"
        self.xyout = []
        self.xyprofiles = []
        self.storageinterval = 50
        self.storagedate: list[str] = []
        self.netcdf = 0
        self.separatepgrids = 1
        ## -------------------------- extract shorelines --------------------------
        self.slplot = {}
        self.extractxy = 0
        self.printfig = 0
        ## ---------------- extract shoreline & dune foot locations ---------------
        self.yesplot = 0
        self.bermwplot = 0
        self.bermwplotint = []
        self.qplot = 0
        self.transect = ""
        self.clplot = 0
        self.clplotint = []
        self.extractbermplot = 0
        ## video
        self.video = 0
        ## debug
        self.debug = 0
        ## --------------------------- data Assimilation---------------------------
        self.da = 0
        self.bs = 0


class ShorelinesInput:
    def __init__(
        self,
        root: PathLike | None = None,
        runfile: PathLike | None = None,
    ) -> None:
        self.variables = Variables()
        self.root = Path(root) if root is not None else Path.cwd()
        self.runfile = normalize_optional_file_name(runfile, argument="runfile")

    @property
    def runfile_name(self) -> str:
        """
        Return the runfile name used for this case.

        Returns
        -------
        str
            Explicit runfile name, or ``<root-name>.txt`` when none is set.
        """
        if self.runfile:
            return str(self.runfile)
        return f"{self.root.name}.txt"

    @property
    def runfile_path(self) -> Path:
        """
        Return the absolute path to the runfile.

        Returns
        -------
        pathlib.Path
            Full path to the active runfile.
        """
        return self.root / self.runfile_name

    def to_dict(self) -> dict[str, object]:
        """
        Convert runfile variables to a plain dictionary.

        Returns
        -------
        dict of str to object
            Mapping of runfile keys to current values.
        """
        return dict(vars(self.variables))

    def write(self, file_name: PathLike | None = None) -> None:
        """
        Write the runfile to disk.

        Parameters
        ----------
        file_name : str or pathlib.Path, optional
            Alternative runfile name to use for this write operation.
        """
        if file_name is not None:
            self.runfile = normalize_optional_file_name(file_name, argument="file_name")
        write_runfile(self.runfile_path, self.to_dict())

    def read(self, file_name: PathLike | None = None) -> None:
        """
        Read runfile values from disk into ``variables``.

        Parameters
        ----------
        file_name : str or pathlib.Path, optional
            Alternative runfile name to read.
        """
        if file_name is not None:
            self.runfile = normalize_optional_file_name(file_name, argument="file_name")
        path = self.runfile_path
        if not path.exists():
            raise FileNotFoundError(path)

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("%") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split("%", 1)[0].strip().rstrip(";")
            setattr(self.variables, key, _parse_matlab_value(value))


def _parse_matlab_value(value: str) -> Any:
    if value in {"[]", "{}"}:
        return []
    if value.lower() in {"nan", "+nan"}:
        return np.nan
    if value.lower() == "inf":
        return np.inf
    if value.lower() == "-inf":
        return -np.inf
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if value.startswith("[") and value.endswith("]"):
        return _parse_matrix(value[1:-1])
    if value.startswith("{") and value.endswith("}"):
        return _parse_cell(value[1:-1])
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return float(value)
        except ValueError:
            return value


def _parse_matrix(value: str) -> list[object] | list[list[object]]:
    if not value.strip():
        return []
    rows = []
    for row in value.split(";"):
        rows.append([_parse_matlab_value(v) for v in row.split()])
    if len(rows) == 1:
        return rows[0]
    return rows


def _parse_cell(value: str) -> list[list[object]]:
    if not value.strip():
        return []
    rows = []
    for row in value.split(";"):
        rows.append([_parse_matlab_value(v) for v in row.split()])
    return rows
