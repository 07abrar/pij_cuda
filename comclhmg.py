import numpy as np

class ComclhmgData:
    def __init__(self):
        # Common block /s1/
        self.lnmax = None
        self.imax = None
        self.mxchi = None
        self.mxr1d = None
        self.mtxr23 = None
        self.mxdnse = None
        self.mxdnsi = None
        self.mxdns2 = None
        self.mxdwne = None
        self.mxdwni = None
        self.mxdwn2 = None
        self.mxreac = None
        self.mxsig0 = None
        self.mxtemp = None
        self.mxr = None
        self.iswh = None

        # Common block /s2/
        self.ncodel = np.zeros(80)
        self.enbnd = np.zeros(75)
        self.chi = np.zeros((75, 20))
        self.ncodx = np.zeros(20)
        self.aw = np.zeros(80)
        self.ld = np.zeros((3, 80))
        self.la = np.zeros((3, 80))
        self.msf = np.zeros((8, 80))
        self.tab = np.zeros((10, 80))

        # Common block /s3/
        self.ft = np.zeros((10, 80))
        self.rparam = np.zeros((8, 7, 80))
        self.ntemp = np.zeros((8, 80))
        self.nr = np.zeros((8, 80))

        # Common block /s4/
        self.sig1d = np.zeros((80, 8, 72))
        self.stre = np.zeros((72, 40, 72))
        self.stri = np.zeros((72, 40, 72))
        self.str2 = np.zeros((72, 40, 72))
        self.ftab = np.zeros((8, 4, 72, 6, 40))

        # Common block /s5/
        self.idnuc = np.zeros(80)
        self.ncstat = np.zeros(80)
        self.tmp1 = np.zeros((8, 4, 4))
        self.tmp2 = np.zeros((72, 80))
        self.mreduc = 0
        self.idnuc1 = np.zeros(80, dtype=int)

        # Common block /c1/
        self.g = np.zeros(10)
        self.x = np.zeros(10)
        self.b = np.zeros(10)
        self.c = np.zeros(10)
        self.d = np.zeros(10)
        self.maxdat = 0
        self.a = np.zeros((10, 3))
        self.s = np.zeros(10)

        # Common block /c2/
        self.gg = np.zeros((10, 5))
        self.bg = np.zeros((10, 5))
        self.cg = np.zeros((10, 5))
        self.dg = np.zeros((10, 5))
        self.yt = np.zeros(5)
        self.tp = np.zeros(5)
        self.val = 0

        # Common block /sp/
        self.vt = np.zeros(4)
        self.tt = np.zeros(4)
        self.as_ = 0
        self.bs = 0
        self.cs = 0
        self.ds = 0
        self.ac = np.zeros(4)
        # additional common block for splan3
        self.yy = np.zeros(4)
        self.xx = np.zeros(4)

        # Common block /cl1/
        self.mclnuc = 0
        self.atden = np.zeros(100)
        self.ncod = np.zeros(100)
        self.idcl = np.zeros(80, dtype=int)
        self.idcl1 = np.zeros(100, dtype=int)
        self.atdreg = np.zeros((100, 40))

        # Common block /cl2/
        self.sigt = np.zeros(100)
        self.sig0 = np.zeros(100)
        self.sig0g = np.zeros((100, 70))
        self.sigtg = np.zeros((100, 70))

        # Common block /cl3/
        self.rd = np.zeros(20)
        self.adnc = np.zeros((20, 100))
        self.mnccel = np.zeros(20)
        self.volrg = np.zeros(20)
        self.mxrg = 0
        self.nfubnd = 0
        self.fbell = 0
        self.anfu = np.zeros(50)
        self.mxncfu = 0
        self.idnc = np.zeros((20, 100), dtype=int)
        self.idnc1 = np.zeros((20, 100), dtype=int)
        self.volrgt = 0

        # Common block /c14/
        self.siga = np.zeros((100, 70))
        self.sigc = np.zeros((100, 70))
        self.sigel = np.zeros((100, 70))
        self.sigin = np.zeros((100, 70))
        self.sig2 = np.zeros((100, 70))
        self.sigf = np.zeros((100, 70))
        self.siger = np.zeros((100, 70))
        self.xnu = np.zeros((100, 70))
        self.xmyu = np.zeros((100, 70))
        self.vsgf = np.zeros((100, 70))
        self.sigtrn = np.zeros((100, 70))

        # Common block /c15/
        self.sigtrg = np.zeros((100, 70))
        self.rrg = np.zeros(100)
        self.volrrg = np.zeros(100)
        self.idmat = np.zeros(100, dtype=int)
        self.mcpreg = np.zeros(100)
        self.maxcrg = 0

        # Common block /c16/
        self.pij = np.zeros((100, 100, 70))
        self.pesc = np.zeros((100, 70))
        self.pescq = np.zeros((100, 70))
        self.q00 = np.zeros(70)

        # Common block /cf1/
        self.absg = np.zeros((100, 100))
        self.weigg = np.zeros((100, 100))

        # Common block /c17/
        self.qs = np.zeros((100, 70))
        self.fl = np.zeros((100, 70))
        self.flo = np.zeros((100, 70))

        # Common block /c18/
        self.scm = np.zeros((100, 70, 70))
        self.vsigf = np.zeros((100, 70))
        self.xai = np.zeros((100, 70))
        self.sigabs = np.zeros((100, 70))

        # Common block /c19/
        self.xkef0 = 0
        self.errfl = 0
        self.errkef = 0
        self.nord = 0
        self.temper = 0
        self.nchi = 0

        # Common block /c20/
        self.sig0b = np.zeros((10, 50, 70))
        self.sigtmi = np.zeros((10, 50, 70))

        # Common block /tpp/
        self.tp1 = np.zeros((100, 70))
        self.qss = np.zeros((100, 70))
        self.qsf = np.zeros((100, 70))

        # Common block /c29/
        self.sgrmv = np.zeros((50, 70))
