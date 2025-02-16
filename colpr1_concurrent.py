from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import math
import time
from comclhmg import ComclhmgData

def read1(filename, data: ComclhmgData):
    # with open(filename, 'r') as file:
    #     content = file.read()
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Create a new content string without comment lines
    content = ""
    for line in lines:
        if not line.strip().startswith('c'):
            content += line

    values = list(map(float, content.split()))

    header_values = values[:7]
    data.mreduc = int(header_values[0])
    data.temper = float(header_values[1])
    data.mclnuc = int(header_values[2])
    data.mxrg = int(header_values[3])
    data.nfubnd = int(header_values[4])
    data.fbell = int(header_values[5])
    data.nord = int(header_values[6])
    print(f'mreduc, temper, mclnuc, mxrg, nfubnd, fbell, nord')
    print(f'{data.mreduc},{data.temper}, {data.mclnuc}, {data.mxrg}, {data.nfubnd}, {data.fbell}, {data.nord}')

    data.idnuc = np.zeros(data.mreduc, dtype=int)
    data.rd = np.zeros(data.mxrg, dtype=float)

    index = 7
    idnuc_values = values[index:index+data.mreduc]
    index += data.mreduc
    rd_values = values[index:index+data.mxrg]
    index += data.mxrg

    for i in range(data.mreduc): data.idnuc[i] = idnuc_values[i]
    for i in range(data.mxrg): data.rd[i] = rd_values[i]

    header2_values = values[index:index+4]
    index += 4
    data.xkef0 = float(header2_values[0])
    data.errfl = float(header2_values[1])
    data.errkef = float(header2_values[2])
    data.nchi = int(header2_values[3])
    print(f'xkef0, errfl, errkef, nchi')
    print(f'{data.xkef0},{data.errfl}, {data.errkef}, {data.nchi}')

    data.mnccel = np.zeros(data.mxrg, dtype=int)

    mnccel_values = values[index:index+data.mxrg]
    index += data.mxrg

    for i in range(data.mxrg): data.mnccel[i] = mnccel_values[i]

    itp = 0
    maxtp = 0
    maxtp1 = 0
    volrgt = 0.0
    print(f'(idnc(i,j),adnc(i,j),j=1,mnccel(i))')
    data.idnc = np.zeros((data.mxrg, np.max(data.mnccel)), dtype=int)
    data.adnc = np.zeros((data.mxrg, np.max(data.mnccel)), dtype=float)
    data.idnc1 = np.zeros((data.mxrg, np.max(data.mnccel)), dtype=int)
    
    for i in range(data.mxrg):
        for j in range(data.mnccel[i]):
            idnc_values = values[index]
            index += 1
            adnc_values = values[index]
            index += 1
            data.idnc[i, j] = int(idnc_values)
            data.adnc[i, j] = float(adnc_values)
        
        if i == 0:
            data.volrg[0] = 3.14159 * data.rd[0] ** 2
        else:
            data.volrg[i] = 3.14159 * (data.rd[i] ** 2 - data.rd[i - 1] ** 2)
        
        data.volrgt += data.volrg[i]

        for j in range(data.mnccel[i]):
            itp += 1

            found = False
            for i1 in range(maxtp + 1):
                if data.ncod[i1] == data.idnc[i, j]:
                    data.idnc1[i, j] = i1
                    found = True
                    continue

                if not found:
                    data.ncod[maxtp1] = data.idnc[i, j]
                    data.idnc1[i, j] = maxtp1
                    maxtp1 = maxtp + 1

                # print(f'idnc1: {i}, {j}, {i1}, {itp}, {maxtp}')
            maxtp = maxtp1
        
        # print(f'idnc1 = {i}', [f"{data.idnc1[i, j1]}" for j1 in range(data.mnccel[i])])

    data.mcpreg = np.zeros(data.mxrg, dtype=int)
    mcpreg_values = values[index:index+data.mxrg]
    index += data.mxrg
    for i in range (data.mxrg): data.mcpreg[i] = mcpreg_values[i]
    print('(mcpreg(i), i=0, mxrg)')
    print([f"{data.mcpreg[i]}" for i in range(data.mxrg)])

    data.mclnuc = maxtp
    data.mxncfu = maxtp

    print(f'volrgt={data.volrgt}')
    print([f"{data.volrg[i]}" for i in range(data.mxrg)])
    print('ncod(i):')
    print([f"{data.ncod[i]}" for i in range(maxtp)])

    for i in range(data.mclnuc):
        data.atden[i] = 0.0
        data.anfu[i] = 0.0

    for i in range(data.mxrg):
        for j in range(data.mnccel[i]):
            id = data.idnc1[i, j]
            data.atden[id] += data.adnc[i, j] * data.volrg[i] / data.volrgt
            if i > data.nfubnd - 1:
                continue
            data.anfu[id] += data.adnc[i, j] * data.volrg[i] / data.volrgt
    
    print(f"atden(i): {data.mclnuc}")
    print([f"{data.atden[i]:.2f}" for i in range(data.mclnuc)])

    # for i in range(0, 80): #Kenapa 80? kenapa bukan data.lnmax?
    data.ncstat = np.zeros(391, dtype=int)
    data.idnuc1 = np.zeros(391, dtype=int)
    for i in range(0, 391): #391 adalah lnmax dari libcel.dat
        data.ncstat[i] = 0
        data.idnuc1[i] = 0
        for j in range(data.mreduc):
            if i+1 == data.idnuc[j]:
                data.ncstat[i] = 1
                data.idnuc1[i] = j
        # print(f"{i}, {data.ncstat[i]}, {data.idnuc1[i]}")

def suppl1(data: ComclhmgData, ng, mode):
    name1, name2 = "", ""  # Character variables in Fortran

    for i in range(data.lnmax):
        if data.ncstat[i] != 1:
            continue

        if mode == 1:
            max1 = data.mxdnse
        elif mode == 2:
            max1 = data.mxdnsi
        elif mode == 3:
            max1 = data.mxdns2

        itp = data.idnuc1[i]

        for j in range(max1):
            igj = ng + j #check, looks good
            if igj > data.imax:
                continue

            if mode == 1:
                data.stre[ng, itp, igj] = data.tmp2[j, i]
            elif mode == 2:
                data.stri[ng, itp, igj] = data.tmp2[j, i]
            elif mode == 3:
                data.str2[ng, itp, igj] = data.tmp2[j, i]

def supl2(data: ComclhmgData, ng, kt, mr, m, mt):
    name1, name2 = "", ""  # Character variables in Fortran

    itp = data.idnuc1[m]
    if itp == -1: #check again
        return

    for j in range(data.mxsig0):
        for k in range(kt):
            data.ftab[j, k, ng, mt, itp] = data.tmp1[j, k, 0]

def rdlbsp(filename, data: ComclhmgData):
    with open(filename, 'r') as file:
        content = file.read()
    
    values = list(map(float, content.split()))

    print("Reading header")

    header_values = list(map(int, values[:16]))
    data.lnmax, data.imax, data.mxchi, data.mxr1d, data.mtxr23, data.mxdnse, \
    data.mxdnsi, data.mxdns2, data.mxdwne, data.mxdwni, data.mxdwn2, data.mxreac, \
    data.mxsig0, data.mxtemp, data.mxr, data.iswh = header_values

    print(f"lnmax, imax, mxchi, mxr1d, mtxr23, mxdnse, mxdnsi, mxdns2, "
        f"mxdwne, mxdwni, mxdwne, mxreac, mxsig0, mxtemp, mxr, iswh")
    print(f"{data.lnmax}, {data.imax}, {data.mxchi}, {data.mxr1d}, {data.mtxr23}, {data.mxdnse}, "
        f"{data.mxdnsi}, {data.mxdns2}, {data.mxdwne}, {data.mxdwni}, {data.mxdwne}, {data.mxreac}, "
        f"{data.mxsig0}, {data.mxtemp}, {data.mxr}, {data.iswh}")
    
    print(f'reading ncode ')
    data.ncodel = np.zeros(data.lnmax, dtype=int)
    data.enbnd = np.zeros(data.imax + 1, dtype=float)
    data.chi = np.zeros((data.mxchi, data.imax), dtype=float)
    data.ncodx = np.zeros(data.mxchi, dtype=int)
    data.aw = np.zeros(data.lnmax, dtype=float)
    data.ld = np.zeros((data.lnmax, 3), dtype=int)
    data.la = np.zeros((data.lnmax, 3), dtype=int)
    data.msf = np.zeros((data.lnmax, data.mxreac), dtype=int)
    data.tab = np.zeros((data.lnmax, data.mxsig0), dtype=float)
    data.ft = np.zeros((data.lnmax, data.mxtemp), dtype=float)
    data.rparam = np.zeros((data.lnmax, data.mxr, data.mxsig0), dtype=float)
    data.ntemp = np.zeros((data.lnmax, data.mxreac), dtype=int)
    data.nr = np.zeros((data.lnmax, data.mxreac), dtype=int)

    index = 16
    ncodel_values = values[index:index+data.lnmax]
    index += data.lnmax
    enbnd_values = values[index:index+data.imax + 1]
    index += data.imax + 1
    chi_values = values[index:index + data.imax * data.mxchi]
    index += data.imax * data.mxchi
    ncodx_values = values[index:index + data.mxchi]
    index += data.mxchi
    aw_values = values[index:index + data.lnmax]
    index += data.lnmax
    ld_values = values[index:index + 3 * data.lnmax]
    index += 3 * data.lnmax
    la_values = values[index:index + 3 * data.lnmax]
    index += 3 * data.lnmax
    msf_values = values[index:index + data.mxreac * data.lnmax]
    index += data.mxreac * data.lnmax
    tab_values = values[index:index + data.mxsig0 * data.lnmax]
    index += data.mxsig0 * data.lnmax
    ft_values = values[index:index + data.mxtemp * data.lnmax]
    index += data.mxtemp * data.lnmax
    rparam_values = values[index:index + data.mxsig0 * data.mxr * data.lnmax]
    index += data.mxsig0 * data.mxr * data.lnmax
    ntemp_values = values[index:index + data.mxreac * data.lnmax]
    index += data.mxreac * data.lnmax
    nr_values = values[index:index + data.mxreac * data.lnmax]
    index += data.mxreac * data.lnmax

    for i in range(data.lnmax): data.ncodel[i] = ncodel_values[i]
    for i in range(data.imax + 1): data.enbnd[i] = enbnd_values[i]
    for i in range(data.mxchi):
        for j in range(data.imax):
            data.chi[i, j] = chi_values[i * data.imax + j]
    data.chi = data.chi.T
    for i in range(data.mxchi): data.ncodx[i] = ncodx_values[i]
    for i in range(data.lnmax): data.aw[i] = aw_values[i]
    for i in range(data.lnmax):
        for j in range(3):
            data.ld[i, j] = ld_values[i * 3 + j]
    data.ld = data.ld.T
    for i in range(data.lnmax):
        for j in range(3):
            data.la[i, j] = la_values[i * 3 + j]
    data.la = data.la.T
    for i in range(data.lnmax):
        for j in range(data.mxreac):
            data.msf[i, j] = msf_values[i * data.mxreac + j]
    data.msf = data.msf.T
    for i in range(data.lnmax):
        for j in range(data.mxsig0):
            data.tab[i, j] = tab_values[i * data.mxsig0 + j]
    data.tab = data.tab.T
    for i in range(data.lnmax):
        for j in range(data.mxtemp):
            data.ft[i, j] = ft_values[i * data.mxtemp + j]
    data.ft = data.ft.T
    for i in range(data.lnmax):
        for j in range(data.mxr):
            for k in range(data.mxsig0):
                data.rparam[i, j, k] = rparam_values[i * data.mxr * data.mxsig0 + j * data.mxsig0 + k]
    data.rparam = np.transpose(data.rparam, (2, 1, 0))
    for i in range(data.lnmax):
        for j in range(data.mxreac):
            data.ntemp[i, j] = ntemp_values[i * data.mxreac + j]
    data.ntemp = data.ntemp.T
    for i in range(data.lnmax):
        for j in range(data.mxreac):
            data.nr[i, j] = nr_values[i * data.mxreac + j]
    data.nr = data.nr.T
    
    data.sig1d = np.zeros((data.lnmax, data.mxr1d, data.imax), dtype=float)
    sig1d_array = []
    data.stre = np.zeros((data.imax, 80, data.imax+data.mxdnse), dtype=float)
    data.stri = np.zeros((data.imax, 80, data.imax+data.mxdnsi), dtype=float)
    data.str2 = np.zeros((data.imax, 80, data.imax+data.mxdns2), dtype=float)
    data.ftab = np.zeros((data.mxsig0, np.max(data.msf), data.imax, data.mxreac, data.mreduc))
    
    for i in range(data.imax):
        sig1d_values = values[index:index + data.lnmax * data.mxr1d]
        index += data.lnmax * data.mxr1d
        sig1d_temp = np.zeros((data.mxr1d, data.lnmax), dtype=float)
        for i1 in range(data.mxr1d):
            for j1 in range(data.lnmax):
                sig1d_temp[i1, j1] = sig1d_values[i1 * data.lnmax + j1]
        sig1d_temp = sig1d_temp.T
        sig1d_array.append(sig1d_temp)

        if i <= data.mxdwne:
            tmp2_values = values[index:index + data.mxdnse * data.lnmax]
            index += data.mxdnse * data.lnmax
            data.tmp2 = np.zeros((data.lnmax, data.mxdnse), dtype=float)
            for j in range(data.lnmax):
                for m in range(data.mxdnse):
                    data.tmp2[j, m] = tmp2_values[j * data.mxdnse + m]
            data.tmp2 = data.tmp2.T
        
        mode = 1
        suppl1(data, i, mode)

        if data.iswh != 0:
            itp = data.idnuc1[data.iswh - 1]
            if itp != 0: #check again
                stre_values = values[index:index + data.imax]
                index += data.imax
                for j in range(data.imax): 
                    data.stre[j, itp, i] = stre_values[j]

            else:
                tmp2_values = values[index:index + int(data.imax)]
                index += data.imax
                data.tmp2 = np.pad(data.tmp2, ((0, data.imax-data.mxdnse), (0, 0)), 'constant', constant_values=0)
                for j in range(data.imax):
                    data.tmp2[j, itp] = tmp2_values[j]
        if i < data.mxdwni:
            tmp2_values = values[index:index + data.mxdnsi * data.lnmax]
            index += data.mxdnsi * data.lnmax
            data.tmp2 = np.zeros((data.lnmax, data.mxdnsi), dtype=float)
            for j in range(data.lnmax):
                for m in range(data.mxdnsi):
                    data.tmp2[j, m] = tmp2_values[j * data.mxdnsi + m]
            data.tmp2 = data.tmp2.T
        
        mode = 2
        suppl1(data, i, mode)

        if i < data.mxdwn2:
            tmp2_values = values[index:index + data.mxdns2 * data.lnmax]
            index += data.mxdns2 * data.lnmax
            data.tmp2 = np.zeros((data.lnmax, data.mxdns2), dtype=float)
            for j in range(data.lnmax):
                for m in range(data.mxdns2):
                    data.tmp2[j, m] = tmp2_values[j * data.mxdns2 + m]
            data.tmp2 = data.tmp2.T
        
        mode = 3
        suppl1(data, i, mode)

        for m in range(data.lnmax):
            for mt in range(data.mxreac):
                mr = data.nr[mt, m]
                kt = data.msf[mt, m]
                itp = data.idnuc1[m]

                if mr == 0 or kt == 0:
                    continue
                
                if mt != 5:
                    tmp1_values = values[index:index + data.mxsig0 * kt * mr]
                    index += data.mxsig0 * kt * mr
                    data.tmp1 = np.zeros((data.mxsig0, kt, mr), dtype=float)
                    for j in range(data.mxsig0):
                        for k in range(kt):
                            for n in range(mr):
                                data.tmp1[j, k, n] = tmp1_values[j * kt * mr + k * mr + n]
                    ng = i
                    supl2(data, ng, kt, mr, m, mt)

                if mt == 5 and data.la[1, m] >= i+1:
                    tmp1_values = values[index:index + data.mxsig0 * kt * mr]
                    index += data.mxsig0 * kt * mr
                    data.tmp1 = np.zeros((data.mxsig0, kt, mr), dtype=float)
                    for j in range(data.mxsig0):
                        for k in range(kt):
                            for n in range(mr):
                                data.tmp1[j, k, n] = tmp1_values[j * kt * mr + k * mr + n]
                    ng = i
                    supl2(data, ng, kt, mr, m, mt)
    data.sig1d = np.stack(sig1d_array, axis=2)

def calsg2(data: ComclhmgData):
    # print('calsg2:', [data.anfu[i] for i in range(data.mclnuc)])
    # print([data.sigt[i] for i in range(data.mclnuc)])
    
    s0pv0 = 2.0 / data.rd[data.nfubnd - 1]
    s0pv2 = 2.0 * data.rd[data.nfubnd - 1] / (data.rd[data.mxrg - 1]**2 - data.rd[data.nfubnd - 1]**2)
    sige1 = s0pv2 / 4.0

    for lnuc in range(data.mxncfu):
        sigt1 = 0.0
        for ir in range(data.nfubnd):
            for inn in range(data.mnccel[ir]):
                if inn == lnuc:
                    continue
                id = data.idnc1[ir, inn]
                sigt1 += data.adnc[ir, inn] * data.volrg[ir] / data.volrgt * data.sigt[id]

        # print('test1')

        sigt2 = 0.0
        for ir in range(data.nfubnd, data.mxrg):
            for inn in range(data.mnccel[ir]):
                id = data.idnc1[ir, inn]
                sigt2 += data.adnc[ir, inn] * data.volrg[ir] / data.volrgt * data.sigt[id]

        # print('test2')

        fgam = 1.0 / (1.0 + sige1 / sigt2)
        fdanc = 1.0 - fgam - fgam**4 * (1.0 - fgam)
        # print('test3')

        corec = 0.25 * data.fbell * (1.0 - fdanc) / (1.0 + (data.fbell - 1.0) * fdanc) * s0pv0 / data.atden[lnuc] #check fbel or fbell
        data.sig0[lnuc] = sigt1 / data.atden[lnuc] + corec

        # print('calsg2:', lnuc, sigt1 / data.atden[lnuc], corec, data.sig0[lnuc])

def calsg1(data: ComclhmgData, ng, temp1):
    # print('Enter calsg0: mclnuc=', data.mclnuc)
    for ln in range(data.mclnuc):
        data.sig0[ln] = 0.0
        for i in range(data.mclnuc):
            if i == ln:
                continue
            id = data.idcl[i]
            id1 = data.idcl1[i]
            nreac = 3
            tpsigt = data.sig1d[id, 0, ng] + data.sig1d[id, 2, ng] + data.sig1d[id, 3, ng] +\
                     data.sig1d[id, 4, ng] + data.sig1d[id, 7, ng]
            tpsig0 = 1.0e5
            xval = inter1(data, ng, nreac, id, tpsig0, temp1)
            data.sig0[ln] += data.atden[i] * xval * tpsigt / data.atden[ln]

        id = data.idcl[ln]
        xval = inter1(data, ng, nreac, id, data.sig0[ln], temp1)
        tpsigt = data.sig1d[id, 0, ng] + data.sig1d[id, 2, ng] + data.sig1d[id, 3, ng] +\
                 data.sig1d[id, 4, ng] + data.sig1d[id, 7, ng]
        data.sigt[ln] = tpsigt * xval

        # print(f'sig0: initial step', [f'{data.sig0[i]}' for i in range(data.mclnuc)])
        # print(f'sigt: initial step', [f'{data.sigt[i]}' for i in range(data.mclnuc)])

    for lp1 in range(3):
        calsg2(data)
        for ln in range(data.mclnuc):
            id = data.idcl[ln]
            xval = inter1(data, ng, nreac, id, data.sig0[ln], temp1)
            tpsigt = data.sig1d[id, 0, ng] + data.sig1d[id, 2, ng] + data.sig1d[id, 3, ng] +\
                     data.sig1d[id, 4, ng] + data.sig1d[id, 7, ng]
            data.sigt[ln] = tpsigt * xval

        # print(f'sig0: {lp1}')
        # print([f'{data.sig0[i]}' for i in range(data.mclnuc)])
        # print(f'sigt: {lp1}')
        # print([f'{data.sigt[i]}' for i in range(data.mclnuc)])

def gauss3(a, b, nd, ndim):
    a = np.zeros((ndim, 3))
    b = np.zeros(ndim)

    a[0, 2] /= a[0, 1]
    b[0] /= a[0, 1]
    a[0, 1] = 1

    for j in range(1, nd):
        piv = a[j, 0]
        if piv != 0:
            for k in range(3):
                a[j, k] /= piv
            b[j] /= piv
            a[j, 1] -= a[j-1, 2]
            a[j, 0] -= a[j-1, 1]
            b[j] -= b[j - 1]
        b[j] /= a[j, 1]
        a[j, 2] /= a[j, 1]
        a[j, 1] = 1
    
    for j in range(nd - 2, -1, -1):
        if a[j, 2] != 0:
            b[j] /= a[j, 2] - b[j + 1]
            a[j, 1] /= a[j, 2]
            a[j, 2] = 0
        b[j] /= a[j, 1]
        a[j, 1] = 1

def spline(data: ComclhmgData):
    for i in range(1, data.maxdat - 1): #do 40 i=2,maxdat-1
        alfap = (data.x[i + 1] - data.x[i]) / 3.0
        alfam = (data.x[i] - data.x[i - 1]) / 3.0
        betap = (data.g[i + 1] - data.g[i]) / (data.x[i + 1] - data.x[i])
        betam = (data.g[i] - data.g[i - 1]) / (data.x[i] - data.x[i - 1])
        data.a[i - 1, 0] = alfam
        data.a[i - 1, 2] = alfap
        data.a[i - 1, 1] = (alfam + alfap) * 2.0
        data.s[i - 1] = betap - betam
    
    data.a[0, 0] = 0.0
    data.a[data.maxdat - 3, 2] = 0.0
    data.c[0] = ((data.g[2] - data.g[0]) / (data.x[2] - data.x[0]) - (data.g[1] - data.g[0]) /\
                 (data.x[1] - data.x[0])) / (data.x[2] - data.x[1])
    i = data.maxdat - 1
    data.c[i] = ((data.g[i] - data.g[i - 2]) / (data.x[i] - data.x[i - 2]) - (data.g[i - 1] - data.g[i - 2]) /\
                    (data.x[i - 1] - data.x[i - 2])) / (data.x[i] - data.x[i - 1])
    data.s[0] -= data.c[0] * (data.x[1] - data.x[0]) / 3.0
    data.s[data.maxdat - 3] -= data.c[data.maxdat - 1] * (data.x[data.maxdat - 1] - data.x[data.maxdat - 2]) / 3.0
    
    gauss3(data.a, data.s, data.maxdat - 2, 10) #check again #looks good

    for i in range(data.maxdat - 2):
        data.c[i + 1] = data.s[i]
    
    for i in range(data.maxdat - 1):
        data.d[i] = (data.c[i + 1] - data.c[i]) / (data.x[i + 1] - data.x[i]) / 3.0
        data.b[i] = (data.g[i + 1] - data.g[i]) / (data.x[i + 1] - data.x[i]) -\
                    (data.c[i + 1] + 2.0* data.c[i]) * (data.x[i + 1] - data.x[i]) / 3.0
    
    data.b[data.maxdat - 1] = data.b[data.maxdat - 2] + (data.c[data.maxdat - 2] + data.c[data.maxdat - 1]) *\
                              (data.x[data.maxdat - 1] - data.x[data.maxdat - 2])
    data.d[data.maxdat] = 0.0

def splan3(data: ComclhmgData):
    icount = 0
    while True:
        alfa1 = (data.x[0]**3 - data.x[1]**3) / (data.x[0] - data.x[1]) -\
                (data.x[2]**3 - data.x[3]**3) / (data.x[2] - data.x[3])
        alfa2 = (data.x[0]**2 - data.x[1]**2) / (data.x[0] - data.x[1]) -\
                (data.x[2]**2 - data.x[3]**2) / (data.x[2] - data.x[3])
        alfa3 = (data.g[0] - data.g[1]) / (data.g[0] - data.g[1]) -\
                (data.g[2] - data.g[3]) / (data.x[2] - data.x[3])
        beta1 = (data.x[0]**3 - data.x[2]**3) / (data.x[0] - data.x[2]) -\
                (data.x[1]**3 - data.x[3]**3) / (data.x[1] - data.x[3])
        beta2 = (data.x[0]**2 - data.x[2]**2) / (data.x[0] - data.x[2]) -\
                (data.x[1]**2 - data.x[3]**2) / (data.x[1] - data.x[3])
        beta3 = (data.g[0] - data.g[2]) / (data.x[0] - data.x[2]) -\
                (data.g[1] - data.g[3]) / (data.x[1] - data.x[3])

        if alfa2 != 0.0 and beta2 != 0.0:
            tem1 = (alfa3 / alfa2 - beta3 / beta2)
            tem2 = (alfa1 / alfa2 - beta1 / beta2)
            if tem2 == 0.0: #goto 100
                if icount > 4: #stop
                    raise Exception("Exceeded maximum iterations")
                icount += 1
                tmpx = data.x[1]
                tmpy = data.g[1]
                data.x[1], data.x[3], data.x[2] = data.x[3], data.x[2], tmpx
                data.g[1], data.g[3], data.g[2] = data.g[3], data.g[2], tmpy
                continue
            data.as_ = tem1 / tem2
            data.bs = alfa3 / alfa2 - alfa1 / alfa2 * data.as_
        else:
            if beta2 != 0.0:
                data.as_ = alfa3 / alfa1
                data.bs = beta3 / beta2 - beta1 / beta2 * data.as_ #goto 90
            elif alfa2 != 0.0:
                data.as_ = beta3 / beta1
                data.bs = alfa3 / alfa2 - alfa1 / alfa2 * data.as_

        data.cs = (data.g[0] - data.g[1]) / (data.x[0] - data.x[1]) - data.as_ * (data.x[0]**3 - data.x[1]**3) /\
                  (data.x[0] - data.x[1]) - data.bs * (data.x[0]**2 - data.x[1]**2) / (data.x[0] - data.x[1])
        data.ds = data.g[0] - data.as_ * data.x[0]**3 - data.bs * data.x[0]**2 - data.cs * data.x[0]

        data.ac[0] = data.as_
        data.ac[1] = data.bs
        data.ac[2] = data.cs
        data.ac[3] = data.ds
        break

def inter1(data: ComclhmgData, ng, nreac, nnuc, vsig0, temp1):
    for i in range(data.mxsig0):
        data.x[i] = np.log10(data.tab[i, nnuc])

    match = 0
    vlsig0 = np.log10(vsig0)
    nncomp = data.idnuc1[nnuc]

    for i in range(data.mxsig0 - 1):
        if vlsig0 < data.x[i + 1]:
            match = i
            break
    
    for j in range(data.msf[nreac, nnuc]):
        for i in range(data.mxsig0):
            data.g[i] = data.ftab[i, j, ng, nreac, nncomp]
            data.gg[i, j] = data.ftab[i, j, ng, nreac, nncomp]
        data.maxdat = data.mxsig0
        spline(data)
        for i in range(data.mxsig0):
            data.bg[i, j] = data.b[i]
            data.cg[i, j] = data.c[i]
            data.dg[i, j] = data.d[i]
        dxx = vlsig0 - data.x[match]
        data.vt[j] = data.gg[match, j] + data.bg[match, j] * dxx + data.cg[match, j] * dxx **2 +\
                     data.dg[match, j] * dxx ** 3
    
    for i in range(data.msf[nreac, nnuc]):
        data.tt[i] = data.ft[i, nnuc]
    
    # if(data.msf[nreac, nnuc] >= 3):
    #     splan3(data)
    #     fval = data.ds + data.cs * temp1 + data.bs * temp1 ** 2 + data.as_ * temp1 ** 3
    # else:
        # fval = data.vt[0]
    fval = data.vt[0] #skip splan3
    return fval

def scelas(data: ComclhmgData, ncl):
    irec = data.idcl1[ncl]
    nuc = data.idcl[ncl]
    aa = data.aw[nuc]
    # print(f'nucleid number :ncl,irec,nuc {ncl}, {irec}, {nuc}')
    alfa = ((aa - 1.0) / (aa + 1.0)) ** 2
    # print(f'atomic weight of the nucleus : {aa}')
    # print(f'Alfa=((A-1)/(A+1))**2 = {alfa}')

    for igs in range(data.imax):
        esup = data.enbnd[igs]
        eslow = data.enbnd[igs + 1]
        for igt in range(data.imax):
            icase = 6
            etup = data.enbnd[igt]
            etlow = data.enbnd[igt + 1]
            if igs > igt:
                icase = 6 #goto 25
            elif igt == igs:
                icase = 0 #goto 25
            elif alfa * esup <= etlow:
                icase = 1 #goto 25
            elif (alfa * esup <= etup) and (alfa * eslow <= etlow):
                icase = 2 
            elif (alfa * esup >= etup) and (alfa * eslow <= etlow):
                icase = 3
            elif (alfa * esup <= etup) and (alfa * eslow <= etup):
                icase = 4
            elif (alfa * esup >= etup) and (alfa * eslow < etup):
                icase = 5
            else:
                icase = 6

            sgstp = 0.0 #25
            if icase == 0:
                if alfa * esup < eslow:
                    sgstp = (esup - eslow) / (1.0 - alfa) * np.log(1.0 / alfa)
                else:
                    sgstp = (esup - eslow) / (1.0 - alfa) - eslow / (1.0 - alfa) * np.log(esup / eslow)
            elif icase == 1:
                sgstp = (etup - etlow) / (1.0 - alfa) * np.log(esup / eslow)
            elif icase == 2:
                sgstp = ((etup - etlow) / (1.0 - alfa) * np.log(etlow / (alfa * eslow)) +
                         (etup - alfa * esup) / (1.0 - alfa) * np.log(esup * alfa / etlow) +
                         alfa * esup / (1.0 - alfa) * np.log(esup * alfa / etlow) -
                         alfa / (1.0 - alfa) * (esup - etlow / alfa))
            elif icase == 3:
                sgstp = ((etup - etlow) / (1.0 - alfa) * np.log(etlow / (alfa * eslow)) +
                         etup / (1.0 - alfa) * np.log(etup / etlow) -
                         (etup - etlow) / (1.0 - alfa))
            elif icase == 4:
                sgstp = ((etup - alfa * esup) / (1.0 - alfa) * np.log(esup / eslow) +
                         alfa * esup / (1.0 - alfa) * np.log(esup / eslow) -
                         (esup - eslow) * alfa / (1.0 - alfa))
            elif icase == 5:
                sgstp = (etup / (1.0 - alfa) * np.log(etup / (alfa * eslow)) -
                         alfa / (1.0 - alfa) * (etup / alfa - eslow))
            
            sgstp = sgstp / (esup - eslow)
            data.stre[igs, irec, igt] = sgstp * data.sigel[ncl, igs]

def process_ig2(data, ig):
    a, b = data.sig0g.shape
    local_sig0g = np.zeros((a))
    local_sigtg = np.zeros((a))
    local_sigf = np.zeros((a))
    local_sigc = np.zeros((a))
    local_sigel = np.zeros((a))
    local_siger = np.zeros((a))
    local_sigin = np.zeros((a))
    local_sig2 = np.zeros((a))
    local_xnu = np.zeros((a))
    local_xmyu = np.zeros((a))
    local_siga = np.zeros((a))

    nreac = 3

    calsg1(data, ig, data.temper)

    for inn in range(data.mclnuc):
        idsp = data.idcl[inn]
        local_sig0g[inn] = data.sig0[inn]
        local_sigtg[inn] = data.sigt[inn]
        
        nreac = 0
        fval = inter1(data, ig, nreac, idsp, local_sig0g[inn], data.temper)
        local_sigf[inn] = fval * data.sig1d[idsp, 1 - 1, ig]

        nreac = 1
        fval = inter1(data, ig, nreac, idsp, local_sig0g[inn], data.temper)
        local_sigc[inn] = fval * data.sig1d[idsp, 3 - 1, ig]

        nreac = 2
        fval = inter1(data, ig, nreac, idsp, local_sig0g[inn], data.temper)
        local_sigel[inn] = fval * data.sig1d[idsp, 5 - 1, ig]

        nreac = 4
        fval = inter1(data, ig, nreac, idsp, local_sig0g[inn], data.temper)
        local_siger[inn] = fval * data.sig1d[idsp, 7 - 1, ig]

        nreac = 5
        fval = inter1(data, ig, nreac, idsp, local_sig0g[inn], data.temper)
        local_sigin[inn] = fval * data.sig1d[idsp, 4 - 1, ig]

        local_sig2[inn] = data.sig1d[idsp, 8 - 1, ig]
        local_xnu[inn] = data.sig1d[idsp, 2 - 1, ig]
        local_xmyu[inn] = data.sig1d[idsp, 6 - 1, ig]

        local_siga[inn] = local_sigc[inn] + local_sigf[inn]

    return ig, local_sig0g, local_sigtg, local_sigf, \
            local_sigc, local_sigel, local_siger, \
            local_sigin, local_sig2, local_xnu, \
            local_xmyu, local_siga

def homog(data: ComclhmgData, workers):
    for i in range(data.mclnuc):
        match = 0
        for j in range(data.lnmax):
            if data.ncod[i] == data.ncodel[j]:
                data.idcl[i] = j

    for i in range(data.mclnuc):
        itp = data.idcl[i]
        data.idcl1[i] = data.idnuc1[itp]
        # print(f"idcl1: {i}, {data.idcl1[i]}")

    # print("\nlist of temperature")
    # for i in range(data.lnmax):
    #     print(f"{i}, {data.ncodel[i]}", [f"{data.ft[j, i]:.5f}" for j in range(data.mxtemp)])

    # print("\nlist of sigma0 value")
    # for i in range(data.lnmax):
    #     print(f"{i}", [f"{data.tab[j, i]:.5f}" for j in range(data.mxsig0)])
    
    # print(f'ig,  vsig0,  temper,  fval')

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_ig2, data, ig) for ig in range(data.imax)]
        
        for future in as_completed(futures):
            ig, local_sig0g, local_sigtg, local_sigf, \
                local_sigc, local_sigel, local_siger, \
                local_sigin, local_sig2, local_xnu, \
                local_xmyu, local_siga = future.result()
            data.sig0g[:, ig] = local_sig0g
            data.sigtg[:, ig] = local_sigtg
            data.sigf[:, ig] = local_sigf
            data.sigc[:, ig] = local_sigc
            data.sigel[:, ig] = local_sigel
            data.siger[:, ig] = local_siger
            data.sigin[:, ig] = local_sigin
            data.sig2[:, ig] = local_sig2
            data.xnu[:, ig] = local_xnu
            data.xmyu[:, ig] = local_xmyu
            data.siga[:, ig] = local_siga

    # for i in range(data.mclnuc):
    #     print(f"fission cross section for: {data.ncod[i]}")
    #     print([f"{data.sigf[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"capture cross section for: {data.ncod[i]}")
    #     print([f"{data.sigc[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"elastic cross section for: {data.ncod[i]}")
    #     print([f"{data.sigel[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"elastic removal cross section for: {data.ncod[i]}")
    #     print([f"{data.siger[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"inelastic cross section for: {data.ncod[i]}")
    #     print([f"{data.sigin[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"n2n cross section for: {data.ncod[i]}")
    #     print([f"{data.sig2[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"total cross section for: {data.ncod[i]}")
    #     print([f"{data.sigtg[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"absorbtion cross section for: {data.ncod[i]}")
    #     print([f"{data.siga[i, j]:.5f}" for j in range(data.imax)])

    # for i in range(data.mclnuc):
    #     print(f"sig0 cross section for: {data.ncod[i]}")
    #     print([f"{data.sig0g[i, j]:.5f}" for j in range(data.imax)])

    for inn in range(data.mclnuc):
        scelas(data, inn)

    return data

def incoef(data: ComclhmgData):
    with open('D:\Materi Sains Komputasi\Tesis\gausquad.dat', 'r') as file:
        file.readline()
        file.readline()

        while True:
            line = file.readline().strip()
            if not line:
                break  # End of file

            if line.startswith('order of gauss integration'):
                order_line = file.readline().strip()
                norde = int(order_line)
                # print(f'norde= {norde}')

                file.readline()  # Read and ignore the line with 'number   absis , weight'

                for inn in range(norde):
                    line = file.readline().strip()
                    parts = line.split()
                    idum = int(parts[0])
                    data.absg[norde-2, inn] = float(parts[1])
                    data.weigg[norde-2, inn] = float(parts[2])
                    # print(idum, data.absg[norde-2, inn], data.weigg[norde-2, inn])

def xlmdk(data: ComclhmgData, ireg, ig, rho):
    delr = data.rrg[ireg]**2 - rho**2
    if ireg > 0:
        delr1 = data.rrg[ireg-1]**2 - rho**2
    else:
        delr1 = 0.0

    if delr1 > 0.0:
        temp1 = np.sqrt(delr1)
    else:
        temp1 = 0.0

    if delr > 0.0:
        tmp = np.sqrt(delr)
    else:
        tmp = 0.0

    xlmdk = (tmp - temp1) * data.sigtrg[ireg, ig]
    return xlmdk

def xlmds1(data: ComclhmgData, ii, ig, rho):
    tmp = 0.0
    for j in range(ii + 1, data.maxcrg):
        tmp += xlmdk(data, j, ig, rho)
    xlmds1 = tmp
    return xlmds1

def xlmds2(data: ComclhmgData, ii, ig, rho):
    tmp = 0.0
    for j in range(data.maxcrg):
        tmp += xlmdk(data, j, ig, rho)
    for i in range(ii):
        tmp += xlmdk(data, i, ig, rho)
    xlmds2 = tmp
    return xlmds2

def xlmd1(data: ComclhmgData, ii, jj, ig, rho):
    tmp = 0.0
    if ii > jj:
        for i in range(jj + 1, ii):
            tmp += xlmdk(data, i, ig, rho)
    else:
        for i in range(ii + 1, jj):
            tmp += xlmdk(data, i, ig, rho)
    xlmd1 = tmp
    return xlmd1

def xlmd2(data: ComclhmgData, ii, jj, ig, rho):
    tmp = 0.0
    for j in range(jj):
        tmp += xlmdk(data, j, ig, rho)
    for i in range(ii):
        tmp += xlmdk(data, i, ig, rho)
    xlmd2 = tmp
    return xlmd2

def xlmd3(data: ComclhmgData, ii, ig, rho):
    tmp = 0.0
    if ii != -1: #check again
        for j in range(ii):
            tmp += xlmdk(data, j, ig, rho)
    xlmd3 = tmp * 2.0
    return xlmd3

def xki3as(x):
    tmp1 = math.exp(-x) * math.sqrt(3.14159 / 2.0 / x)
    bb = -13.0 / 8.0
    cc = 3.0 * (16.0 * 9.0 + 72.0 + 3.0) / 2.0 / 64.0
    dd = -2.5 * (64.0 * 27.0 + 240.0 * 9.0 + 212.0 * 3.0 + 15.0) / 512.0
    tmp2 = 1.0 + bb / x + cc / x**2 + dd / x**3
    xki3as = tmp1 * tmp2
    return xki3as

def spfact(n, j):
    sptmp = 1.0
    for i in range(1, n + 1):
        sptmp *= i / 2.0
        if i <= j:
            sptmp /= i
        if i <= n - j:
            sptmp /= i
    spfact = sptmp
    return spfact

def xki3s(x):
    alfa = 5.0
    nmax = 10
    xkitmp = 0.0
    factj = 1.0
    tpsp1 = 1.0
    expalf = math.exp(alfa)
    
    xkitmp = 0.5 * math.sinh(alfa) / math.cosh(alfa)**2 + math.atan(expalf) - 3.14159 / 4.0
    xkitmp -= x * math.tanh(alfa)
    xkitmp += x**2 * (math.atan(expalf) - 3.14159 / 4.0) - x**3 * alfa / 6.0
    
    sign = -1.0
    fact = 6.0
    xpn = x**3
    
    for n in range(1, nmax + 1):
        fact *= (3 + n)
        xpn *= x
        np2 = int(n / 2) #EDIT
        tmp = xpn / fact * sign
        
        if n != 2 * np2:
            tmpint = 0.0
            for j in range(0, n + 1):
                tmpexp = alfa * (n - 2 * j)
                tmpint += spfact(n, j) * (math.exp(tmpexp) - 1.0) / (n - 2 * j)
            xkitmp += tmp * tmpint
        else:
            tmpint = 0.0
            for j in range(0, np2):
                tmpexp = alfa * (n - 2 * j)
                tmpint += spfact(n, j) * (math.exp(tmpexp) - math.exp(-tmpexp)) / (n - 2 * j)
            tmpint += spfact(n, np2) * alfa
            xkitmp += tmp * tmpint
        
        sign = -1.0 * sign
    xki3s = xkitmp
    return xki3s

def xki3(x, n):
    spl = 0.0
    xd2 = x / 2.0
    factj = 1.0
    tpsp1 = 1.0

    if x > 8.0:
        vxki3 = xki3as(x)
        xki3 = vxki3
        return xki3

    if x < 0.05:
        vxki3 = xki3s(x)
        xki3 = vxki3
        return xki3

    gamma = 0.577215664901532
    pi = 3.141592653589793

    vxki3 = 1.0 / 4.0 * pi - x + 0.25 * pi * x**2 - (11.0 / 6.0 - gamma) / 6.0 * x**3 +\
            x**3 / 6.0 * math.log(xd2)

    for j in range(1, n + 1):
        j2p3 = 2 * j + 3
        j2p2 = 2 * j + 2
        j2p1 = 2 * j + 1

        vj2p1 = float(j2p1)
        vj2p2 = float(j2p2)
        vj2p3 = float(j2p3)

        spl += 1.0 / float(j)
        tpsp1 *= xd2 / float(j)
        tpsp2 = tpsp1**2 * xd2**3 / (vj2p3 * vj2p2 * vj2p1)
        vxki3 += 8.0 * math.log(xd2) * tpsp2
        vxki3 += (gamma - 1.0 / vj2p1 - 1.0 / vj2p2 - 1.0 / vj2p3 - spl) * 8.0 * tpsp2
    
    xki3 = vxki3
    return xki3

def calcp0(data: ComclhmgData, ii, iord, ig):
    # print("\n----------calcp0() start----------")
    # print(f'enter calcp: {ii}, {iord}, {ig}')
    nbkf = 20
    xinteg = 0.0
    if ii > 0:
        rrm = data.rrg[ii - 1]

        for iteta in range(iord):
            rho = rrm / 2.0 * (data.absg[iord, iteta] + 1.0)
            xlamda = xlmdk(data, ii, ig, rho)
            xlamd1 = xlamda 
            xinteg += 2.0 * xlamda * data.weigg[iord, iteta] 
            xinteg += 2.0 * data.weigg[iord, iteta] * xki3(xlamda, nbkf)
            xlamda = 0.0
            xinteg -= data.weigg[iord, iteta] * np.pi / 2.0 
            xlamd3 = xlmd3(data, ii, ig, rho)
            xinteg += data.weigg[iord, iteta] * xki3(xlamd3, nbkf)
            xlamda = xlamd3 + xlamd1
            xinteg -= 2.0 * data.weigg[iord, iteta] * xki3(xlamda, nbkf)
            xlamda = xlamd3 + 2.0 * xlamd1
            xinteg += data.weigg[iord, iteta] * xki3(xlamda, nbkf)

        xinteg = xinteg * 2.0 / data.sigtrg[ii, ig] / data.volrrg[ii] * rrm / 2.0

    else:
        rrm = 0.0 #goto 45
    
    if ii == 0:
        rim = 0
        ri = data.rrg[0]
    else:
        rim = data.rrg[ii - 1]
        ri = data.rrg[ii]

    rav = (ri + rim) / 2.0
    delr = (ri - rim) / 2.0
    xsupl = 0.0

    for iteta in range(iord):
        rho = delr * (data.absg[iord, iteta] + 1.0) + rim
        xlamd1 = data.sigtrg[ii, ig] * np.sqrt(data.rrg[ii] ** 2 - rho ** 2)
        xlamda = xlamd1
        xsupl += 4.0 * xlamda * data.weigg[iord, iteta]
        xlamda = 2.0 * xlamd1
        xsupl += 2.0 * data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamda = 0.0
        xsupl -= data.weigg[iord, iteta] * np.pi / 2.0

    xsupl = xsupl / data.sigtrg[ii, ig] / data.volrrg[ii] * delr
    valcp = xinteg + xsupl
    # print(f'xinteg, xsupl, valcp = {xinteg}, {xsupl}, {valcp}, {ii}')
    # print("----------calcp0() end----------\n")
    return valcp

def calcp(data: ComclhmgData, ii, jj, iord, ig):
    # print("\n----------calcp() start----------")
    nbkf = 20
    if ii == jj:
        valcp = calcp0(data, ii, iord, ig)
        # print(f'calcp: {ii}, {jj}, {valcp}, {data.volrrg[ii]}')
        return valcp
    
    xinteg = 0.0
    for iteta in range(iord):
        rho = data.rrg[ii] / 2.0 * (data.absg[iord, iteta] + 1.0)
        xlamda = xlmd1(data, ii, jj, ig, rho) #EDIT
        xlamd1 = xlamda
        xinteg += data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamda = xlamd1 + xlmdk(data, ii, ig, rho)
        xinteg -= data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamda = xlamd1 + xlmdk(data, jj, ig, rho)
        xinteg -= data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamda = xlamd1 + xlmdk(data, ii, ig, rho) + xlmdk(data, jj, ig, rho)
        xinteg += data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        
        xlamda = xlmd2(data, ii, jj, ig, rho)
        xlamd2 = xlamda
        xinteg += data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamda = xlamd2 + xlmdk(data, ii, ig, rho)
        xinteg -= data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamda = xlamd2 + xlmdk(data, jj, ig, rho)
        xinteg -= data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamda = xlamd2 + xlmdk(data, ii, ig, rho) + xlmdk(data, jj, ig, rho)
        xinteg += data.weigg[iord, iteta] * xki3(xlamda, nbkf)
    
    valcp = xinteg * 2.0 / data.sigtrg[ii, ig] / data.volrrg[ii] * data.rrg[ii] / 2.0
    # print(f'calcp: {ii}, {jj}, {valcp}, {data.volrrg[ii]}')
    # print("----------calcp() end----------\n")
    return valcp

def escpr(data: ComclhmgData, ii, iord, ig):
    # print("\n----------escpr() start----------")
    nbkf = 20
    xinteg = 0.0
    
    for iteta in range(iord):
        rho = data.rrg[ii] / 2.0 * (data.absg[iord, iteta] + 1.0)
        xlamd1 = xlmds1(data, ii, ig, rho)
        xlamda = xlamd1
        xinteg += data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamda = xlamd1 + xlmdk(data, ii, ig, rho)
        xinteg -= data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamd2 = xlmds2(data, ii, ig, rho)
        xlamda = xlamd2
        xinteg += data.weigg[iord, iteta] * xki3(xlamda, nbkf)
        xlamda = xlamd2 + xlmdk(data, ii, ig, rho)
        xinteg -= data.weigg[iord, iteta] * xki3(xlamda, nbkf)

    escprb = xinteg * 2.0 / data.sigtrg[ii, ig] / data.volrrg[ii] * data.rrg[ii] / 2.0
    # print(f'finished escpr: {ii}, {escprb}, {data.volrrg[ii]}')
    # print("----------escpr() end----------\n")
    return escprb

def pijadj(data: ComclhmgData):
    print("\n----------pijadj() start----------")
    print("Collision probability after adjustment")
    surf = 2.0 * 3.14159 * data.rrg[data.maxcrg - 1]

    for ig in range(data.imax):
        data.q00[ig] = 1.0
        for irg in range(data.maxcrg):
            data.pescq[irg, ig] = (4.0 * data.volrrg[irg] / surf *
                                   data.sigtrg[irg, ig] * data.pesc[irg, ig])
            data.q00[ig] -= data.pescq[irg, ig]
        #     print(f"Q00: {irg}, {ig}, {data.pescq[irg, ig]:.5f}, {data.q00[ig]:.5f}")

        # print(f"{ig:5}, {data.q00[ig]:.5f}",
            #   [f"{data.pescq[i3, ig]:.5f}" for i3 in range(data.maxcrg)])

    for ig in range(data.imax):
        for ii in range(data.maxcrg):
            for jj in range(data.maxcrg):
                data.pij[ii, jj, ig] += (data.pesc[ii, ig] * data.pescq[jj, ig] /
                                         (1.0 - data.q00[ig]))
    
    for ig in range(data.imax):
        print(f"Energy group: {ig}")
        print("Escape probability pesc(i), i=1, maxcrg")
        print([f"{data.pesc[i3, ig]:.5f}" for i3 in range(data.maxcrg)])
        print("Collision prob. pij i:! j ->")

        for ii in range(data.maxcrg):
            print(f"{ii:5}",
                  [f"{data.pij[ii, i3, ig]:.5f}" for i3 in range(data.maxcrg)])
    print("----------pijadj() end----------\n")

def process_ig(data, ig):
        a, b, c = data.pij.shape
        d, e = data.pesc.shape
        local_pij = np.zeros((a, b))
        local_pesc = np.zeros((d))

        for ii in range(data.maxcrg):
            for jj in range(data.maxcrg):
                ii1 = ii
                jj1 = jj
                colprb = calcp(data, ii1, jj1, data.nord, ig)
                local_pij[ii, jj] = colprb
            escprb = escpr(data, ii, data.nord, ig)
            local_pesc[ii] = escprb
        
        return ig, local_pij, local_pesc

def colpr1(data: ComclhmgData, workers):
    # print("\n----------colpr1() start----------")
    ii = 0
    rb = 0.0

    for ireg in range(data.mxrg):
        if ireg > 0:
            dro = data.rd[ireg] - data.rd[ireg - 1]
        else:
            dro = data.rd[0]

        for i1 in range(data.mcpreg[ireg]):
            rbo = rb
            rb += dro / data.mcpreg[ireg]
            data.rrg[ii] = rb
            data.volrrg[ii] = 3.14159 * (rb ** 2 - rbo ** 2)
            data.idmat[ii] = ireg
            totxai = 0.0

            for ig in range(data.imax):
                data.sigtrg[ii, ig] = 0.0
                data.vsigf[ii, ig] = 0.0
                data.sigabs[ii, ig] = 0.0

                for i3 in range(data.imax):
                    data.scm[ii, ig, i3] = 0.0

                for inn in range(data.mnccel[ireg]):
                    id = data.idnc1[ireg, inn]
                    idsp = data.idcl1[id]
                    data.sigtrg[ii, ig] += data.adnc[ireg, inn] * data.sigtg[id, ig]
                    data.vsigf[ii, ig] += data.adnc[ireg, inn] * data.sigf[id, ig] * data.xnu[id, ig]
                    data.sigabs[ii, ig] += data.adnc[ireg, inn] * data.siga[id, ig]

                    for i3 in range(data.imax):
                        igsp = ig + i3
                        ndelta = ig - i3
                        if igsp <= data.imax and ig <= data.mxdwni:
                            data.scm[ii, ig, i3] += data.adnc[ireg, inn] * data.stri[ig, idsp, i3]
                        if igsp <= data.imax and ig <= data.mxdwn2:
                            data.scm[ii, ig, i3] += data.adnc[ireg, inn] * data.str2[ig, idsp, i3]

                data.xai[ii, ig] = data.chi[ig, data.nchi - 1]
                totxai += data.xai[ii, ig]
            

            # print(f'init. total xai for region: {ii}, {totxai}')

            for ig in range(data.imax):
                data.xai[ii, ig] /= totxai
            
            # Display scattering, absorption, and other metrics
            # print(f'sigtot of reg: {ii}')
            # print([f"{data.sigtrg[ii, j2]:.5f}" for j2 in range(data.imax)])

            # print(f'vsigf of reg: {ii}')
            # print([f"{data.vsigf[ii, j2]:.5f}" for j2 in range(data.imax)])

            # print(f'sigabs of reg: {ii}')
            # print([f"{data.sigabs[ii, j2]:.5f}" for j2 in range(data.imax)])

            # print(f'xai for reg: {ii}')
            # print([f"{data.xai[ii, j2]:.5f}" for j2 in range(data.imax)])
            # print([f"{data.adnc[ii, j2]:.5f}" for j2 in range(data.mclnuc)])

            # print(f'total scattering matrix of region: {ii}')

            # for ig in range(data.imax):
            #     print(f'source group: {ig}')
            #     print([f"{data.scm[ii, ig, igt]:.5f}" for igt in range(data.imax)])
            
            # print(f'self consistency chek:')

            for ig in range(data.imax):
                data.qss[ii, ig] = 0.0
                for igt in range(data.imax):
                    data.qss[ii, ig] += data.scm[ii, ig, igt]
            
            # print(f'total scattering:')
            # print([f"{data.qss[ii, igt]:.5f}" for igt in range(data.imax)])

            for ig in range(data.imax):
                data.qss[ii, ig] += data.sigabs[ii, ig]
                data.sigtrg[ii, ig] = data.qss[ii, ig]
            
            # print(f'total scattering+absorbtion:')
            # print([f"{data.qss[ii, igt]:.5f}" for igt in range(data.imax)])

            ii += 1

    data.maxcrg = ii
    # print(f'ir,(atdreg(ir,j),j=1,mclnuc)')
    for ir in range(data.maxcrg):
        for inn in range(data.mclnuc):
            data.atdreg[ir, inn] = 0.0
        id1 = data.idmat[ir]
        for inn in range(data.mnccel[id1]):
            id2 = data.idnc1[id1, inn]
            data.atdreg[ir, id2] = data.adnc[id1, inn]
        # print(f'{ir}', end=',')
        # print([f'{data.atdreg[ir, j]:.5f}' for j in range(data.mclnuc)])
    
    # print(f'radius of regions in cell {data.maxcrg}')
    # print([f"{data.rrg[i]:.5f}" for i in range(data.maxcrg)])
    # print(f'volume of regions in cell {data.maxcrg}')
    # print([f"{data.volrrg[i]:.5f}" for i in range(data.maxcrg)])

    incoef(data)

    # print('enter calcp')
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_ig, data, ig) for ig in range(data.imax)]
        
        for future in as_completed(futures):
            ig, local_pij, local_pesc = future.result()
            data.pij[:, :, ig] = local_pij
            data.pesc[:, ig] = local_pesc

    # print(f'ii,(pij(ii,i3,ig),i3=1,{data.maxcrg})')

    # for ig in range(data.imax):
        # print(f'energy group: {ig}')
        # print(f'escape probability pesc(i), i=1, {data.maxcrg}')
        # print([f"{data.pesc[i3, ig]:.5f}" for i3 in range(data.maxcrg)])
        # print('collision prob. pij i:! j ->')
        # for ii in range(data.maxcrg):
        #     print(f'{ii}', end=',')
        #     print([f"{data.pij[ii, i3, ig]:.5f}" for i3 in range(data.maxcrg)])

    pijadj(data)
    # print("----------colpr1() end----------\n")

# Run the colpr1 function with the prepared data
if __name__ == '__main__':
    workers = 1
    data = ComclhmgData()
    start = time.time()
    read1(r'D:\Materi Sains Komputasi\Tesis\tesinput.inp', data)
    timeread1 = time.time()-start
    start1 = time.time()
    rdlbsp(r'D:\Materi Sains Komputasi\Tesis\libcel.dat', data)
    timerdlbsp = time.time()-start1
    start2 = time.time()
    data = homog(data, workers)
    timehomog = time.time()-start2
    start3 = time.time()
    colpr1(data, workers)
    timecolpr1 = time.time()-start3
    print('timeread1 = ', timeread1)
    print('timerdlbsp = ', timerdlbsp)
    print('timehomog = ', timehomog)
    print('timecolpr1 = ', timecolpr1)
    print('time total = ', time.time() - start)